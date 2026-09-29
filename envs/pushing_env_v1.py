"""Compact state/action Gymnasium wrapper for the clean LIBERO push task."""
from __future__ import annotations

from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from .custom_libero_env import ProjectLiberoEnv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BDDL_V1 = ROOT / "tasks" / "clean_tabletop_push_v1.bddl"


class LiberoPushingEnvV1(gym.Env):
    """Formal v1 benchmark for contact-aware tabletop pushing."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
            self,
            max_steps=300,
            action_scale=0.5,
            observation_size=64,
            bddl_file=DEFAULT_BDDL_V1,
            include_object_velocity=False,
            include_contact_force=False,
            friction_range=None,
            success_radius=0.06,
            object_speed_threshold=0.02,
            stable_steps_required=5,
            initial_yaw_range=(-0.50, 0.50),
            workspace_margin=0.02,
            workspace_z_tolerance=0.05,
            render_mode="rgb_array",
    ):
        super().__init__()

        if workspace_margin < 0:
            raise ValueError(
                "workspace_margin cannot be negative."
            )

        if workspace_z_tolerance < 0:
            raise ValueError(
                "workspace_z_tolerance cannot be negative."
            )

        self.workspace_margin = float(workspace_margin)
        self.workspace_z_tolerance = float(
            workspace_z_tolerance
        )

        if success_radius <= 0:
            raise ValueError("success_radius must be positive.")

        if object_speed_threshold < 0:
            raise ValueError(
                "object_speed_threshold cannot be negative."
            )

        if stable_steps_required < 1:
            raise ValueError(
                "stable_steps_required must be at least one."
            )

        if len(initial_yaw_range) != 2:
            raise ValueError(
                "initial_yaw_range must contain two values."
            )
        yaw_min, yaw_max = map(float, initial_yaw_range)
        if not np.isfinite([yaw_min, yaw_max]).all():
            raise ValueError(
                "initial_yaw_range values must be finite."
            )
        if yaw_min > yaw_max:
            raise ValueError(
                "initial_yaw_range minimum cannot exceed maximum."
            )

        self.max_steps = int(max_steps)
        self.action_scale = float(action_scale)
        self.include_object_velocity = bool(
            include_object_velocity
        )
        self.include_contact_force = bool(
            include_contact_force
        )
        self.friction_range = friction_range
        self.success_radius = float(success_radius)
        self.object_speed_threshold = float(
            object_speed_threshold
        )
        self.stable_steps_required = int(
            stable_steps_required
        )
        self.initial_yaw_range = (yaw_min, yaw_max)
        self.current_initial_yaw = None
        self.current_initial_yaw_offset = None
        self.render_mode = render_mode

        self.step_count = 0
        self.stable_count = 0

        obs_dim = (
                6
                + 3 * self.include_object_velocity
                + 3 * self.include_contact_force
        )

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            (3,),
            dtype=np.float32,
        )

        self.base_env = ProjectLiberoEnv(
            bddl_file,
            observation_size,
        )

        self.prev_goal_distance = None
        self.prev_reach_distance = None
        self.current_friction = None
        self.requested_friction = None
        self.current_contact_friction = None

    def _get_libero_task_env(self):
        if self.base_env._env is None:
            raise RuntimeError("LIBERO environment has not been reset yet")
        return self.base_env._env.env

    def _get_position(self, name):
        state = self._get_libero_task_env().object_states_dict[name].get_geom_state()
        return np.asarray(state["pos"], dtype=np.float32).copy()

    def _get_plate_position(self):
        return self._get_position("plate_1")

    def _get_goal_position(self):
        return self._get_position("main_table_push_target_region")

    def _get_eef_position(self, obs):
        return np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float32).copy()

    def _get_plate_velocity(self):
        env = self._get_libero_task_env()
        body_name = env.objects_dict["plate_1"].root_body
        return np.asarray(env.sim.data.get_body_xvelp(body_name), dtype=np.float32).copy()

    def _get_plate_joint_name(self):
        env = self._get_libero_task_env()
        joints = env.objects_dict["plate_1"].joints
        if not joints:
            raise RuntimeError("Plate object has no free joint")
        return joints[0]

    def _get_plate_world_yaw(self):
        env = self._get_libero_task_env()
        body_name = env.objects_dict["plate_1"].root_body
        w, x, y, z = np.asarray(
            env.sim.data.get_body_xquat(body_name),
            dtype=np.float64,
        )
        return float(
            np.arctan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        )

    def _randomize_plate_yaw(self):
        """Apply world-Z yaw because this LIBERO ignores BDDL plate yaw."""
        env = self._get_libero_task_env()
        sim = env.sim
        joint_name = self._get_plate_joint_name()
        joint_qpos = np.asarray(
            sim.data.get_joint_qpos(joint_name),
            dtype=np.float64,
        ).copy()
        if joint_qpos.shape != (7,):
            raise RuntimeError(
                "Expected a 7D free-joint qpos for plate_1, "
                f"got {joint_qpos.shape}"
            )

        yaw_offset = float(
            self.np_random.uniform(*self.initial_yaw_range)
        )
        half_yaw = 0.5 * yaw_offset
        yaw_quaternion = np.asarray(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(
            quaternion,
            yaw_quaternion,
            joint_qpos[3:],
        )
        quaternion /= np.linalg.norm(quaternion)
        joint_qpos[3:] = quaternion
        sim.data.set_joint_qpos(joint_name, joint_qpos)
        sim.forward()

        self.current_initial_yaw_offset = yaw_offset
        self.current_initial_yaw = self._get_plate_world_yaw()

    def _update_stable_success(
            self,
            distance_success,
            object_speed,
    ):
        stable_now = bool(
            distance_success
            and object_speed <= self.object_speed_threshold
        )

        if stable_now:
            self.stable_count += 1
        else:
            self.stable_count = 0

        return bool(
            self.stable_count >= self.stable_steps_required
        )

    @staticmethod
    def _values_are_finite(*values):
        return all(
            np.isfinite(np.asarray(value)).all()
            for value in values
        )

    def _is_plate_in_workspace(self, plate_position):
        env = self._get_libero_task_env()
        sim = env.sim

        table_geom_id = sim.model.geom_name2id(
            "table_collision"
        )

        table_center = np.asarray(
            sim.data.geom_xpos[table_geom_id],
            dtype=np.float64,
        )
        table_half_size = np.asarray(
            sim.model.geom_size[table_geom_id],
            dtype=np.float64,
        )

        allowed_half_size = (
                table_half_size[:2] - self.workspace_margin
        )

        if np.any(allowed_half_size <= 0):
            raise ValueError(
                "workspace_margin is too large for the table."
            )

        xy_offset = np.abs(
            np.asarray(plate_position[:2])
            - table_center[:2]
        )
        xy_inside = bool(
            np.all(xy_offset <= allowed_half_size)
        )

        table_top = (
                table_center[2] + table_half_size[2]
        )
        z_above_limit = bool(
            plate_position[2]
            >= table_top - self.workspace_z_tolerance
        )

        return bool(xy_inside and z_above_limit)

    @staticmethod
    def _select_termination_reason(
            *,
            invalid_state,
            out_of_workspace,
            stable_success,
            time_limit,
    ):
        if invalid_state:
            return "invalid_state"

        if out_of_workspace:
            return "out_of_workspace"

        if stable_success:
            return "stable_success"

        if time_limit:
            return "time_limit"

        return None

    def _get_contact_force(self):
        """Net world-frame plate contact force, for optional future ablations."""
        env, wrench, net = self._get_libero_task_env(), np.zeros(6), np.zeros(3)
        sim, plate_body = env.sim, env.obj_body_id["plate_1"]
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            body1, body2 = sim.model.geom_bodyid[contact.geom1], sim.model.geom_bodyid[contact.geom2]
            if body1 != plate_body and body2 != plate_body:
                continue
            mujoco.mj_contactForce(sim.model._model, sim.data._data, i, wrench)
            force = contact.frame.reshape(3, 3).T @ wrench[:3]
            net += force if body1 == plate_body else -force
        return net.astype(np.float32)

    def _get_plate_table_contact_frictions(self):
        env = self._get_libero_task_env()
        sim = env.sim

        plate_body_id = env.obj_body_id["plate_1"]
        table_geom_id = sim.model.geom_name2id(
            "table_collision"
        )

        values = []

        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]

            body1 = sim.model.geom_bodyid[contact.geom1]
            body2 = sim.model.geom_bodyid[contact.geom2]

            is_plate_table = (
                                     body1 == plate_body_id
                                     and contact.geom2 == table_geom_id
                             ) or (
                                     body2 == plate_body_id
                                     and contact.geom1 == table_geom_id
                             )

            if is_plate_table:
                values.append(float(contact.friction[0]))

        return np.asarray(values, dtype=np.float64)

    def _set_surface_friction(self, friction):
        env = self._get_libero_task_env()

        table_geom_id = env.sim.model.geom_name2id(
            "table_collision"
        )

        # Randomize sliding friction only.
        env.sim.model.geom_friction[
            table_geom_id, 0
        ] = float(friction)

        # Ensure the table coefficient wins when MuJoCo
        # combines plate-table contact parameters.
        env.sim.model.geom_priority[
            table_geom_id
        ] = 1

        env.sim.forward()

        requested = float(friction)
        actual_values = self._get_plate_table_contact_frictions()

        if actual_values.size == 0:
            raise RuntimeError(
                "No plate-table contacts found after setting friction"
            )

        if not np.allclose(
                actual_values,
                requested,
                atol=1e-6,
        ):
            raise RuntimeError(
                "Contact friction mismatch: "
                f"requested={requested}, "
                f"actual={actual_values.tolist()}"
            )

        actual = float(actual_values.mean())

        self.requested_friction = requested
        self.current_contact_friction = actual
        self.current_friction = actual

    def _build_observation(self, obs):
        eef, plate, goal = self._get_eef_position(obs), self._get_plate_position(), self._get_goal_position()
        parts = [plate - eef, goal - plate]
        if self.include_object_velocity:
            parts.append(self._get_plate_velocity())
        if self.include_contact_force:
            parts.append(self._get_contact_force())
        return np.concatenate(parts).astype(np.float32)

    def _compute_distances(self, obs):
        eef, plate, goal = self._get_eef_position(obs), self._get_plate_position(), self._get_goal_position()
        return float(np.linalg.norm(plate - eef)), float(np.linalg.norm(goal[:2] - plate[:2]))

    def _convert_action(self, action):
        xyz = self.action_scale * np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        return np.array([*xyz, 0, 0, 0, -1], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is None:
            seed = int(
                self.np_random.integers(0, 2 ** 31 - 1)
            )

        obs, _ = self.base_env.reset(seed)

        self.step_count = 0
        self.stable_count = 0

        if self.friction_range is not None:
            friction = self.np_random.uniform(
                *self.friction_range
            )
            self._set_surface_friction(friction)
        else:
            self.requested_friction = None

            actual_values = (
                self._get_plate_table_contact_frictions()
            )

            self.current_contact_friction = (
                float(actual_values.mean())
                if actual_values.size
                else None
            )
            self.current_friction = (
                self.current_contact_friction
            )

        # Keep friction sampling first so existing seed-to-friction
        # mappings remain stable when yaw randomization is introduced.
        self._randomize_plate_yaw()

        self.prev_reach_distance, self.prev_goal_distance = (
            self._compute_distances(obs)
        )

        observation = self._build_observation(obs)
        plate_position = self._get_plate_position()
        object_velocity = self._get_plate_velocity()
        object_speed = float(
            np.linalg.norm(object_velocity)
        )

        initial_state_valid = self._values_are_finite(
            observation,
            plate_position,
            object_velocity,
            self.prev_reach_distance,
            self.prev_goal_distance,
        )

        if not initial_state_valid:
            raise RuntimeError(
                "Invalid numerical state immediately after reset."
            )

        if not self._is_plate_in_workspace(plate_position):
            raise RuntimeError(
                "Plate starts outside the valid workspace."
            )
        distance_success = bool(
            self.prev_goal_distance <= self.success_radius
        )

        info = {
            # 兼容已有训练和评估脚本，但在 v1 中只表示稳定成功。
            "is_success": False,
            "libero_success": False,
            "distance_success": distance_success,
            "stable_success": False,
            "stable_count": 0,
            "object_speed": object_speed,
            "initial_plate_x": float(plate_position[0]),
            "initial_plate_y": float(plate_position[1]),
            "initial_plate_z": float(plate_position[2]),
            "initial_plate_yaw": self.current_initial_yaw,
            "initial_plate_yaw_offset": (
                self.current_initial_yaw_offset
            ),
            "reach_distance": self.prev_reach_distance,
            "goal_distance": self.prev_goal_distance,
            "friction": self.current_friction,
            "requested_friction": self.requested_friction,
            "contact_friction": (
                self.current_contact_friction
            ),
            "termination_reason": None,
            "invalid_state": False,
            "out_of_workspace": False,
            "time_limit": False,
        }

        return observation, info

    def step(self, action):
        self.step_count += 1

        action_array = np.asarray(
            action,
            dtype=np.float32,
        )

        obs, sparse_reward, base_done, _, base_info = (
            self.base_env.step(
                self._convert_action(action_array)
            )
        )

        reach, goal = self._compute_distances(obs)
        plate_position = self._get_plate_position()
        object_velocity = self._get_plate_velocity()
        object_speed = float(
            np.linalg.norm(object_velocity)
        )
        observation = self._build_observation(obs)

        reach_progress = (
                self.prev_reach_distance - reach
        )
        goal_progress = (
                self.prev_goal_distance - goal
        )
        action_penalty = 0.001 * float(
            np.square(action_array).sum()
        )

        raw_reward = (
                5 * reach_progress
                + 50 * goal_progress
                - action_penalty
                - 0.0005
        )

        libero_success = bool(
            base_info.get("is_success", False)
        )
        distance_success = bool(
            np.isfinite(goal)
            and goal <= self.success_radius
        )

        invalid_state = not self._values_are_finite(
            action_array,
            observation,
            plate_position,
            object_velocity,
            reach,
            goal,
            raw_reward,
        )

        out_of_workspace = bool(
            not invalid_state
            and not self._is_plate_in_workspace(
                plate_position
            )
        )

        if invalid_state or out_of_workspace:
            self.stable_count = 0
            stable_success = False
        else:
            stable_success = self._update_stable_success(
                distance_success,
                object_speed,
            )

        time_limit = bool(
            self.step_count >= self.max_steps
        )

        termination_reason = (
            self._select_termination_reason(
                invalid_state=invalid_state,
                out_of_workspace=out_of_workspace,
                stable_success=stable_success,
                time_limit=time_limit,
            )
        )

        terminated = termination_reason in {
            "invalid_state",
            "out_of_workspace",
            "stable_success",
        }
        truncated = termination_reason == "time_limit"

        # 避免异常数值继续进入 PPO。
        if invalid_state:
            observation = np.nan_to_num(
                observation,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            reward = -10.0
        else:
            reward = raw_reward

        if stable_success:
            reward += 10.0

        if np.isfinite(reach):
            self.prev_reach_distance = reach

        if np.isfinite(goal):
            self.prev_goal_distance = goal

        info = {
            "is_success": stable_success,
            "libero_success": libero_success,
            "distance_success": distance_success,
            "stable_success": stable_success,
            "stable_count": self.stable_count,
            "object_speed": (
                object_speed
                if np.isfinite(object_speed)
                else None
            ),
            "termination_reason": termination_reason,
            "invalid_state": invalid_state,
            "out_of_workspace": out_of_workspace,
            "time_limit": time_limit,
            "reach_distance": (
                reach if np.isfinite(reach) else None
            ),
            "goal_distance": (
                goal if np.isfinite(goal) else None
            ),
            "reach_progress": (
                reach_progress
                if np.isfinite(reach_progress)
                else None
            ),
            "goal_progress": (
                goal_progress
                if np.isfinite(goal_progress)
                else None
            ),
            "action_penalty": (
                action_penalty
                if np.isfinite(action_penalty)
                else None
            ),
            "libero_sparse_reward": (
                sparse_reward
                if np.isfinite(sparse_reward)
                else None
            ),
            "base_env_done": bool(base_done),
            "friction": self.current_friction,
            "requested_friction": self.requested_friction,
            "contact_friction": (
                self.current_contact_friction
            ),
        }

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
