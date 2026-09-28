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
            render_mode="rgb_array",
    ):
        super().__init__()

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

        self.prev_reach_distance, self.prev_goal_distance = (
            self._compute_distances(obs)
        )

        object_speed = float(
            np.linalg.norm(self._get_plate_velocity())
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
            "reach_distance": self.prev_reach_distance,
            "goal_distance": self.prev_goal_distance,
            "friction": self.current_friction,
            "requested_friction": self.requested_friction,
            "contact_friction": (
                self.current_contact_friction
            ),
        }

        return self._build_observation(obs), info

    def step(self, action):
        self.step_count += 1

        obs, sparse_reward, base_done, _, base_info = (
            self.base_env.step(
                self._convert_action(action)
            )
        )

        reach, goal = self._compute_distances(obs)

        reach_progress = (
                self.prev_reach_distance - reach
        )
        goal_progress = (
                self.prev_goal_distance - goal
        )

        action_penalty = 0.001 * float(
            np.square(action).sum()
        )

        reward = (
                5 * reach_progress
                + 50 * goal_progress
                - action_penalty
                - 0.0005
        )

        libero_success = bool(
            base_info.get("is_success", False)
        )
        distance_success = bool(
            goal <= self.success_radius
        )

        object_speed = float(
            np.linalg.norm(self._get_plate_velocity())
        )

        stable_success = self._update_stable_success(
            distance_success,
            object_speed,
        )

        # v1 只在 stable_success 时给予成功奖励。
        if stable_success:
            reward += 10.0

        self.prev_reach_distance = reach
        self.prev_goal_distance = goal

        terminated = stable_success

        # 如果刚好在第 300 步成功，优先算成功，而不是超时。
        truncated = bool(
            self.step_count >= self.max_steps
            and not stable_success
        )

        info = {
            # v1 中 is_success 是 stable_success 的兼容别名。
            "is_success": stable_success,
            "libero_success": libero_success,
            "distance_success": distance_success,
            "stable_success": stable_success,
            "stable_count": self.stable_count,
            "object_speed": object_speed,
            "reach_distance": reach,
            "goal_distance": goal,
            "reach_progress": reach_progress,
            "goal_progress": goal_progress,
            "action_penalty": action_penalty,
            "libero_sparse_reward": sparse_reward,
            "base_env_done": bool(base_done),
            "friction": self.current_friction,
            "requested_friction": self.requested_friction,
            "contact_friction": (
                self.current_contact_friction
            ),
        }

        return (
            self._build_observation(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
