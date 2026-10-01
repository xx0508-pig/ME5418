"""Compact state/action Gymnasium wrapper for the clean LIBERO push task."""
from __future__ import annotations

from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from .custom_libero_env import ProjectLiberoEnv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BDDL = ROOT / "tasks" / "clean_tabletop_push.bddl"


class LiberoPushingEnv(gym.Env):
    """Panda + table + plate + target, with the existing PPO-compatible I/O."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, max_steps=300, action_scale=0.5, observation_size=64,
                 bddl_file=DEFAULT_BDDL, include_object_velocity=False,
                 include_contact_force=False, contact_force_components=3,
                 friction_range=None,
                 success_radius=0.06, render_mode="rgb_array"):
        super().__init__()
        self.max_steps, self.action_scale = int(max_steps), float(action_scale)
        self.include_object_velocity = bool(include_object_velocity)
        self.include_contact_force = bool(include_contact_force)
        requested_force_components = int(contact_force_components)
        if self.include_contact_force:
            if requested_force_components not in {2, 3}:
                raise ValueError(
                    "contact_force_components must be 2 or 3 "
                    "when contact force is enabled"
                )
            self.contact_force_components = (
                requested_force_components
            )
        else:
            self.contact_force_components = 0
        self.friction_range, self.success_radius = friction_range, float(success_radius)
        self.render_mode = render_mode
        self.step_count = 0
        obs_dim = (
            6
            + 3 * self.include_object_velocity
            + self.contact_force_components
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.base_env = ProjectLiberoEnv(bddl_file, observation_size)
        self.prev_goal_distance = self.prev_reach_distance = None
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
            parts.append(
                self._get_contact_force()[
                    :self.contact_force_components
                ]
            )
        return np.concatenate(parts).astype(np.float32)

    def _compute_distances(self, obs):
        eef, plate, goal = self._get_eef_position(obs), self._get_plate_position(), self._get_goal_position()
        return float(np.linalg.norm(plate - eef)), float(np.linalg.norm(goal[:2] - plate[:2]))

    def _convert_action(self, action):
        xyz = self.action_scale * np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        return np.array([*xyz, 0, 0, 0, -1], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        seed = int(self.np_random.integers(0, 2**31 - 1)) if seed is None else seed
        obs, _ = self.base_env.reset(seed)
        self.step_count = 0
        if self.friction_range is not None:
            self._set_surface_friction(self.np_random.uniform(*self.friction_range))
        else:
            self.requested_friction = None

            actual_values = self._get_plate_table_contact_frictions()

            self.current_contact_friction = (
                float(actual_values.mean())
                if actual_values.size
                else None
            )
            self.current_friction = self.current_contact_friction
        self.prev_reach_distance, self.prev_goal_distance = self._compute_distances(obs)
        info = {"is_success": False, "reach_distance": self.prev_reach_distance,
                "goal_distance": self.prev_goal_distance, "friction": self.current_friction,
                "requested_friction": self.requested_friction,"contact_friction": self.current_contact_friction,
                }
        return self._build_observation(obs), info

    def step(self, action):
        self.step_count += 1
        obs, sparse_reward, base_done, _, base_info = self.base_env.step(self._convert_action(action))
        reach, goal = self._compute_distances(obs)
        reach_progress, goal_progress = self.prev_reach_distance - reach, self.prev_goal_distance - goal
        action_penalty = 0.001 * float(np.square(action).sum())
        reward = 5 * reach_progress + 50 * goal_progress - action_penalty - 0.0005
        success = bool(base_info.get("is_success", False) or goal <= self.success_radius)
        reward += 10.0 if success else 0.0
        self.prev_reach_distance, self.prev_goal_distance = reach, goal
        info = {"is_success": success, "reach_distance": reach, "goal_distance": goal,
                "reach_progress": reach_progress, "goal_progress": goal_progress,
                "action_penalty": action_penalty, "libero_sparse_reward": sparse_reward,
                "friction": self.current_friction,"requested_friction": self.requested_friction,
                "contact_friction": self.current_contact_friction,}
        return self._build_observation(obs), float(reward), success or base_done, self.step_count >= self.max_steps, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
