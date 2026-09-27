"""Small LeRobot-compatible facade for a project-local LIBERO BDDL task."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "contact_aware_pushing_numba"))

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

NOOP_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


class ProjectLiberoEnv:
    """Load one BDDL file and retain the LeRobot API subset used here.

    ``_env`` is LIBERO's ``OffScreenRenderEnv``, so legacy access through
    ``base_env._env.env`` remains valid.
    """

    def __init__(self, bddl_file: str | Path, image_size: int = 64, settle_steps: int = 10):
        self.bddl_file = str(Path(bddl_file).resolve())
        self.image_size = image_size
        self.settle_steps = settle_steps
        self._env: OffScreenRenderEnv | None = None

    def _ensure_env(self):
        if self._env is None:
            self._env = OffScreenRenderEnv(
                bddl_file_name=self.bddl_file,
                robots=["Panda"],
                controller="OSC_POSE",
                camera_names=["agentview"],
                camera_heights=self.image_size,
                camera_widths=self.image_size,
                control_freq=20,
                horizon=1000,
                hard_reset=True,
            )

    def _format_obs(self, raw_obs):
        return {
            "pixels": {"image": raw_obs["agentview_image"]},
            "robot_state": {"eef": {
                "pos": np.asarray(raw_obs["robot0_eef_pos"]),
                "quat": np.asarray(raw_obs["robot0_eef_quat"]),
            }},
        }

    def reset(self, seed=None):
        self._ensure_env()
        assert self._env is not None
        if seed is not None:
            self._env.seed(int(seed))
        raw_obs = self._env.reset()
        for _ in range(self.settle_steps):
            raw_obs, _, _, _ = self._env.step(NOOP_ACTION)
        for robot in self._env.robots:
            robot.controller.use_delta = True
        return self._format_obs(raw_obs), {"is_success": False}

    def step(self, action):
        assert self._env is not None
        raw_obs, reward, done, info = self._env.step(np.asarray(action, dtype=np.float32))
        success = bool(self._env.check_success())
        info = dict(info)
        info["is_success"] = success
        return self._format_obs(raw_obs), float(reward), bool(done or success), False, info

    def render(self):
        assert self._env is not None
        image = self._env.env._get_observations()["agentview_image"]
        return image[::-1, ::-1].copy()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
