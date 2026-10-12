"""Deterministic, versioned normalization for base observations."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .observation_modes import get_observation_channel_names


NORMALIZATION_NONE = "none"
NORMALIZATION_FIXED_V1 = "fixed_v1"
NORMALIZATION_CHOICES = (
    NORMALIZATION_NONE,
    NORMALIZATION_FIXED_V1,
)

# These constants come from development rollouts and conservative physical
# ranges. Zero remains zero; clipping only limits exceptional contact spikes.
FIXED_V1_GROUP_SCALES = {
    "position": 0.30,
    "velocity": 0.20,
    "contact_force_xy": 0.10,
    "contact_force_z": 10.0,
}
FIXED_V1_CLIP = 5.0


def fixed_v1_scales(observation_mode):
    scales = []
    for channel in get_observation_channel_names(observation_mode):
        if channel.startswith(("plate_minus_eef_", "goal_minus_plate_")):
            scale = FIXED_V1_GROUP_SCALES["position"]
        elif channel.startswith("plate_velocity_"):
            scale = FIXED_V1_GROUP_SCALES["velocity"]
        elif channel in {"contact_force_x", "contact_force_y"}:
            scale = FIXED_V1_GROUP_SCALES["contact_force_xy"]
        elif channel == "contact_force_z":
            scale = FIXED_V1_GROUP_SCALES["contact_force_z"]
        else:
            raise ValueError(f"No fixed scale for channel {channel!r}.")
        scales.append(scale)
    return np.asarray(scales, dtype=np.float32)


def normalization_metadata(name, observation_mode):
    if name == NORMALIZATION_NONE:
        return {"name": name, "enabled": False}
    if name != NORMALIZATION_FIXED_V1:
        raise ValueError(f"Unknown normalization {name!r}.")
    channels = get_observation_channel_names(observation_mode)
    scales = fixed_v1_scales(observation_mode)
    return {
        "name": name,
        "enabled": True,
        "method": "divide_by_fixed_scale_then_clip",
        "clip": FIXED_V1_CLIP,
        "preserves_zero": True,
        "channels": {
            channel: {"center": 0.0, "scale": round(float(scale), 8)}
            for channel, scale in zip(channels, scales)
        },
    }


class FixedObservationNormalizationWrapper(gym.ObservationWrapper):
    """Scale one base observation frame before history stacking."""

    def __init__(self, env, observation_mode, clip=FIXED_V1_CLIP):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("Fixed normalization requires a Box space.")
        if len(env.observation_space.shape) != 1:
            raise TypeError("Fixed normalization requires flat observations.")
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive.")

        self.observation_mode = observation_mode
        self.scales = fixed_v1_scales(observation_mode)
        self.clip = float(clip)
        if self.scales.shape != env.observation_space.shape:
            raise ValueError(
                "Normalization scale dimension does not match observation "
                f"space: {self.scales.shape} != {env.observation_space.shape}."
            )
        self.observation_space = spaces.Box(
            low=-self.clip,
            high=self.clip,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        values = np.asarray(observation, dtype=np.float32)
        normalized = np.clip(
            values / self.scales,
            -self.clip,
            self.clip,
        ).astype(np.float32, copy=False)
        if not np.isfinite(normalized).all():
            raise RuntimeError("Normalized observation contains NaN or Inf.")
        return normalized


def apply_observation_normalization(env, name, observation_mode):
    if name == NORMALIZATION_NONE:
        return env
    if name == NORMALIZATION_FIXED_V1:
        return FixedObservationNormalizationWrapper(env, observation_mode)
    raise ValueError(f"Unknown normalization {name!r}.")
