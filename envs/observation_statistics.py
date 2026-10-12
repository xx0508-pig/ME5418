"""Collect per-channel scale statistics without changing observations."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ObservationStatisticsWrapper(gym.Wrapper):
    """Record base observations for later scale diagnostics."""

    def __init__(self, env, channel_names):
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "ObservationStatisticsWrapper requires a Box "
                "observation space."
            )
        if len(env.observation_space.shape) != 1:
            raise TypeError(
                "ObservationStatisticsWrapper requires flat "
                "observations."
            )

        self.channel_names = tuple(channel_names)
        observation_dimension = env.observation_space.shape[0]
        if len(self.channel_names) != observation_dimension:
            raise ValueError(
                "Channel-name count does not match observation "
                f"dimension: {len(self.channel_names)} != "
                f"{observation_dimension}."
            )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Observation channel names must be unique.")

        self._observations = []

    def _record(self, observation):
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != self.observation_space.shape:
            raise ValueError(
                "Observation shape changed while collecting statistics: "
                f"{values.shape} != {self.observation_space.shape}."
            )
        self._observations.append(values.copy())

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._record(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        self._record(observation)
        return observation, reward, terminated, truncated, info

    def clear_statistics(self):
        self._observations.clear()

    @property
    def sample_count(self):
        return len(self._observations)

    def summary(self):
        if not self._observations:
            raise RuntimeError(
                "No observations have been recorded. Reset and step "
                "the environment before requesting statistics."
            )

        values = np.vstack(self._observations)
        rows = []
        for index, name in enumerate(self.channel_names):
            channel = values[:, index]
            finite = channel[np.isfinite(channel)]
            if finite.size == 0:
                row = {
                    "index": index,
                    "channel": name,
                    "sample_count": int(channel.size),
                    "finite_count": 0,
                    "non_finite_count": int(channel.size),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "abs_max": None,
                    "p01": None,
                    "p50": None,
                    "p99": None,
                }
            else:
                percentiles = np.percentile(finite, [1, 50, 99])
                row = {
                    "index": index,
                    "channel": name,
                    "sample_count": int(channel.size),
                    "finite_count": int(finite.size),
                    "non_finite_count": int(
                        channel.size - finite.size
                    ),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "abs_max": float(np.max(np.abs(finite))),
                    "p01": float(percentiles[0]),
                    "p50": float(percentiles[1]),
                    "p99": float(percentiles[2]),
                }
            rows.append(row)

        return {
            "sample_count": int(values.shape[0]),
            "observation_dimension": int(values.shape[1]),
            "channels": rows,
        }
