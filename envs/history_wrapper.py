"""Fixed-window action-observation history for feed-forward policies."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HistoryStackWrapper(gym.Wrapper):
    """
    Build [o_t, a_t-1, o_t-1, ..., a_t-k+1, o_t-k+1].

    At reset, every observation slot contains the initial observation
    and every past-action slot contains zeros.
    """

    def __init__(self, env, history_length=1):
        super().__init__(env)

        if history_length < 1:
            raise ValueError(
                "history_length must be at least one"
            )

        if not isinstance(
            env.observation_space,
            spaces.Box,
        ) or len(env.observation_space.shape) != 1:
            raise TypeError(
                "HistoryStackWrapper requires a flat Box "
                "observation space"
            )

        if not isinstance(
            env.action_space,
            spaces.Box,
        ) or len(env.action_space.shape) != 1:
            raise TypeError(
                "HistoryStackWrapper requires a flat Box "
                "action space"
            )

        self.history_length = int(history_length)
        self.observation_history = deque(
            maxlen=self.history_length
        )
        self.action_history = deque(
            maxlen=self.history_length - 1
        )

        low_parts = [env.observation_space.low]
        high_parts = [env.observation_space.high]

        for _ in range(self.history_length - 1):
            low_parts.extend(
                [
                    env.action_space.low,
                    env.observation_space.low,
                ]
            )
            high_parts.extend(
                [
                    env.action_space.high,
                    env.observation_space.high,
                ]
            )

        self.observation_space = spaces.Box(
            low=np.concatenate(low_parts).astype(
                np.float32
            ),
            high=np.concatenate(high_parts).astype(
                np.float32
            ),
            dtype=np.float32,
        )

    def _stack_history(self):
        if len(self.observation_history) != self.history_length:
            raise RuntimeError(
                "History is not initialized; call reset first"
            )

        parts = [self.observation_history[0]]

        for index in range(self.history_length - 1):
            parts.extend(
                [
                    self.action_history[index],
                    self.observation_history[index + 1],
                ]
            )

        observation = np.concatenate(parts).astype(
            np.float32,
            copy=False,
        )

        if not self.observation_space.contains(observation):
            raise RuntimeError(
                "Stacked observation falls outside its declared space"
            )

        return observation

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        self.observation_history.clear()
        self.action_history.clear()

        for _ in range(self.history_length):
            self.observation_history.append(
                observation.copy()
            )

        zero_action = np.zeros(
            self.action_space.shape,
            dtype=np.float32,
        )
        for _ in range(self.history_length - 1):
            self.action_history.append(
                zero_action.copy()
            )

        return self._stack_history(), info

    def step(self, action):
        action_array = np.asarray(
            action,
            dtype=np.float32,
        )
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                "Action shape does not match the wrapped environment: "
                f"{action_array.shape} != {self.action_space.shape}"
            )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action_array)

        self.observation_history.appendleft(
            np.asarray(
                observation,
                dtype=np.float32,
            ).copy()
        )
        self.action_history.appendleft(
            action_array.copy()
        )

        return (
            self._stack_history(),
            reward,
            terminated,
            truncated,
            info,
        )
