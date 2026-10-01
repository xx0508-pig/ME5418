import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from envs.history_wrapper import HistoryStackWrapper
from envs.observation_modes import (
    FORMAL_OBSERVATION_MODES,
    get_observation_mode_spec,
    stacked_observation_dimension,
)


class CountingEnv(gym.Env):
    def __init__(self, observation_dimension=2, action_dimension=1):
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(observation_dimension,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(action_dimension,),
            dtype=np.float32,
        )
        self.step_count = 0

    def _observation(self):
        return np.arange(
            self.observation_space.shape[0],
            dtype=np.float32,
        ) + self.step_count

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self._observation(), {"reset": True}

    def step(self, action):
        self.step_count += 1
        return (
            self._observation(),
            0.0,
            False,
            False,
            {"step": self.step_count},
        )


def test_formal_observation_mode_dimensions():
    assert FORMAL_OBSERVATION_MODES == (
        "pos6",
        "posvel9",
        "posvel_force11",
    )
    assert get_observation_mode_spec("pos6").dimension == 6
    assert get_observation_mode_spec("posvel9").dimension == 9

    force_spec = get_observation_mode_spec(
        "posvel_force11"
    )
    assert force_spec.dimension == 11
    assert force_spec.contact_force_components == 2
    assert force_spec.contact_force_axes == ("x", "y")


@pytest.mark.parametrize(
    "base_dimension,history_length,expected_dimension",
    [
        (6, 1, 6),
        (6, 4, 33),
        (9, 4, 45),
        (11, 4, 53),
    ],
)
def test_stacked_observation_dimensions(
    base_dimension,
    history_length,
    expected_dimension,
):
    assert stacked_observation_dimension(
        base_dimension,
        3,
        history_length,
    ) == expected_dimension

    env = HistoryStackWrapper(
        CountingEnv(
            observation_dimension=base_dimension,
            action_dimension=3,
        ),
        history_length=history_length,
    )
    assert env.observation_space.shape == (
        expected_dimension,
    )


def test_reset_repeats_observation_and_zeros_actions():
    env = HistoryStackWrapper(
        CountingEnv(),
        history_length=4,
    )

    observation, info = env.reset(seed=42)

    np.testing.assert_array_equal(
        observation,
        np.array(
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            dtype=np.float32,
        ),
    )
    assert info == {"reset": True}
    assert env.observation_space.contains(observation)


def test_step_uses_previous_action_and_shifts_history():
    env = HistoryStackWrapper(
        CountingEnv(),
        history_length=3,
    )
    env.reset()

    first, _, _, _, first_info = env.step(
        np.array([0.25], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        first,
        np.array(
            [1, 2, 0.25, 0, 1, 0, 0, 1],
            dtype=np.float32,
        ),
    )
    assert first_info == {"step": 1}

    second, _, _, _, second_info = env.step(
        np.array([-0.5], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        second,
        np.array(
            [2, 3, -0.5, 1, 2, 0.25, 0, 1],
            dtype=np.float32,
        ),
    )
    assert second_info == {"step": 2}


def test_reset_clears_history_from_previous_episode():
    env = HistoryStackWrapper(
        CountingEnv(),
        history_length=3,
    )
    initial, _ = env.reset()
    env.step(np.array([0.75], dtype=np.float32))
    reset_observation, _ = env.reset()

    np.testing.assert_array_equal(
        reset_observation,
        initial,
    )


def test_memoryless_wrapper_preserves_observation():
    env = HistoryStackWrapper(
        CountingEnv(),
        history_length=1,
    )
    observation, _ = env.reset()
    np.testing.assert_array_equal(
        observation,
        np.array([0, 1], dtype=np.float32),
    )

    observation, _, _, _, _ = env.step(
        np.array([0.1], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        observation,
        np.array([1, 2], dtype=np.float32),
    )


def test_invalid_history_length_is_rejected():
    with pytest.raises(ValueError):
        HistoryStackWrapper(
            CountingEnv(),
            history_length=0,
        )

    with pytest.raises(ValueError):
        stacked_observation_dimension(9, 3, 0)


def test_unknown_observation_mode_is_rejected():
    with pytest.raises(ValueError):
        get_observation_mode_spec("unknown")
