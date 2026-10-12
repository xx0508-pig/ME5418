import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from envs.history_wrapper import HistoryStackWrapper
from envs.fixed_normalization import (
    FixedObservationNormalizationWrapper,
    fixed_v1_scales,
)
from envs.observation_statistics import ObservationStatisticsWrapper
from envs.observation_modes import (
    FORMAL_OBSERVATION_MODES,
    get_observation_channel_names,
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
    assert get_observation_channel_names("posvel_force11") == (
        "plate_minus_eef_x",
        "plate_minus_eef_y",
        "plate_minus_eef_z",
        "goal_minus_plate_x",
        "goal_minus_plate_y",
        "goal_minus_plate_z",
        "plate_velocity_x",
        "plate_velocity_y",
        "plate_velocity_z",
        "contact_force_x",
        "contact_force_y",
    )


def test_observation_statistics_collects_reset_and_steps():
    env = ObservationStatisticsWrapper(
        CountingEnv(observation_dimension=2),
        channel_names=("first", "second"),
    )
    env.reset()
    env.step(np.array([0.0], dtype=np.float32))
    env.step(np.array([0.0], dtype=np.float32))

    summary = env.summary()

    assert summary["sample_count"] == 3
    assert summary["observation_dimension"] == 2
    first, second = summary["channels"]
    assert first["channel"] == "first"
    assert first["mean"] == pytest.approx(1.0)
    assert first["std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert first["min"] == 0.0
    assert first["max"] == 2.0
    assert first["p50"] == 1.0
    assert second["channel"] == "second"
    assert second["mean"] == pytest.approx(2.0)


def test_observation_statistics_does_not_change_observation():
    base_env = CountingEnv(observation_dimension=2)
    env = ObservationStatisticsWrapper(
        base_env,
        channel_names=("first", "second"),
    )
    observation, info = env.reset(seed=42)
    next_observation, reward, terminated, truncated, step_info = (
        env.step(np.array([0.25], dtype=np.float32))
    )

    np.testing.assert_array_equal(
        observation,
        np.array([0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        next_observation,
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert info == {"reset": True}
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert step_info == {"step": 1}


def test_observation_statistics_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="Channel-name count"):
        ObservationStatisticsWrapper(
            CountingEnv(observation_dimension=2),
            channel_names=("only_one",),
        )


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


def test_fixed_normalization_scales_and_clips_before_history():
    base_env = CountingEnv(observation_dimension=9, action_dimension=3)
    normalized = FixedObservationNormalizationWrapper(
        base_env,
        observation_mode="posvel9",
    )
    history = HistoryStackWrapper(normalized, history_length=4)

    observation, _ = history.reset()

    expected_frame = np.clip(
        np.arange(9, dtype=np.float32)
        / fixed_v1_scales("posvel9"),
        -5.0,
        5.0,
    )
    np.testing.assert_allclose(observation[:9], expected_frame)
    np.testing.assert_allclose(observation[12:21], expected_frame)
    np.testing.assert_array_equal(observation[9:12], np.zeros(3))
    assert history.observation_space.contains(observation)


def test_fixed_normalization_preserves_zero():
    env = FixedObservationNormalizationWrapper(
        CountingEnv(observation_dimension=6),
        observation_mode="pos6",
    )
    normalized = env.observation(np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(normalized, np.zeros(6))
