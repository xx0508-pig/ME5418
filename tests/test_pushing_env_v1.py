import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import registry

import envs
from envs import V0_ENV_ID, V1_ENV_ID
from envs.pushing_env_v1 import LiberoPushingEnvV1


def test_v0_and_v1_are_registered():
    assert V0_ENV_ID in registry
    assert V1_ENV_ID in registry
    assert V0_ENV_ID != V1_ENV_ID


@pytest.mark.integration
def test_v1_environment_smoke():
    env = gym.make(
        V1_ENV_ID,
        max_steps=2,
        observation_size=32,
        friction_range=(0.6, 0.6),
    )

    try:
        obs, info = env.reset(seed=42)

        assert obs.shape == (6,)
        assert np.isfinite(obs).all()
        assert info["friction"] == pytest.approx(0.6)

        action = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert obs.shape == (6,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    finally:
        env.close()


def test_stable_success_requires_five_steps():
    env = LiberoPushingEnvV1(
        stable_steps_required=5,
        object_speed_threshold=0.02,
    )

    try:
        for expected_count in range(1, 5):
            success = env._update_stable_success(
                distance_success=True,
                object_speed=0.01,
            )

            assert not success
            assert env.stable_count == expected_count

        success = env._update_stable_success(
            distance_success=True,
            object_speed=0.01,
        )

        assert success
        assert env.stable_count == 5

    finally:
        env.close()


def test_unstable_step_resets_stable_count():
    env = LiberoPushingEnvV1(
        stable_steps_required=5,
        object_speed_threshold=0.02,
    )

    try:
        env._update_stable_success(
            distance_success=True,
            object_speed=0.01,
        )
        env._update_stable_success(
            distance_success=True,
            object_speed=0.01,
        )

        assert env.stable_count == 2

        success = env._update_stable_success(
            distance_success=True,
            object_speed=0.03,
        )

        assert not success
        assert env.stable_count == 0

    finally:
        env.close()


def test_termination_reason_priority():
    select = (
        LiberoPushingEnvV1
        ._select_termination_reason
    )

    assert select(
        invalid_state=True,
        out_of_workspace=True,
        stable_success=True,
        time_limit=True,
    ) == "invalid_state"

    assert select(
        invalid_state=False,
        out_of_workspace=True,
        stable_success=True,
        time_limit=True,
    ) == "out_of_workspace"

    assert select(
        invalid_state=False,
        out_of_workspace=False,
        stable_success=True,
        time_limit=True,
    ) == "stable_success"

    assert select(
        invalid_state=False,
        out_of_workspace=False,
        stable_success=False,
        time_limit=True,
    ) == "time_limit"

    assert select(
        invalid_state=False,
        out_of_workspace=False,
        stable_success=False,
        time_limit=False,
    ) is None


def test_finite_value_check():
    check = LiberoPushingEnvV1._values_are_finite

    assert check(
        np.zeros(3),
        1.0,
        np.array([1.0, 2.0]),
    )

    assert not check(
        np.array([0.0, np.nan]),
    )

    assert not check(
        np.array([0.0, np.inf]),
    )


@pytest.mark.integration
def test_v1_time_limit_reason():
    env = LiberoPushingEnvV1(
        max_steps=1,
        observation_size=32,
        success_radius=0.001,
    )

    try:
        env.reset(seed=42)

        action = np.zeros(3, dtype=np.float32)
        _, _, terminated, truncated, info = (
            env.step(action)
        )

        assert not terminated
        assert truncated
        assert not info["is_success"]
        assert info["time_limit"]
        assert info["termination_reason"] == "time_limit"

    finally:
        env.close()


@pytest.mark.integration
def test_workspace_detection():
    env = LiberoPushingEnvV1(
        max_steps=2,
        observation_size=32,
    )

    try:
        env.reset(seed=42)

        plate_position = env._get_plate_position()

        assert env._is_plate_in_workspace(
            plate_position
        )

        outside_position = plate_position.copy()
        outside_position[0] = 10.0

        assert not env._is_plate_in_workspace(
            outside_position
        )

    finally:
        env.close()
