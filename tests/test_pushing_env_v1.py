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
