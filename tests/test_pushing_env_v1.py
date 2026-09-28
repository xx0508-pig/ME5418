import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import registry

import envs
from envs import V0_ENV_ID, V1_ENV_ID


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