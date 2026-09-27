import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from envs.pushing_env import LiberoPushingEnv


def get_plate_table_contact_frictions(env):
    task = env._get_libero_task_env()
    sim = task.sim

    plate_body_id = task.obj_body_id["plate_1"]
    table_geom_id = sim.model.geom_name2id("table_collision")

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

@pytest.mark.integration
def test_environment_smoke():
    env = LiberoPushingEnv(
        max_steps=3,
        observation_size=32,
        friction_range=(0.3, 0.9),
    )

    try:
        check_env(env, warn=True, skip_render_check=True)

        obs, info = env.reset(seed=42)

        assert obs.shape == (6,)
        assert np.isfinite(obs).all()
        assert 0.3 <= info["friction"] <= 0.9

        frame = env.render()
        assert frame.shape == (32, 32, 3)

        action = np.zeros(3, dtype=np.float32)
        obs, reward, _, _, _ = env.step(action)

        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
    finally:
        env.close()

@pytest.mark.integration
@pytest.mark.parametrize(
    "include_velocity,include_force,expected_dim",
    [
        (False, False, 6),
        (True, False, 9),
        (False, True, 9),
        (True, True, 12),
    ],
)
def test_observation_modes(
    include_velocity,
    include_force,
    expected_dim,
):
    env = LiberoPushingEnv(
        max_steps=2,
        observation_size=32,
        include_object_velocity=include_velocity,
        include_contact_force=include_force,
    )

    try:
        obs, _ = env.reset(seed=42)

        assert obs.shape == (expected_dim,)
        assert np.isfinite(obs).all()
    finally:
        env.close()

@pytest.mark.integration
def test_reset_is_reproducible_with_same_seed():
    env = LiberoPushingEnv(
        max_steps=2,
        observation_size=32,
        friction_range=(0.3, 0.9),
    )

    try:
        obs1, info1 = env.reset(seed=123)
        obs2, info2 = env.reset(seed=123)

        np.testing.assert_allclose(obs1, obs2, atol=1e-6)
        assert info1["friction"] == pytest.approx(
            info2["friction"]
        )
    finally:
        env.close()

@pytest.mark.integration
@pytest.mark.parametrize("friction", [0.2, 0.6, 1.1])
def test_runtime_contact_friction_matches_requested(friction):
    env = LiberoPushingEnv(
        max_steps=2,
        observation_size=32,
        friction_range=(friction, friction),
    )

    try:
        _, info = env.reset(seed=42)

        actual = get_plate_table_contact_frictions(env)

        assert actual.size > 0
        assert info["friction"] == pytest.approx(friction)
        np.testing.assert_allclose(
            actual,
            friction,
            atol=1e-6,
        )
    finally:
        env.close()

@pytest.mark.integration
def test_episode_truncates_at_max_steps():
    env = LiberoPushingEnv(
        max_steps=2,
        observation_size=32,
        success_radius=0.0,
    )

    try:
        env.reset(seed=42)
        action = np.zeros(3, dtype=np.float32)

        _, _, terminated1, truncated1, _ = env.step(action)
        _, _, terminated2, truncated2, _ = env.step(action)

        assert not terminated1
        assert not truncated1
        assert not terminated2
        assert truncated2
    finally:
        env.close()


@pytest.mark.integration
def test_large_success_radius_terminates_episode():
    env = LiberoPushingEnv(
        max_steps=5,
        observation_size=32,
        success_radius=10.0,
    )

    try:
        env.reset(seed=42)

        action = np.zeros(3, dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)

        assert terminated
        assert not truncated
        assert info["is_success"] is True
    finally:
        env.close()