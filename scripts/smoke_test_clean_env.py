"""Construction, Gym API, render, and scene-content smoke test."""
import numpy as np
from stable_baselines3.common.env_checker import check_env
from envs import LiberoPushingEnv


def main():
    env = LiberoPushingEnv(max_steps=20, observation_size=64, friction_range=(0.3, 0.9))
    try:
        check_env(env, warn=True, skip_render_check=True)
        obs, info = env.reset(seed=42)
        assert obs.shape == (6,) and np.isfinite(obs).all()
        frame = env.render()
        assert frame.shape == (64, 64, 3)
        for _ in range(3):
            obs, reward, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            assert np.isfinite(obs).all() and np.isfinite(reward)
        task = env._get_libero_task_env()
        assert set(task.objects_dict) == {"plate_1"}
        assert set(task.fixtures_dict) == set()
        print("PASS: clean LIBERO pushing environment")
        print(f"obs={obs.shape}, frame={frame.shape}, friction={info['friction']:.3f}")
        print(f"objects={list(task.objects_dict)}, fixtures={list(task.fixtures_dict)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
