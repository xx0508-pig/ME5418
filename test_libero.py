import numpy as np

from lerobot.envs.configs import LiberoEnv


def main():
    cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=[0],
        observation_height=128,
        observation_width=128,
    )

    envs = cfg.create_envs(
        n_envs=1,
        use_async_envs=False,
    )

    env = envs["libero_spatial"][0]

    obs, info = env.reset(seed=42)

    print("Environment reset successfully")
    print("Observation keys:")
    print(obs.keys())

    # LIBERO action:
    # dx dy dz dRx dRy dRz gripper
    action = np.zeros((1, 7), dtype=np.float32)

    # gripper open / idle
    action[:, -1] = -1.0

    for i in range(20):
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={i}, "
            f"reward={reward}, "
            f"terminated={terminated}"
        )

    env.close()


if __name__ == "__main__":
    main()

