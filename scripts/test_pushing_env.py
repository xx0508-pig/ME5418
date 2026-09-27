import numpy as np

from stable_baselines3.common.env_checker import check_env

from envs.pushing_env import LiberoPushingEnv


def main():

    print("\n================================")
    print("Creating environment")
    print("================================")

    env = LiberoPushingEnv(
        max_steps=300,
        action_scale=0.5,
        observation_size=64,
    )

    # --------------------------------------------------
    # Stable-Baselines3 API validation
    # --------------------------------------------------

    print("\n================================")
    print("Running SB3 check_env")
    print("================================")

    check_env(
        env,
        warn=True,
        skip_render_check=True,
    )

    print("\ncheck_env passed!")

    # --------------------------------------------------
    # Manual reset
    # --------------------------------------------------

    obs, info = env.reset(seed=42)

    print("\n================================")
    print("Initial observation")
    print("================================")

    print("Observation:")
    print(obs)

    print("\nObservation shape:")
    print(obs.shape)

    print("\nInfo:")
    print(info)

    # --------------------------------------------------
    # Random actions
    # --------------------------------------------------

    print("\n================================")
    print("Random policy rollout")
    print("================================")

    for step in range(30):

        action = env.action_space.sample()

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        print(
            f"step={step:03d} | "
            f"reward={reward:+.5f} | "
            f"reach={info['reach_distance']:.4f} | "
            f"goal={info['goal_distance']:.4f} | "
            f"reach_progress={info['reach_progress']:+.5f} | "
            f"goal_progress={info['goal_progress']:+.5f} | "
            f"success={info['is_success']}"
        )

        if terminated or truncated:

            print("\nEpisode finished.")

            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
