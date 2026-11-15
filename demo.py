"""Copy to the repository root; run one random episode and save demo.mp4.

Usage: MUJOCO_GL=egl python demo.py --output demo.mp4
No checkpoint is needed. Random actions do not guarantee task success.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # Before simulator imports.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("demo.mp4"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose another --output path.")
    import imageio.v2 as imageio
    from envs.pushing_env_v1 import LiberoPushingEnvV1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = LiberoPushingEnvV1(observation_size=256,
                            friction_range=(0.3, 0.9), render_mode="rgb_array")
    total_reward = 0.0
    try:
        observation, info = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        print("Observation shape:", observation.shape)
        # Stream frames to avoid keeping an entire video in memory.
        with imageio.get_writer(str(args.output), fps=20) as video:
            video.append_data(env.render())
            for step in range(env.max_steps):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                video.append_data(env.render())
                if terminated or truncated:
                    break
        print(f"Steps: {step + 1}; return: {total_reward:.3f}")
        print("Ending:", info.get("termination_reason"))
        print("Video:", args.output.resolve())
    finally:
        env.close()


if __name__ == "__main__":
    main()
