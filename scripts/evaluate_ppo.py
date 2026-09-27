"""Evaluate a trained PPO policy and record its deterministic rollouts."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from envs.pushing_env import LiberoPushingEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/models/ppo_fixed_friction.zip")
    parser.add_argument("--video", default="outputs/videos/ppo_evaluation.mp4")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--friction", type=float, default=None,
                        help="Set one sliding-friction value; omit for LIBERO's default.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample policy actions instead of using their deterministic mean.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    friction_range = None if args.friction is None else (args.friction, args.friction)
    env = LiberoPushingEnv(
        max_steps=300,
        action_scale=0.5,
        observation_size=256,
        friction_range=friction_range,
    )
    model = PPO.load(model_path, env=env, device="auto")
    frames, results = [], []

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            episode_reward, length = 0.0, 0
            record = episode < args.video_episodes
            if record:
                frames.extend([env.render()] * 8)

            while True:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                length += 1
                if record:
                    frames.append(env.render())
                if terminated or truncated:
                    break

            results.append((bool(info["is_success"]), episode_reward, length, info["goal_distance"]))
            print(
                f"episode={episode + 1:02d} seed={args.seed + episode} "
                f"success={info['is_success']} return={episode_reward:+.3f} "
                f"steps={length} final_goal={info['goal_distance']:.4f} "
                f"friction={info['friction']}"
            )
            if record:
                frames.extend([env.render()] * 12)
    finally:
        env.close()

    successes = np.array([x[0] for x in results], dtype=np.float32)
    returns = np.array([x[1] for x in results])
    lengths = np.array([x[2] for x in results])
    distances = np.array([x[3] for x in results])
    print("\nEvaluation summary")
    print(f"success_rate={successes.mean():.3f} ({int(successes.sum())}/{len(results)})")
    print(f"mean_return={returns.mean():.3f}")
    print(f"mean_episode_length={lengths.mean():.1f}")
    print(f"mean_final_goal_distance={distances.mean():.4f}")

    if frames:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=20)
        print(f"video={video_path}")


if __name__ == "__main__":
    main()
