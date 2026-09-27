"""Train or continue PPO with optional per-episode friction randomization."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from envs.pushing_env import LiberoPushingEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-name", default="ppo_fixed_friction")
    parser.add_argument("--resume", default=None, help="Existing .zip model to continue training.")
    parser.add_argument("--friction-min", type=float, default=None)
    parser.add_argument("--friction-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="Entropy coefficient. Fresh training defaults to 0.01; "
                             "when resuming, omit to retain the saved value.")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.friction_min is None) != (args.friction_max is None):
        raise ValueError("Provide both --friction-min and --friction-max, or neither.")
    if args.friction_min is not None and not 0 < args.friction_min <= args.friction_max:
        raise ValueError("Require 0 < friction-min <= friction-max.")
    friction_range = None if args.friction_min is None else (args.friction_min, args.friction_max)

    env = Monitor(
        LiberoPushingEnv(max_steps=300, action_scale=0.5, observation_size=64,
                         friction_range=friction_range),
        info_keywords=("is_success", "goal_distance", "friction"),
    )
    output_dir = Path("outputs")
    model_dir = output_dir / "models"
    checkpoint_dir = output_dir / "checkpoints" / args.save_name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        model = PPO.load(args.resume, env=env, device="auto")
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef
        reset_num_timesteps = False
        print(f"Continuing from {args.resume}; ent_coef={model.ent_coef}")
    else:
        ent_coef = 0.01 if args.ent_coef is None else args.ent_coef
        model = PPO(
            "MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
            batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=ent_coef, verbose=1,
            tensorboard_log=str(output_dir / "logs"), device="auto", seed=args.seed,
        )
        reset_num_timesteps = True

    callback = CheckpointCallback(
        save_freq=args.checkpoint_freq, save_path=str(checkpoint_dir),
        name_prefix=args.save_name, save_replay_buffer=False, save_vecnormalize=False,
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    reset_num_timesteps=reset_num_timesteps, tb_log_name=args.save_name)
        path = model_dir / args.save_name
        model.save(path)
        print(f"Saved final model to {path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
