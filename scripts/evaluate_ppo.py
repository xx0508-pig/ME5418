"""Evaluate a PPO run and save structured results."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from envs.pushing_env import LiberoPushingEnv


OBSERVATION_MODES = {
    "pos6": (False, False),
    "posvel9": (True, False),
    "posforce9": (False, True),
    "posvelforce12": (True, True),
}


def current_time():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory containing config.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model path. Defaults to "
            "<run-dir>/final_model.zip."
        ),
    )
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="Name used for CSV, JSON, and video files.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help=(
            "Use one fixed sliding-friction value. "
            "Omit to use the environment default."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic actions.",
    )

    return parser.parse_args()


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    if args.video_episodes < 0:
        raise ValueError(
            "--video-episodes cannot be negative."
        )

    if args.video_episodes > args.episodes:
        raise ValueError(
            "--video-episodes cannot exceed --episodes."
        )

    if args.friction is not None and args.friction <= 0:
        raise ValueError("--friction must be positive.")

    evaluation_name = args.evaluation_name

    if (
        not evaluation_name
        or evaluation_name in {".", ".."}
        or Path(evaluation_name).name != evaluation_name
    ):
        raise ValueError(
            "--evaluation-name must be one directory-safe name."
        )

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.json"

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training config not found: {config_path}"
        )

    training_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if args.model is None:
        model_path = run_dir / "final_model.zip"
    else:
        model_path = Path(args.model).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    observation_mode = training_config["environment"][
        "observation_mode"
    ]

    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(
            "Unsupported observation mode in training config: "
            f"{observation_mode}"
        )

    include_velocity, include_force = OBSERVATION_MODES[
        observation_mode
    ]

    evaluation_dir = run_dir / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        evaluation_dir
        / f"{evaluation_name}_episodes.csv"
    )
    summary_path = (
        evaluation_dir
        / f"{evaluation_name}_summary.json"
    )
    video_path = evaluation_dir / f"{evaluation_name}.mp4"

    protected_paths = [csv_path, summary_path]

    if args.video_episodes > 0:
        protected_paths.append(video_path)

    existing_paths = [
        path for path in protected_paths if path.exists()
    ]

    if existing_paths:
        raise FileExistsError(
            "Evaluation output already exists: "
            + ", ".join(str(path) for path in existing_paths)
        )

    friction_range = (
        None
        if args.friction is None
        else (args.friction, args.friction)
    )

    env = None
    frames = []
    results = []

    try:
        env = LiberoPushingEnv(
            max_steps=300,
            action_scale=0.5,
            observation_size=256,
            friction_range=friction_range,
            include_object_velocity=include_velocity,
            include_contact_force=include_force,
        )

        model = PPO.load(
            str(model_path),
            env=env,
            device="cpu",
        )

        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index

            obs, info = env.reset(seed=episode_seed)
            episode_return = 0.0
            episode_length = 0

            record_video = (
                episode_index < args.video_episodes
            )

            if record_video:
                frames.extend([env.render()] * 8)

            while True:
                action, _ = model.predict(
                    obs,
                    deterministic=not args.stochastic,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                episode_return += reward
                episode_length += 1

                if record_video:
                    frames.append(env.render())

                if terminated or truncated:
                    break

            result = {
                "episode": episode_index + 1,
                "seed": episode_seed,
                "success": bool(info["is_success"]),
                "episode_return": float(episode_return),
                "episode_length": int(episode_length),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_goal_distance": float(
                    info["goal_distance"]
                ),
                "requested_friction": (
                    None
                    if info.get("requested_friction") is None
                    else float(info["requested_friction"])
                ),
                "contact_friction": (
                    None
                    if info.get("contact_friction") is None
                    else float(info["contact_friction"])
                ),
            }
            results.append(result)

            print(
                f"episode={result['episode']:02d} "
                f"seed={episode_seed} "
                f"success={result['success']} "
                f"return={episode_return:+.3f} "
                f"steps={episode_length} "
                f"final_goal="
                f"{result['final_goal_distance']:.4f} "
                f"friction={result['contact_friction']}"
            )

            if record_video:
                frames.extend([env.render()] * 12)

    finally:
        if env is not None:
            env.close()

    fieldnames = [
        "episode",
        "seed",
        "success",
        "episode_return",
        "episode_length",
        "terminated",
        "truncated",
        "final_goal_distance",
        "requested_friction",
        "contact_friction",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    successes = np.asarray(
        [row["success"] for row in results],
        dtype=np.float64,
    )
    returns = np.asarray(
        [row["episode_return"] for row in results],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [row["episode_length"] for row in results],
        dtype=np.float64,
    )
    distances = np.asarray(
        [row["final_goal_distance"] for row in results],
        dtype=np.float64,
    )
    contact_frictions = np.asarray(
        [
            row["contact_friction"]
            for row in results
            if row["contact_friction"] is not None
        ],
        dtype=np.float64,
    )

    if frames:
        imageio.mimsave(video_path, frames, fps=20)

    summary = {
        "schema_version": 1,
        "created_at": current_time(),
        "command": {
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
        },
        "evaluation": {
            "name": evaluation_name,
            "episodes": args.episodes,
            "base_seed": args.seed,
            "deterministic": not args.stochastic,
            "requested_friction": args.friction,
            "video_episodes": args.video_episodes,
        },
        "model": {
            "path": str(model_path),
            "run_directory": str(run_dir),
            "training_config_path": str(config_path),
            "training_git": training_config.get("git"),
            "observation_mode": observation_mode,
        },
        "metrics": {
            "success_count": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "mean_episode_length": float(lengths.mean()),
            "std_episode_length": float(lengths.std()),
            "mean_final_goal_distance": float(
                distances.mean()
            ),
            "std_final_goal_distance": float(
                distances.std()
            ),
            "mean_contact_friction": (
                float(contact_frictions.mean())
                if contact_frictions.size
                else None
            ),
        },
        "artifacts": {
            "episodes_csv": str(csv_path),
            "summary_json": str(summary_path),
            "video": (
                str(video_path) if frames else None
            ),
        },
    }

    write_json(summary_path, summary)

    print("\nEvaluation summary")
    print(
        f"success_rate={successes.mean():.3f} "
        f"({int(successes.sum())}/{len(results)})"
    )
    print(f"mean_return={returns.mean():.3f}")
    print(
        f"mean_episode_length={lengths.mean():.1f}"
    )
    print(
        "mean_final_goal_distance="
        f"{distances.mean():.4f}"
    )
    print(f"episodes_csv={csv_path}")
    print(f"summary_json={summary_path}")

    if frames:
        print(f"video={video_path}")


if __name__ == "__main__":
    main()
