"""Evaluate a PPO policy and save structured benchmark results."""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from envs import (
    ENV_IDS,
    OBSERVATION_MODES,
    HistoryStackWrapper,
    get_observation_mode_spec,
    make_pushing_env,
    stacked_observation_dimension,
)


def current_time():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Training run containing config.json. Required unless "
            "legacy-model arguments are supplied explicitly."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model path. With --run-dir, defaults to "
            "<run-dir>/final_model.zip."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional evaluation output directory. With --run-dir, "
            "defaults to <run-dir>/evaluations. Required without a run."
        ),
    )
    parser.add_argument(
        "--observation-mode",
        choices=sorted(OBSERVATION_MODES),
        default=None,
        help=(
            "Required for a legacy model without --run-dir. Otherwise "
            "read from the training config."
        ),
    )
    parser.add_argument(
        "--history-length",
        type=int,
        choices=[1, 4],
        default=None,
        help=(
            "Required for a legacy history model. With --run-dir, "
            "the value is read from config.json; old configs default to 1."
        ),
    )
    parser.add_argument(
        "--env-version",
        choices=sorted(ENV_IDS),
        default=None,
        help=(
            "Override the evaluation environment version. Otherwise "
            "read from config.json; old configs default to v0."
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
        help="Use one fixed sliding-friction value.",
    )
    parser.add_argument("--friction-min", type=float, default=None)
    parser.add_argument("--friction-max", type=float, default=None)
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


def optional_float(value):
    if value is None:
        return None
    return float(value)


def optional_bool(value):
    if value is None:
        return None
    return bool(value)


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def std_or_none(values):
    return float(np.std(values)) if values else None


def resolve_friction_range(args):
    has_range = (
        args.friction_min is not None
        or args.friction_max is not None
    )
    if args.friction is not None and has_range:
        raise ValueError(
            "--friction cannot be combined with "
            "--friction-min/--friction-max."
        )
    if (args.friction_min is None) != (
        args.friction_max is None
    ):
        raise ValueError(
            "--friction-min and --friction-max must be provided together."
        )
    if args.friction is not None:
        if not np.isfinite(args.friction):
            raise ValueError("--friction must be finite.")
        if args.friction <= 0:
            raise ValueError("--friction must be positive.")
        value = float(args.friction)
        return value, value
    if args.friction_min is None:
        return None
    if not np.isfinite(
        [args.friction_min, args.friction_max]
    ).all():
        raise ValueError("Friction bounds must be finite.")
    if args.friction_min <= 0 or args.friction_max <= 0:
        raise ValueError("Friction bounds must be positive.")
    if args.friction_min > args.friction_max:
        raise ValueError(
            "--friction-min cannot be greater than --friction-max."
        )
    return float(args.friction_min), float(args.friction_max)


def resolve_evaluation_context(args):
    run_dir = None
    config_path = None
    training_config = None

    if args.run_dir is not None:
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
        environment_config = training_config["environment"]
        observation_mode = environment_config["observation_mode"]
        if (
            args.observation_mode is not None
            and args.observation_mode != observation_mode
        ):
            raise ValueError(
                "--observation-mode does not match config.json: "
                f"{args.observation_mode!r} != {observation_mode!r}"
            )

        history_length = int(
            environment_config.get("history_length", 1)
        )
        requested_history_length = getattr(
            args,
            "history_length",
            None,
        )
        if (
            requested_history_length is not None
            and requested_history_length != history_length
        ):
            raise ValueError(
                "--history-length does not match config.json: "
                f"{requested_history_length!r} != {history_length!r}"
            )

        training_env_version = environment_config.get(
            "env_version",
            "v0",
        )
        evaluation_env_version = (
            args.env_version
            if args.env_version is not None
            else training_env_version
        )
        model_path = (
            Path(args.model).resolve()
            if args.model is not None
            else run_dir / "final_model.zip"
        )
        evaluation_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir is not None
            else run_dir / "evaluations"
        )
    else:
        missing = []
        if args.model is None:
            missing.append("--model")
        if args.output_dir is None:
            missing.append("--output-dir")
        if args.observation_mode is None:
            missing.append("--observation-mode")
        if args.env_version is None:
            missing.append("--env-version")
        if missing:
            raise ValueError(
                "Without --run-dir, explicitly provide: "
                + ", ".join(missing)
            )

        model_path = Path(args.model).resolve()
        evaluation_dir = Path(args.output_dir).resolve()
        observation_mode = args.observation_mode
        requested_history_length = getattr(
            args,
            "history_length",
            None,
        )
        history_length = (
            requested_history_length
            if requested_history_length is not None
            else 1
        )
        training_env_version = None
        evaluation_env_version = args.env_version

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(
            f"Unsupported observation mode: {observation_mode}"
        )

    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "training_config": training_config,
        "model_path": model_path,
        "evaluation_dir": evaluation_dir,
        "observation_mode": observation_mode,
        "history_length": history_length,
        "training_env_version": training_env_version,
        "evaluation_env_version": evaluation_env_version,
    }


def legacy_termination_reason(info, terminated, truncated):
    reason = info.get("termination_reason")
    if reason is not None:
        return reason
    if bool(info.get("is_success", False)):
        return "legacy_success"
    if truncated:
        return "time_limit"
    if terminated:
        return "legacy_terminated"
    return "unknown"


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    if args.video_episodes < 0:
        raise ValueError("--video-episodes cannot be negative.")
    if args.video_episodes > args.episodes:
        raise ValueError(
            "--video-episodes cannot exceed --episodes."
        )

    evaluation_name = args.evaluation_name
    if (
        not evaluation_name
        or evaluation_name in {".", ".."}
        or Path(evaluation_name).name != evaluation_name
    ):
        raise ValueError(
            "--evaluation-name must be one directory-safe name."
        )

    friction_range = resolve_friction_range(args)
    context = resolve_evaluation_context(args)
    model_path = context["model_path"]
    observation_mode = context["observation_mode"]
    history_length = context["history_length"]
    evaluation_env_version = context["evaluation_env_version"]
    observation_spec = get_observation_mode_spec(
        observation_mode
    )

    evaluation_dir = context["evaluation_dir"]
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evaluation_dir / f"{evaluation_name}_episodes.csv"
    summary_path = evaluation_dir / f"{evaluation_name}_summary.json"
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

    env = None
    frames = []
    results = []
    try:
        raw_env = make_pushing_env(
            evaluation_env_version,
            max_steps=300,
            action_scale=0.5,
            observation_size=256,
            friction_range=friction_range,
            include_object_velocity=(
                observation_spec.include_object_velocity
            ),
            include_contact_force=(
                observation_spec.include_contact_force
            ),
            contact_force_components=(
                observation_spec.contact_force_components
            ),
        )
        env = HistoryStackWrapper(
            raw_env,
            history_length=history_length,
        )
        model = PPO.load(str(model_path), env=env, device="cpu")

        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            obs, reset_info = env.reset(seed=episode_seed)
            info = reset_info
            episode_return = 0.0
            episode_length = 0
            record_video = episode_index < args.video_episodes
            if record_video:
                frames.extend([env.render()] * 8)

            while True:
                action, _ = model.predict(
                    obs,
                    deterministic=not args.stochastic,
                )
                obs, reward, terminated, truncated, info = env.step(
                    action
                )
                episode_return += reward
                episode_length += 1
                if record_video:
                    frames.append(env.render())
                if terminated or truncated:
                    break

            final_goal_distance = optional_float(
                info.get("goal_distance")
            )
            termination_reason = legacy_termination_reason(
                info,
                terminated,
                truncated,
            )
            result = {
                "episode": episode_index + 1,
                "seed": episode_seed,
                "success": bool(info["is_success"]),
                "libero_success": optional_bool(
                    info.get("libero_success")
                ),
                "distance_success": optional_bool(
                    info.get("distance_success")
                ),
                "stable_success": optional_bool(
                    info.get("stable_success")
                ),
                "stable_count": info.get("stable_count"),
                "object_speed": optional_float(
                    info.get("object_speed")
                ),
                "termination_reason": termination_reason,
                "invalid_state": optional_bool(
                    info.get("invalid_state")
                ),
                "out_of_workspace": optional_bool(
                    info.get("out_of_workspace")
                ),
                "time_limit": optional_bool(info.get("time_limit")),
                "episode_return": float(episode_return),
                "episode_length": int(episode_length),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "initial_plate_x": optional_float(
                    reset_info.get("initial_plate_x")
                ),
                "initial_plate_y": optional_float(
                    reset_info.get("initial_plate_y")
                ),
                "initial_plate_z": optional_float(
                    reset_info.get("initial_plate_z")
                ),
                "initial_plate_yaw": optional_float(
                    reset_info.get("initial_plate_yaw")
                ),
                "initial_plate_yaw_offset": optional_float(
                    reset_info.get("initial_plate_yaw_offset")
                ),
                "final_goal_distance": final_goal_distance,
                "requested_friction": optional_float(
                    info.get("requested_friction")
                ),
                "contact_friction": optional_float(
                    info.get("contact_friction")
                ),
            }
            results.append(result)

            goal_text = (
                f"{final_goal_distance:.4f}"
                if final_goal_distance is not None
                else "invalid"
            )
            print(
                f"episode={result['episode']:02d} "
                f"seed={episode_seed} "
                f"success={result['success']} "
                f"reason={termination_reason} "
                f"return={episode_return:+.3f} "
                f"steps={episode_length} "
                f"final_goal={goal_text} "
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
        "libero_success",
        "distance_success",
        "stable_success",
        "stable_count",
        "object_speed",
        "termination_reason",
        "invalid_state",
        "out_of_workspace",
        "time_limit",
        "episode_return",
        "episode_length",
        "terminated",
        "truncated",
        "initial_plate_x",
        "initial_plate_y",
        "initial_plate_z",
        "initial_plate_yaw",
        "initial_plate_yaw_offset",
        "final_goal_distance",
        "requested_friction",
        "contact_friction",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    successes = [float(row["success"]) for row in results]
    returns = [row["episode_return"] for row in results]
    lengths = [row["episode_length"] for row in results]
    distances = [
        row["final_goal_distance"]
        for row in results
        if row["final_goal_distance"] is not None
    ]
    contact_frictions = [
        row["contact_friction"]
        for row in results
        if row["contact_friction"] is not None
    ]
    initial_x = [
        row["initial_plate_x"]
        for row in results
        if row["initial_plate_x"] is not None
    ]
    initial_y = [
        row["initial_plate_y"]
        for row in results
        if row["initial_plate_y"] is not None
    ]
    stable_successes = [
        float(row["stable_success"])
        for row in results
        if row["stable_success"] is not None
    ]
    termination_counts = dict(
        sorted(
            Counter(
                row["termination_reason"] for row in results
            ).items()
        )
    )

    if frames:
        imageio.mimsave(video_path, frames, fps=20)

    training_config = context["training_config"]
    summary = {
        "schema_version": 2,
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
            "requested_friction_range": (
                list(friction_range)
                if friction_range is not None
                else None
            ),
            "video_episodes": args.video_episodes,
            "evaluation_env_version": evaluation_env_version,
            "evaluation_env_id": ENV_IDS[evaluation_env_version],
        },
        "model": {
            "path": str(model_path),
            "run_directory": (
                str(context["run_dir"])
                if context["run_dir"] is not None
                else None
            ),
            "training_config_path": (
                str(context["config_path"])
                if context["config_path"] is not None
                else None
            ),
            "training_git": (
                training_config.get("git")
                if training_config is not None
                else None
            ),
            "training_env_version": context[
                "training_env_version"
            ],
            "observation_mode": observation_mode,
            "base_observation_dimension": (
                observation_spec.dimension
            ),
            "history_length": history_length,
            "stacked_observation_dimension": (
                stacked_observation_dimension(
                    observation_spec.dimension,
                    3,
                    history_length,
                )
            ),
            "contact_force_axes": list(
                observation_spec.contact_force_axes
            ),
        },
        "metrics": {
            "success_count": int(sum(successes)),
            "success_rate": mean_or_none(successes),
            "stable_success_count": (
                int(sum(stable_successes))
                if stable_successes
                else None
            ),
            "stable_success_rate": mean_or_none(
                stable_successes
            ),
            "mean_return": mean_or_none(returns),
            "std_return": std_or_none(returns),
            "mean_episode_length": mean_or_none(lengths),
            "std_episode_length": std_or_none(lengths),
            "mean_final_goal_distance": mean_or_none(distances),
            "std_final_goal_distance": std_or_none(distances),
            "mean_contact_friction": mean_or_none(
                contact_frictions
            ),
            "termination_counts": termination_counts,
            "initial_plate_x_range": (
                [min(initial_x), max(initial_x)]
                if initial_x
                else None
            ),
            "initial_plate_y_range": (
                [min(initial_y), max(initial_y)]
                if initial_y
                else None
            ),
        },
        "artifacts": {
            "episodes_csv": str(csv_path),
            "summary_json": str(summary_path),
            "video": str(video_path) if frames else None,
        },
    }
    write_json(summary_path, summary)

    print("\nEvaluation summary")
    print(
        f"success_rate={mean_or_none(successes):.3f} "
        f"({int(sum(successes))}/{len(results)})"
    )
    print(f"mean_return={mean_or_none(returns):.3f}")
    print(f"mean_episode_length={mean_or_none(lengths):.1f}")
    if distances:
        print(
            "mean_final_goal_distance="
            f"{mean_or_none(distances):.4f}"
        )
    else:
        print("mean_final_goal_distance=invalid")
    print(f"termination_counts={termination_counts}")
    print(f"episodes_csv={csv_path}")
    print(f"summary_json={summary_path}")
    if frames:
        print(f"video={video_path}")


if __name__ == "__main__":
    main()
