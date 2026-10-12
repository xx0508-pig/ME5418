"""Run the five observation/history ablations on identical episodes."""

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
from pathlib import Path


RUNS = (
    ("pos6_k1", "formal_pos6_k1_seed42"),
    ("posvel9_k1", "formal_posvel9_k1_seed42"),
    ("posvel_force11_k1", "formal_posvel_force11_k1_seed42"),
    ("posvel9_k4", "formal_posvel9_k4_seed42"),
    ("posvel_force11_k4", "formal_posvel_force11_k4_seed42"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all five seed-42 ablation models with identical "
            "episode seeds and friction settings."
        )
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help=(
            "Episodes per model and condition "
            "(default: 500; 7500 total)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4000,
        help="First episode seed; subsequent episodes use seed + index.",
    )
    parser.add_argument(
        "--friction-min",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--friction-max",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--fixed-friction",
        type=float,
        action="append",
        dest="fixed_frictions",
        default=None,
        help=(
            "Fixed-friction condition. Repeat for multiple values. "
            "Defaults to 0.9 and 1.0."
        ),
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=0,
        help="Videos per model. Keep this at 0 for long evaluations.",
    )
    parser.add_argument(
        "--suite-name",
        "--evaluation-name",
        dest="suite_name",
        default=None,
        help="Optional suite name; a descriptive name is generated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running evaluations.",
    )
    return parser.parse_args()


def number_slug(value):
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def generated_suite_name(args):
    final_seed = args.seed + args.episodes - 1
    return (
        f"dev_ablation_n{args.episodes}_"
        f"seeds_{args.seed}_{final_seed}"
    )


def build_conditions(args, suite_name):
    random_label = (
        "random_mu_"
        f"{number_slug(args.friction_min)}_"
        f"{number_slug(args.friction_max)}"
    )
    conditions = [
        {
            "label": random_label,
            "evaluation_name": (
                f"{suite_name}_{random_label}"
            ),
            "friction_range": [
                args.friction_min,
                args.friction_max,
            ],
            "command_arguments": [
                "--friction-min",
                str(args.friction_min),
                "--friction-max",
                str(args.friction_max),
            ],
        }
    ]
    fixed_frictions = args.fixed_frictions or [0.9, 1.0]
    for friction in fixed_frictions:
        slug = number_slug(friction)
        conditions.append(
            {
                "label": f"fixed_mu_{slug}",
                "evaluation_name": (
                    f"{suite_name}_fixed_mu_{slug}"
                ),
                "friction_range": [friction, friction],
                "command_arguments": [
                    "--friction",
                    str(friction),
                ],
            }
        )
    return conditions


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    if args.video_episodes < 0:
        raise ValueError("--video-episodes cannot be negative.")
    if args.video_episodes > args.episodes:
        raise ValueError("--video-episodes cannot exceed --episodes.")
    if not math.isfinite(args.friction_min) or not math.isfinite(
        args.friction_max
    ):
        raise ValueError("Friction bounds must be finite.")
    if args.friction_min <= 0 or args.friction_max <= 0:
        raise ValueError("Friction bounds must be positive.")
    if args.friction_min > args.friction_max:
        raise ValueError(
            "--friction-min cannot exceed --friction-max."
        )
    fixed_frictions = args.fixed_frictions or [0.9, 1.0]
    if not all(math.isfinite(value) for value in fixed_frictions):
        raise ValueError("Fixed friction values must be finite.")
    if not all(value > 0 for value in fixed_frictions):
        raise ValueError("Fixed friction values must be positive.")
    if len(set(fixed_frictions)) != len(fixed_frictions):
        raise ValueError("Fixed friction values must be unique.")


def wilson_interval(success_count, episodes, z=1.959963984540054):
    rate = success_count / episodes
    denominator = 1.0 + z * z / episodes
    center = (rate + z * z / (2.0 * episodes)) / denominator
    half_width = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / episodes
            + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )
    return center - half_width, center + half_width


def completed_evaluation_is_compatible(
    summary_path,
    csv_path,
    args,
    expected_friction_range,
):
    if not summary_path.exists() and not csv_path.exists():
        return False
    if not summary_path.exists() or not csv_path.exists():
        raise RuntimeError(
            "Incomplete existing evaluation artifacts: "
            f"{summary_path.parent}. Rename or remove the incomplete "
            "artifacts before retrying."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    compatible = (
        evaluation["episodes"] == args.episodes
        and evaluation["base_seed"] == args.seed
        and evaluation["requested_friction_range"]
        == expected_friction_range
        and evaluation["video_episodes"] == args.video_episodes
    )
    if not compatible:
        raise RuntimeError(
            "Existing evaluation has different settings: "
            f"{summary_path}"
        )
    return True


def evaluation_command(
    repo_root,
    run_directory,
    evaluation_name,
    args,
    friction_arguments,
):
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_ppo",
        "--run-dir",
        str(run_directory.relative_to(repo_root)),
        "--evaluation-name",
        evaluation_name,
        "--episodes",
        str(args.episodes),
        "--video-episodes",
        str(args.video_episodes),
        "--seed",
        str(args.seed),
    ]
    command.extend(friction_arguments)
    return command


def write_aggregate(repo_root, suite_dir, suite_name, conditions):
    rows = []
    for condition in conditions:
        evaluation_name = condition["evaluation_name"]
        for label, run_name in RUNS:
            summary_path = (
                repo_root
                / "outputs"
                / "runs"
                / run_name
                / "evaluations"
                / f"{evaluation_name}_summary.json"
            )
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            metrics = summary["metrics"]
            model = summary["model"]
            safety = metrics.get("contact_force_safety", {})
            normalization = model.get(
                "normalization",
                {"name": "none"},
            )
            confidence_low, confidence_high = wilson_interval(
                metrics["success_count"],
                summary["evaluation"]["episodes"],
            )
            rows.append(
                {
                    "condition": condition["label"],
                    "requested_friction_range": json.dumps(
                        condition["friction_range"]
                    ),
                    "model": label,
                    "run_name": run_name,
                    "observation_mode": model["observation_mode"],
                    "history_length": model["history_length"],
                    "normalization": normalization.get(
                        "name",
                        "none",
                    ),
                    "episodes": summary["evaluation"]["episodes"],
                    "success_count": metrics["success_count"],
                    "success_rate": metrics["success_rate"],
                    "success_rate_ci95_low": confidence_low,
                    "success_rate_ci95_high": confidence_high,
                    "mean_episode_length": metrics[
                        "mean_episode_length"
                    ],
                    "std_episode_length": metrics[
                        "std_episode_length"
                    ],
                    "mean_return": metrics["mean_return"],
                    "std_return": metrics["std_return"],
                    "mean_episode_peak_contact_force_xy": safety.get(
                        "mean_episode_peak_xy"
                    ),
                    "max_episode_peak_contact_force_xy": safety.get(
                        "max_episode_peak_xy"
                    ),
                    "contact_force_episode_exceedance_rate": safety.get(
                        "episode_exceedance_rate"
                    ),
                    "contact_force_step_exceedance_rate": safety.get(
                        "exceedance_step_rate"
                    ),
                    "termination_counts": json.dumps(
                        metrics["termination_counts"],
                        sort_keys=True,
                    ),
                    "summary_json": str(summary_path),
                }
            )

    suite_dir.mkdir(parents=True, exist_ok=True)
    csv_path = suite_dir / "summary.csv"
    json_path = suite_dir / "summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "suite_name": suite_name,
                "conditions": [
                    {
                        "label": condition["label"],
                        "evaluation_name": condition[
                            "evaluation_name"
                        ],
                        "friction_range": condition["friction_range"],
                    }
                    for condition in conditions
                ],
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def acquire_suite_lock(suite_dir):
    suite_dir.mkdir(parents=True, exist_ok=True)
    lock_path = suite_dir / "suite.lock"
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(
            lock_stream.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError(
            "This evaluation suite is already running. "
            f"Lock file: {lock_path}"
        ) from error
    lock_stream.write(f"pid={os.getpid()}\n")
    lock_stream.flush()
    return lock_stream


def main():
    args = parse_args()
    validate_args(args)
    repo_root = Path(__file__).resolve().parents[1]
    suite_name = args.suite_name or generated_suite_name(args)
    conditions = build_conditions(args, suite_name)
    suite_dir = (
        repo_root / "outputs" / "evaluation_suites" / suite_name
    )
    suite_lock = acquire_suite_lock(suite_dir)

    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault(
        "MPLCONFIGDIR",
        "/tmp/matplotlib-contact-pushing",
    )

    total_jobs = len(RUNS) * len(conditions)
    total_episodes = total_jobs * args.episodes
    print(f"Suite name: {suite_name}", flush=True)
    print(
        f"Models: {len(RUNS)}, conditions: {len(conditions)}, "
        f"episodes/model/condition: {args.episodes}, "
        f"total episodes: {total_episodes}",
        flush=True,
    )

    job_index = 0
    for condition in conditions:
        for label, run_name in RUNS:
            job_index += 1
            run_directory = repo_root / "outputs" / "runs" / run_name
            if not (run_directory / "final_model.zip").is_file():
                raise FileNotFoundError(
                    f"Final model not found: {run_directory}"
                )

            evaluation_name = condition["evaluation_name"]
            evaluation_dir = run_directory / "evaluations"
            summary_path = (
                evaluation_dir / f"{evaluation_name}_summary.json"
            )
            csv_path = (
                evaluation_dir / f"{evaluation_name}_episodes.csv"
            )
            command = evaluation_command(
                repo_root,
                run_directory,
                evaluation_name,
                args,
                condition["command_arguments"],
            )

            print(
                f"\n[{job_index}/{total_jobs}] "
                f"{condition['label']} | {label} ({run_name})",
                flush=True,
            )
            if args.dry_run:
                print("  " + " ".join(command), flush=True)
                continue
            if completed_evaluation_is_compatible(
                summary_path,
                csv_path,
                args,
                condition["friction_range"],
            ):
                print("  Already complete; skipping.", flush=True)
                continue

            subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                check=True,
            )

    if args.dry_run:
        print("\nDry run complete; no evaluation was executed.")
        return

    csv_path, json_path = write_aggregate(
        repo_root,
        suite_dir,
        suite_name,
        conditions,
    )
    print("\nAblation evaluation suite complete.")
    print(f"Aggregate CSV: {csv_path}")
    print(f"Aggregate JSON: {json_path}")
    suite_lock.close()


if __name__ == "__main__":
    main()
