"""Run and aggregate the final multi-training-seed benchmark."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = (
    ("pos6_k1", "pos6", 1),
    ("posvel9_k1", "posvel9", 1),
    ("posvel_force11_k1", "posvel_force11", 1),
    ("posvel9_k4", "posvel9", 4),
    ("posvel_force11_k4", "posvel_force11", 4),
)

DEFAULT_TRAINING_SEEDS = (42, 43, 44)
DEFAULT_FIXED_FRICTIONS = (0.3, 0.6, 0.9, 1.0)


def current_time():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def number_slug(value):
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the five normalized ablations across independent "
            "training seeds and aggregate at the policy-seed level."
        )
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        action="append",
        dest="training_seeds",
        default=None,
        help="Repeat to override the default training seeds 42, 43, 44.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Evaluation episodes per model, training seed, and condition.",
    )
    parser.add_argument(
        "--episode-seed",
        type=int,
        default=10_000,
        help="First final-test environment seed.",
    )
    parser.add_argument("--friction-min", type=float, default=0.3)
    parser.add_argument("--friction-max", type=float, default=0.9)
    parser.add_argument(
        "--fixed-friction",
        type=float,
        action="append",
        dest="fixed_frictions",
        default=None,
        help=(
            "Repeat to override the default curve 0.3, 0.6, 0.9, 1.0."
        ),
    )
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument(
        "--contact-force-threshold",
        type=float,
        default=0.20,
    )
    parser.add_argument("--observation-stats", action="store_true")
    parser.add_argument("--run-prefix", default="formal_norm")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a non-formal run from a dirty Git working tree.",
    )
    return parser.parse_args()


def resolved_training_seeds(args):
    return tuple(args.training_seeds or DEFAULT_TRAINING_SEEDS)


def resolved_fixed_frictions(args):
    return tuple(args.fixed_frictions or DEFAULT_FIXED_FRICTIONS)


def generated_suite_name(args):
    final_seed = args.episode_seed + args.episodes - 1
    return (
        f"formal_multiseed_n{args.episodes}_"
        f"envseeds_{args.episode_seed}_{final_seed}"
    )


def validate_args(args):
    training_seeds = resolved_training_seeds(args)
    fixed_frictions = resolved_fixed_frictions(args)
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    if args.video_episodes < 0 or args.video_episodes > args.episodes:
        raise ValueError("--video-episodes must be between 0 and episodes.")
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Training seeds must be unique.")
    values = (args.friction_min, args.friction_max, *fixed_frictions)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("All friction values must be finite and positive.")
    if args.friction_min > args.friction_max:
        raise ValueError("--friction-min cannot exceed --friction-max.")
    if len(set(fixed_frictions)) != len(fixed_frictions):
        raise ValueError("Fixed friction values must be unique.")
    if (
        not math.isfinite(args.contact_force_threshold)
        or args.contact_force_threshold <= 0
    ):
        raise ValueError("Contact-force threshold must be positive.")
    for value, label in (
        (args.run_prefix, "--run-prefix"),
        (args.suite_name, "--suite-name"),
    ):
        if value is None:
            continue
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError(f"{label} must be one directory-safe name.")


def build_conditions(args):
    random_label = (
        f"random_mu_{number_slug(args.friction_min)}_"
        f"{number_slug(args.friction_max)}"
    )
    conditions = [
        {
            "label": random_label,
            "kind": "random",
            "friction": None,
            "friction_range": [args.friction_min, args.friction_max],
            "command_arguments": [
                "--friction-min",
                str(args.friction_min),
                "--friction-max",
                str(args.friction_max),
            ],
        }
    ]
    for friction in resolved_fixed_frictions(args):
        label = f"fixed_mu_{number_slug(friction)}"
        conditions.append(
            {
                "label": label,
                "kind": "fixed",
                "friction": friction,
                "friction_range": [friction, friction],
                "command_arguments": ["--friction", str(friction)],
            }
        )
    return conditions


def run_name(args, model_label, training_seed):
    return f"{args.run_prefix}_{model_label}_seed{training_seed}"


def build_jobs(args, suite_name):
    jobs = []
    for training_seed in resolved_training_seeds(args):
        for condition in build_conditions(args):
            for model_label, observation_mode, history_length in MODELS:
                name = run_name(args, model_label, training_seed)
                evaluation_name = (
                    f"{suite_name}_trainseed{training_seed}_"
                    f"{condition['label']}"
                )
                jobs.append(
                    {
                        "training_seed": training_seed,
                        "model": model_label,
                        "observation_mode": observation_mode,
                        "history_length": history_length,
                        "run_name": name,
                        "run_directory": PROJECT_ROOT / "outputs" / "runs" / name,
                        "condition": condition,
                        "evaluation_name": evaluation_name,
                    }
                )
    return jobs


def git_metadata():
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"sha": sha, "dirty": bool(status), "status": status}


def validate_model_run(job):
    run_dir = job["run_directory"]
    config_path = run_dir / "config.json"
    model_path = run_dir / "final_model.zip"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Completed model run not found: {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env = config["environment"]
    training = config["training"]
    normalization = env.get("normalization", {})
    checks = {
        "completed": config.get("status") == "completed",
        "training_seed": training.get("seed") == job["training_seed"],
        "env_version": env.get("env_version") == "v1",
        "observation_mode": env.get("observation_mode")
        == job["observation_mode"],
        "history_length": env.get("history_length") == job["history_length"],
        "normalization": normalization.get("name") == "fixed_v1",
        "at_least_500k": training.get("final_num_timesteps", 0) >= 500_000,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Incompatible model run ({', '.join(failed)}): {run_dir}"
        )


def artifact_paths(job):
    directory = job["run_directory"] / "evaluations"
    prefix = job["evaluation_name"]
    return {
        "episodes": directory / f"{prefix}_episodes.csv",
        "summary": directory / f"{prefix}_summary.json",
        "stratified": directory / f"{prefix}_stratified.csv",
        "observation_stats": directory / f"{prefix}_observation_stats.csv",
    }


def completed_evaluation_is_compatible(job, args):
    paths = artifact_paths(job)
    required = [paths["episodes"], paths["summary"], paths["stratified"]]
    if args.observation_stats:
        required.append(paths["observation_stats"])
    existing = [path.exists() for path in required]
    if not any(existing):
        return False
    if not all(existing):
        raise RuntimeError(
            f"Incomplete evaluation artifacts: {paths['summary'].parent}"
        )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    compatible = (
        evaluation["episodes"] == args.episodes
        and evaluation["base_seed"] == args.episode_seed
        and evaluation["requested_friction_range"]
        == job["condition"]["friction_range"]
        and evaluation["video_episodes"] == args.video_episodes
        and evaluation["contact_force_threshold"]
        == args.contact_force_threshold
        and summary["model"]["normalization"]["name"] == "fixed_v1"
    )
    if not compatible:
        raise RuntimeError(
            f"Existing evaluation uses different settings: {paths['summary']}"
        )
    return True


def evaluation_command(job, args):
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_ppo",
        "--run-dir",
        str(job["run_directory"].relative_to(PROJECT_ROOT)),
        "--evaluation-name",
        job["evaluation_name"],
        "--episodes",
        str(args.episodes),
        "--video-episodes",
        str(args.video_episodes),
        "--seed",
        str(args.episode_seed),
        "--contact-force-threshold",
        str(args.contact_force_threshold),
    ]
    if args.observation_stats:
        command.append("--observation-stats")
    command.extend(job["condition"]["command_arguments"])
    return command


def per_seed_row(job, args):
    summary_path = artifact_paths(job)["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    safety = metrics["contact_force_safety"]
    return {
        "condition": job["condition"]["label"],
        "condition_kind": job["condition"]["kind"],
        "fixed_friction": job["condition"]["friction"],
        "training_seed": job["training_seed"],
        "model": job["model"],
        "run_name": job["run_name"],
        "episodes": args.episodes,
        "success_count": metrics["success_count"],
        "success_rate": metrics["success_rate"],
        "mean_episode_length": metrics["mean_episode_length"],
        "mean_return": metrics["mean_return"],
        "mean_final_goal_distance": metrics["mean_final_goal_distance"],
        "mean_episode_peak_contact_force_xy": safety[
            "mean_episode_peak_xy"
        ],
        "max_episode_peak_contact_force_xy": safety[
            "max_episode_peak_xy"
        ],
        "contact_force_episode_exceedance_rate": safety[
            "episode_exceedance_rate"
        ],
        "contact_force_step_exceedance_rate": safety[
            "exceedance_step_rate"
        ],
        "termination_counts": json.dumps(
            metrics["termination_counts"], sort_keys=True
        ),
        "summary_json": str(summary_path),
    }


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(per_seed_rows):
    rows = []
    keys = sorted(
        {(row["condition"], row["model"]) for row in per_seed_rows}
    )
    metric_names = (
        "success_rate",
        "mean_episode_length",
        "mean_return",
        "mean_final_goal_distance",
        "mean_episode_peak_contact_force_xy",
        "contact_force_episode_exceedance_rate",
        "contact_force_step_exceedance_rate",
    )
    for condition, model in keys:
        members = [
            row
            for row in per_seed_rows
            if row["condition"] == condition and row["model"] == model
        ]
        first = members[0]
        output = {
            "condition": condition,
            "condition_kind": first["condition_kind"],
            "fixed_friction": first["fixed_friction"],
            "model": model,
            "training_seed_count": len(members),
            "training_seeds": json.dumps(
                sorted(row["training_seed"] for row in members)
            ),
            "episodes_per_training_seed": first["episodes"],
            "total_episodes": sum(row["episodes"] for row in members),
            "pooled_success_rate": (
                sum(row["success_count"] for row in members)
                / sum(row["episodes"] for row in members)
            ),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in members]
            output[f"{metric}_mean_across_training_seeds"] = mean(values)
            output[f"{metric}_std_across_training_seeds"] = sample_std(values)
            output[f"{metric}_min"] = min(values)
            output[f"{metric}_max"] = max(values)
        rows.append(output)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_friction_curve(path, aggregate):
    import matplotlib.pyplot as plt

    fixed_rows = [
        row for row in aggregate if row["condition_kind"] == "fixed"
    ]
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for model_label, _, _ in MODELS:
        model_rows = sorted(
            (row for row in fixed_rows if row["model"] == model_label),
            key=lambda row: row["fixed_friction"],
        )
        axis.errorbar(
            [row["fixed_friction"] for row in model_rows],
            [
                row["success_rate_mean_across_training_seeds"]
                for row in model_rows
            ],
            yerr=[
                row["success_rate_std_across_training_seeds"]
                for row in model_rows
            ],
            marker="o",
            capsize=3,
            linewidth=1.6,
            label=model_label,
        )
    axis.set_xlabel("Contact friction coefficient (mu)")
    axis.set_ylabel("Stable success rate")
    axis.set_xticks(sorted({row["fixed_friction"] for row in fixed_rows}))
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    axis.set_title("Fixed-friction generalization across training seeds")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_outputs(suite_dir, args, jobs, git, suite_name):
    per_seed_rows = [per_seed_row(job, args) for job in jobs]
    aggregate = aggregate_rows(per_seed_rows)
    per_seed_path = suite_dir / "per_training_seed.csv"
    aggregate_path = suite_dir / "aggregate_across_training_seeds.csv"
    summary_path = suite_dir / "summary.json"
    friction_curve_path = suite_dir / "friction_curve_success_rate.png"
    write_csv(per_seed_path, per_seed_rows)
    write_csv(aggregate_path, aggregate)
    write_friction_curve(friction_curve_path, aggregate)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": suite_name,
                "created_at": current_time(),
                "evaluation_git": {
                    "sha": git["sha"],
                    "dirty": git["dirty"],
                },
                "protocol": {
                    "training_seeds": list(resolved_training_seeds(args)),
                    "episodes_per_training_seed_and_condition": args.episodes,
                    "episode_seed_range": [
                        args.episode_seed,
                        args.episode_seed + args.episodes - 1,
                    ],
                    "random_friction_range": [
                        args.friction_min,
                        args.friction_max,
                    ],
                    "fixed_frictions": list(resolved_fixed_frictions(args)),
                    "contact_force_threshold": args.contact_force_threshold,
                    "deterministic_policy": True,
                    "aggregation_unit": "independent_training_seed",
                },
                "per_training_seed": per_seed_rows,
                "aggregate_across_training_seeds": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        per_seed_path,
        aggregate_path,
        summary_path,
        friction_curve_path,
    )


def main():
    args = parse_args()
    validate_args(args)
    suite_name = args.suite_name or generated_suite_name(args)
    suite_dir = PROJECT_ROOT / "outputs" / "evaluation_suites" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    git = git_metadata()
    if git["dirty"] and not args.allow_dirty and not args.dry_run:
        raise RuntimeError(
            "Git working tree is dirty. Commit first or use --allow-dirty "
            "only for development.\n" + git["status"]
        )

    lock_stream = (suite_dir / "suite.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise RuntimeError("This formal evaluation suite is already running.") from exc
    lock_stream.write(f"pid={os.getpid()}\n")
    lock_stream.flush()

    jobs = build_jobs(args, suite_name)
    total_episodes = len(jobs) * args.episodes
    print(
        f"Formal suite: {suite_name}\n"
        f"Training seeds: {list(resolved_training_seeds(args))}\n"
        f"Jobs: {len(jobs)}; total episodes: {total_episodes}",
        flush=True,
    )
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-contact-pushing")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    logs_dir = suite_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        for index, job in enumerate(jobs, start=1):
            command = evaluation_command(job, args)
            print(
                f"[{index}/{len(jobs)}] train_seed={job['training_seed']} "
                f"{job['condition']['label']} {job['model']}",
                flush=True,
            )
            if args.dry_run:
                print("  " + " ".join(command), flush=True)
                continue
            validate_model_run(job)
            if completed_evaluation_is_compatible(job, args):
                print("  Compatible result exists; skipping.", flush=True)
                continue
            log_path = logs_dir / (
                f"trainseed{job['training_seed']}_"
                f"{job['condition']['label']}_{job['model']}.log"
            )
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(f"  Completed; log={log_path}", flush=True)

        if args.dry_run:
            print("Dry run complete; no evaluation was executed.")
            return
        paths = write_outputs(suite_dir, args, jobs, git, suite_name)
        print("Formal evaluation complete.")
        for path in paths:
            print(path)
    finally:
        lock_stream.close()


if __name__ == "__main__":
    main()
