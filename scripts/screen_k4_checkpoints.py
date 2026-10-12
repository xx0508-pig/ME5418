"""Screen all k=4 checkpoints on paired high-friction episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

K4_RUNS = (
    ("posvel9_k4", "formal_posvel9_k4_seed42"),
    ("posvel_force11_k4", "formal_posvel_force11_k4_seed42"),
)

DEFAULT_FRICTIONS = (0.9, 1.0)
CHECKPOINT_PATTERN = re.compile(r"ppo_(\d+)_steps\.zip$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every k=4 checkpoint on identical fixed "
            "high-friction episodes."
        )
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per checkpoint and friction value.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of checkpoint evaluations to run concurrently.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6000,
        help="First episode seed.",
    )
    parser.add_argument(
        "--screen-name",
        default=None,
        help="Optional output directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running evaluations.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero.")
    if not args.screen_name:
        return
    if args.screen_name in {".", ".."}:
        raise ValueError("--screen-name must be directory-safe.")
    if Path(args.screen_name).name != args.screen_name:
        raise ValueError("--screen-name must be one directory name.")


def generated_screen_name(args):
    final_seed = args.seed + args.episodes - 1
    return (
        f"k4_high_friction_n{args.episodes}_"
        f"seeds_{args.seed}_{final_seed}"
    )


def discover_checkpoints(run_directory):
    checkpoints = []
    for path in (run_directory / "checkpoints").glob("ppo_*_steps.zip"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), path.resolve()))
    checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError(
            f"No PPO checkpoints found in {run_directory}"
        )
    return checkpoints


def friction_slug(value):
    return f"{value:g}".replace(".", "p")


def evaluation_command(
    run_directory,
    checkpoint_path,
    output_directory,
    evaluation_name,
    friction,
    args,
):
    return [
        sys.executable,
        "-m",
        "scripts.evaluate_ppo",
        "--run-dir",
        str(run_directory.relative_to(PROJECT_ROOT)),
        "--model",
        str(checkpoint_path),
        "--output-dir",
        str(output_directory),
        "--evaluation-name",
        evaluation_name,
        "--episodes",
        str(args.episodes),
        "--video-episodes",
        "0",
        "--seed",
        str(args.seed),
        "--friction",
        str(friction),
        "--observation-stats",
    ]


def completed_evaluation_is_compatible(
    summary_path,
    episodes_path,
    observation_stats_path,
    checkpoint_path,
    friction,
    args,
):
    paths = (
        summary_path,
        episodes_path,
        observation_stats_path,
    )
    existing = [path.exists() for path in paths]
    if not any(existing):
        return False
    if not all(existing):
        raise RuntimeError(
            "Incomplete checkpoint evaluation artifacts in "
            f"{summary_path.parent}."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    compatible = (
        evaluation["episodes"] == args.episodes
        and evaluation["base_seed"] == args.seed
        and evaluation["requested_friction_range"]
        == [friction, friction]
        and Path(summary["model"]["path"]) == checkpoint_path
        and summary["metrics"]["observation_statistics"] is not None
    )
    if not compatible:
        raise RuntimeError(
            "Existing checkpoint evaluation uses different settings: "
            f"{summary_path}"
        )
    return True


def build_aggregate(screen_directory, jobs):
    rows = []
    for job in jobs:
        summary = json.loads(
            job["summary_path"].read_text(encoding="utf-8")
        )
        metrics = summary["metrics"]
        rows.append(
            {
                "model": job["model"],
                "checkpoint_steps": job["checkpoint_steps"],
                "friction": job["friction"],
                "episodes": summary["evaluation"]["episodes"],
                "success_count": metrics["success_count"],
                "success_rate": metrics["success_rate"],
                "mean_episode_length": metrics[
                    "mean_episode_length"
                ],
                "std_episode_length": metrics[
                    "std_episode_length"
                ],
                "mean_return": metrics["mean_return"],
                "std_return": metrics["std_return"],
                "termination_counts": json.dumps(
                    metrics["termination_counts"],
                    sort_keys=True,
                ),
                "observation_sample_count": metrics[
                    "observation_statistics"
                ]["sample_count"],
                "checkpoint_path": str(job["checkpoint_path"]),
                "summary_path": str(job["summary_path"]),
                "observation_stats_path": str(
                    job["observation_stats_path"]
                ),
            }
        )

    aggregate_path = screen_directory / "summary.csv"
    with aggregate_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ranking_rows = []
    for model, _ in K4_RUNS:
        model_rows = [row for row in rows if row["model"] == model]
        steps = sorted(
            {row["checkpoint_steps"] for row in model_rows}
        )
        for checkpoint_steps in steps:
            checkpoint_rows = [
                row
                for row in model_rows
                if row["checkpoint_steps"] == checkpoint_steps
            ]
            success_rates = [
                row["success_rate"] for row in checkpoint_rows
            ]
            mean_lengths = [
                row["mean_episode_length"] for row in checkpoint_rows
            ]
            mean_returns = [
                row["mean_return"] for row in checkpoint_rows
            ]
            ranking_rows.append(
                {
                    "model": model,
                    "checkpoint_steps": checkpoint_steps,
                    "worst_success_rate": min(success_rates),
                    "mean_success_rate": sum(success_rates)
                    / len(success_rates),
                    "mean_episode_length": sum(mean_lengths)
                    / len(mean_lengths),
                    "mean_return": sum(mean_returns)
                    / len(mean_returns),
                }
            )

    ranking_rows.sort(
        key=lambda row: (
            row["model"],
            -row["worst_success_rate"],
            -row["mean_success_rate"],
            row["mean_episode_length"],
        )
    )
    model_rank = {}
    for row in ranking_rows:
        rank = model_rank.get(row["model"], 0) + 1
        model_rank[row["model"]] = rank
        row["rank_within_model"] = rank

    ranking_path = screen_directory / "ranking.csv"
    with ranking_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=ranking_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(ranking_rows)

    json_path = screen_directory / "summary.json"
    json_path.write_text(
        json.dumps(
            {
                "screening_protocol": {
                    "episodes_per_checkpoint_and_friction": rows[0][
                        "episodes"
                    ],
                    "frictions": list(DEFAULT_FRICTIONS),
                    "ranking_order": [
                        "worst_success_rate descending",
                        "mean_success_rate descending",
                        "mean_episode_length ascending",
                    ],
                },
                "evaluations": rows,
                "ranking": ranking_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return aggregate_path, ranking_path, json_path


def main():
    args = parse_args()
    validate_args(args)
    screen_name = args.screen_name or generated_screen_name(args)
    screen_directory = (
        PROJECT_ROOT / "outputs" / "checkpoint_screening" / screen_name
    )

    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault(
        "MPLCONFIGDIR",
        "/tmp/matplotlib-contact-pushing",
    )

    jobs = []
    for model, run_name in K4_RUNS:
        run_directory = PROJECT_ROOT / "outputs" / "runs" / run_name
        for checkpoint_steps, checkpoint_path in discover_checkpoints(
            run_directory
        ):
            checkpoint_directory = (
                screen_directory
                / model
                / f"checkpoint_{checkpoint_steps}"
            )
            for friction in DEFAULT_FRICTIONS:
                evaluation_name = (
                    f"fixed_mu_{friction_slug(friction)}"
                )
                jobs.append(
                    {
                        "model": model,
                        "run_directory": run_directory,
                        "checkpoint_steps": checkpoint_steps,
                        "checkpoint_path": checkpoint_path,
                        "friction": friction,
                        "evaluation_name": evaluation_name,
                        "output_directory": checkpoint_directory,
                        "summary_path": checkpoint_directory
                        / f"{evaluation_name}_summary.json",
                        "episodes_path": checkpoint_directory
                        / f"{evaluation_name}_episodes.csv",
                        "observation_stats_path": checkpoint_directory
                        / f"{evaluation_name}_observation_stats.csv",
                        "log_path": checkpoint_directory
                        / f"{evaluation_name}.log",
                    }
                )

    print(f"Screen name: {screen_name}", flush=True)
    print(
        f"Jobs: {len(jobs)}, episodes/job: {args.episodes}, "
        f"total episodes: {len(jobs) * args.episodes}",
        flush=True,
    )

    pending = []
    for index, job in enumerate(jobs, start=1):
        command = evaluation_command(
            job["run_directory"],
            job["checkpoint_path"],
            job["output_directory"],
            job["evaluation_name"],
            job["friction"],
            args,
        )
        print(
            f"\n[{index}/{len(jobs)}] {job['model']} "
            f"steps={job['checkpoint_steps']} "
            f"friction={job['friction']}",
            flush=True,
        )
        if args.dry_run:
            print("  " + " ".join(command), flush=True)
            continue
        if completed_evaluation_is_compatible(
            job["summary_path"],
            job["episodes_path"],
            job["observation_stats_path"],
            job["checkpoint_path"],
            job["friction"],
            args,
        ):
            print("  Already complete; skipping.", flush=True)
            continue
        pending.append((index, job, command))

    if args.dry_run:
        print("\nDry run complete; no evaluation was executed.")
        return

    def run_job(item):
        index, job, command = item
        job["output_directory"].mkdir(parents=True, exist_ok=True)
        with job["log_path"].open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return index, job

    if pending:
        print(
            f"Running {len(pending)} pending jobs with "
            f"{args.workers} workers.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_job, item): item for item in pending
            }
            for future in as_completed(futures):
                index, job = future.result()
                print(
                    f"Completed [{index}/{len(jobs)}] "
                    f"{job['model']} steps={job['checkpoint_steps']} "
                    f"friction={job['friction']}",
                    flush=True,
                )

    screen_directory.mkdir(parents=True, exist_ok=True)
    aggregate_path, ranking_path, json_path = build_aggregate(
        screen_directory,
        jobs,
    )
    print("\nCheckpoint screening complete.")
    print(f"Summary CSV: {aggregate_path}")
    print(f"Ranking CSV: {ranking_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
