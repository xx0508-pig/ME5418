"""Sequentially train the five fixed-normalization ablation models."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUNS = (
    ("pos6_k1", "pos6", 1),
    ("posvel9_k1", "posvel9", 1),
    ("posvel_force11_k1", "posvel_force11", 1),
    ("posvel9_k4", "posvel9", 4),
    ("posvel_force11_k4", "posvel_force11", 4),
)


def current_time():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train all five v1 observation/history ablations "
            "sequentially with fixed_v1 normalization."
        )
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--friction-min", type=float, default=0.3)
    parser.add_argument("--friction-max", type=float, default=0.9)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument(
        "--suite-name",
        default=None,
        help=(
            "Training-suite directory name. When omitted, it is generated "
            "from timesteps and seed."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        default="formal_norm",
        help="Run names become <prefix>_<ablation>_seed<seed>.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow formal training from a Git working tree with changes.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be greater than zero.")
    if args.checkpoint_freq <= 0:
        raise ValueError("--checkpoint-freq must be greater than zero.")
    if args.friction_min <= 0 or args.friction_max <= 0:
        raise ValueError("Friction bounds must be positive.")
    if args.friction_min > args.friction_max:
        raise ValueError("--friction-min cannot exceed --friction-max.")
    for value, name in (
        (args.suite_name, "--suite-name"),
        (args.run_prefix, "--run-prefix"),
    ):
        if value is None:
            continue
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError(f"{name} must be one directory-safe name.")


def resolved_suite_name(args):
    if args.suite_name:
        return args.suite_name
    return f"formal_normalized_v1_{args.timesteps}_seed{args.seed}"


def git_status():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_name(args, label):
    return f"{args.run_prefix}_{label}_seed{args.seed}"


def expected_run_configuration(args, label, mode, history_length):
    return {
        "label": label,
        "run_name": run_name(args, label),
        "env_version": "v1",
        "observation_mode": mode,
        "history_length": history_length,
        "normalization": "fixed_v1",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "friction_range": [args.friction_min, args.friction_max],
        "checkpoint_frequency": args.checkpoint_freq,
        "ent_coef": args.ent_coef,
        "device": args.device,
        "network": {"actor": [128, 128], "critic": [128, 128]},
    }


def completed_run_is_compatible(run_dir, expected):
    config_path = run_dir / "config.json"
    model_path = run_dir / "final_model.zip"
    if not run_dir.exists():
        return False
    if not config_path.is_file() or not model_path.is_file():
        raise RuntimeError(
            f"Incomplete run directory already exists: {run_dir}. "
            "Do not delete it until its logs/checkpoints have been inspected."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = config["environment"]
    training = config["training"]
    policy = config["policy"]
    normalization = environment.get("normalization", {})
    checks = {
        "status": config.get("status") == "completed",
        "env_version": environment.get("env_version") == "v1",
        "observation_mode": environment.get("observation_mode")
        == expected["observation_mode"],
        "history_length": environment.get("history_length")
        == expected["history_length"],
        "normalization": normalization.get("name") == "fixed_v1",
        "seed": training.get("seed") == expected["seed"],
        "requested_timesteps": training.get("requested_timesteps")
        == expected["timesteps"],
        "friction_range": environment.get("requested_friction_range")
        == expected["friction_range"],
        "actor_network": policy.get("actor_hidden_layers") == [128, 128],
        "critic_network": policy.get("critic_hidden_layers") == [128, 128],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Existing run is incompatible ({', '.join(failed)}): {run_dir}"
        )
    return True


def training_command(args, expected):
    return [
        sys.executable,
        "-m",
        "scripts.train_ppo",
        "--timesteps",
        str(args.timesteps),
        "--run-name",
        expected["run_name"],
        "--save-name",
        expected["label"],
        "--env-version",
        "v1",
        "--observation-mode",
        expected["observation_mode"],
        "--history-length",
        str(expected["history_length"]),
        "--normalization",
        "fixed_v1",
        "--friction-min",
        str(args.friction_min),
        "--friction-max",
        str(args.friction_max),
        "--seed",
        str(args.seed),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--ent-coef",
        str(args.ent_coef),
        "--device",
        args.device,
    ]


def write_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    suite_name = resolved_suite_name(args)
    dirty = git_status()
    if dirty and not args.allow_dirty and not args.dry_run:
        raise RuntimeError(
            "Git working tree is dirty. Commit the implementation first, "
            "or use --allow-dirty only for a non-formal development run.\n"
            + dirty
        )

    suite_dir = PROJECT_ROOT / "outputs" / "training_suites" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (suite_dir / "suite.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise RuntimeError("This training suite is already running.") from exc
    lock_stream.write(f"pid={os.getpid()}\n")
    lock_stream.flush()

    jobs = [
        expected_run_configuration(args, label, mode, history)
        for label, mode, history in RUNS
    ]
    manifest_path = suite_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "suite_name": suite_name,
        "created_at": current_time(),
        "status": "dry_run" if args.dry_run else "running",
        "sequential": True,
        "git_dirty_at_start": bool(dirty),
        "jobs": [{**job, "status": "pending"} for job in jobs],
    }
    write_manifest(manifest_path, manifest)

    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-contact-pushing")
    environment.setdefault("PYTHONUNBUFFERED", "1")

    try:
        for index, job in enumerate(jobs):
            run_dir = PROJECT_ROOT / "outputs" / "runs" / job["run_name"]
            command = training_command(args, job)
            print(
                f"[{index + 1}/{len(jobs)}] {job['label']} -> {job['run_name']}",
                flush=True,
            )
            print("  " + " ".join(command), flush=True)
            if args.dry_run:
                continue
            if completed_run_is_compatible(run_dir, job):
                manifest["jobs"][index]["status"] = "skipped_completed"
                write_manifest(manifest_path, manifest)
                print("  Compatible completed run found; skipping.", flush=True)
                continue

            log_path = suite_dir / f"{job['label']}.log"
            manifest["jobs"][index].update(
                {"status": "running", "started_at": current_time(), "log": str(log_path)}
            )
            write_manifest(manifest_path, manifest)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            manifest["jobs"][index].update(
                {"status": "completed", "finished_at": current_time()}
            )
            write_manifest(manifest_path, manifest)
            print("  Completed.", flush=True)
        manifest["status"] = "dry_run" if args.dry_run else "completed"
        manifest["finished_at"] = current_time()
        write_manifest(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = current_time()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_manifest(manifest_path, manifest)
        raise
    finally:
        lock_stream.close()

    print(f"Suite manifest: {manifest_path}")


if __name__ == "__main__":
    main()
