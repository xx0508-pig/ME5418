"""Train or continue PPO with optional per-episode friction randomization."""
import argparse
import shlex
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs import ENV_IDS, make_pushing_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

def get_git_metadata():
    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    git_sha = (
        sha_result.stdout.strip()
        if sha_result.returncode == 0
        else None
    )

    git_dirty = bool(status_result.stdout.strip())

    return {
        "sha": git_sha,
        "dirty": git_dirty,
    }

def get_package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None

def get_dependency_versions():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "stable_baselines3": get_package_version(
            "stable-baselines3"
        ),
        "gymnasium": get_package_version("gymnasium"),
        "mujoco": get_package_version("mujoco"),
        "numpy": get_package_version("numpy"),
        "torch": get_package_version("torch"),
        "libero": get_package_version("hf-libero"),
    }

def write_config(path, config):
    path.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-name", default="ppo_fixed_friction")
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Unique output directory name. "
            "A timestamped name is generated when omitted."
        ),
    )

    parser.add_argument(
        "--env-version",
        choices=sorted(ENV_IDS),
        default="v0",
        help=(
            "Environment benchmark version. "
            "Defaults to v0 for backward compatibility."
        ),
    )
    parser.add_argument(
        "--observation-mode",
        choices=sorted(OBSERVATION_MODES),
        default="pos6",
    )
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

    observation_modes = {
        "pos6": (False, False),
        "posvel9": (True, False),
        "posforce9": (False, True),
        "posvelforce12": (True, True),
    }
    include_velocity, include_force = observation_modes[
        args.observation_mode
    ]

    observation_dim = (
        6
        + (3 if include_velocity else 0)
        + (3 if include_force else 0)
    )

    if (args.friction_min is None) != (args.friction_max is None):
        raise ValueError(
            "--friction-min and --friction-max must be provided together."
        )

    friction_range = None
    if args.friction_min is not None:
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

        friction_range = (
            float(args.friction_min),
            float(args.friction_max),
        )

    generated_run_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + args.save_name
    )
    run_name = args.run_name or generated_run_name

    if not run_name or run_name in {".", ".."}:
        raise ValueError("--run-name must be a non-empty directory name.")

    if Path(run_name).name != run_name:
        raise ValueError(
            "--run-name must be one directory name, not a path."
        )

    run_dir = Path("outputs") / "runs" / run_name

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}"
        )

    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    evaluations_dir = run_dir / "evaluations"
    config_path = run_dir / "config.json"
    final_model_path = run_dir / "final_model"
    final_model_zip = run_dir / "final_model.zip"

    checkpoint_dir.mkdir(parents=True)
    tensorboard_dir.mkdir(parents=True)
    evaluations_dir.mkdir(parents=True)

    info_keywords = [
        "is_success",
        "goal_distance",
        "friction",
        "contact_friction",
    ]

    if args.env_version == "v1":
        info_keywords.extend(
            [
                "libero_success",
                "distance_success",
                "stable_success",
                "stable_count",
                "object_speed",
                "termination_reason",
                "invalid_state",
                "out_of_workspace",
                "time_limit",
            ]
        )

    info_keywords = tuple(info_keywords)

    config = {
        "schema_version": 1,
        "status": "created",
        "created_at": current_time(),
        "command": {
            "argv": list(sys.argv),
            "shell": shlex.join(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
        },
        "run": {
            "run_name": run_name,
            "run_directory": str(run_dir.resolve()),
            "save_name": args.save_name,
        },
        "training": {
            "requested_timesteps": args.timesteps,
            "seed": args.seed,
            "checkpoint_frequency": args.checkpoint_freq,
            "resume_from": (
                str(Path(args.resume).resolve())
                if args.resume
                else None
            ),
            "ent_coef_argument": args.ent_coef,
        },
        "environment": {
            "env_version": args.env_version,
            "env_id": ENV_IDS[args.env_version],
            "observation_mode": args.observation_mode,
            "observation_dimension": observation_dim,
            "include_object_velocity": include_velocity,
            "include_contact_force": include_force,
            "requested_friction_range": (
                list(friction_range)
                if friction_range is not None
                else None
            ),
            "runtime_friction_validation": True,
        },
        "artifacts": {
            "config_path": str(config_path.resolve()),
            "final_model_path": str(final_model_zip.resolve()),
            "checkpoint_directory": str(checkpoint_dir.resolve()),
            "tensorboard_directory": str(tensorboard_dir.resolve()),
            "evaluations_directory": str(evaluations_dir.resolve()),
        },
        "git": get_git_metadata(),
        "dependencies": get_dependency_versions(),
    }

    write_config(config_path, config)

    env = None

    try:
        raw_env = make_pushing_env(
            args.env_version,
            friction_range=friction_range,
            include_object_velocity=include_velocity,
            include_contact_force=include_force,
        )

        env = Monitor(
            raw_env,
            info_keywords=info_keywords,
        )

        config["environment"]["bddl_file"] = (
            raw_env.base_env.bddl_file
        )

        if args.env_version == "v1":
            config["environment"]["success_definition"] = (
                "distance_and_speed_stable"
            )
            config["environment"]["benchmark_parameters"] = {
                "success_radius": raw_env.success_radius,
                "object_speed_threshold": (
                    raw_env.object_speed_threshold
                ),
                "stable_steps_required": (
                    raw_env.stable_steps_required
                ),
                "initial_yaw_range": list(
                    raw_env.initial_yaw_range
                ),
                "workspace_margin": raw_env.workspace_margin,
                "workspace_z_tolerance": (
                    raw_env.workspace_z_tolerance
                ),
            }
        else:
            config["environment"]["success_definition"] = (
                "legacy_libero_or_distance"
            )

        write_config(config_path, config)

        if args.resume:
            resume_path = Path(args.resume).resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(
                    f"Resume model does not exist: {resume_path}"
                )

            model = PPO.load(
                str(resume_path),
                env=env,
                tensorboard_log=str(tensorboard_dir),
                seed=args.seed,
            )

            if args.ent_coef is not None:
                model.ent_coef = args.ent_coef

            print(
                f"Resuming {resume_path} at "
                f"{model.num_timesteps:,} steps"
            )
            reset_num_timesteps = False

        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=(
                    args.ent_coef
                    if args.ent_coef is not None
                    else 0.01
                ),
                verbose=1,
                tensorboard_log=str(tensorboard_dir),
                seed=args.seed,
            )
            reset_num_timesteps = True

        callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo",
        )

        config["status"] = "running"
        config["started_at"] = current_time()
        config["training"].update(
            {
                "reset_num_timesteps": reset_num_timesteps,
                "starting_num_timesteps": int(model.num_timesteps),
                "effective_ent_coef": float(model.ent_coef),
                "effective_learning_rate": float(
                    model.lr_schedule(1.0)
                ),
                "effective_clip_range": float(
                    model.clip_range(1.0)
                ),
            }
        )
        config["model_spaces"] = {
            "observation_shape": list(
                model.observation_space.shape
            ),
            "action_shape": list(model.action_space.shape),
        }

        write_config(config_path, config)

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=args.save_name,
            reset_num_timesteps=reset_num_timesteps,
        )

        model.save(final_model_path)

        config["status"] = "completed"
        config["finished_at"] = current_time()
        config["training"]["final_num_timesteps"] = int(
            model.num_timesteps
        )
        config["artifacts"]["final_model_exists"] = (
            final_model_zip.exists()
        )

        write_config(config_path, config)

        print(f"Saved final model to {final_model_zip}")
        print(f"Saved run metadata to {config_path}")

    except Exception as exc:
        config["status"] = "failed"
        config["finished_at"] = current_time()
        config["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        write_config(config_path, config)
        raise

    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
