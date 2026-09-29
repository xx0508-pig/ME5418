"""Evaluate the conventional PI feedback baseline on benchmark v1."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from controllers.conventional_feedback import (
    CONTROLLER_VARIANTS,
    PI_NO_RELEASE,
    ConventionalFeedbackController,
    make_feedback_controller_config,
)
from envs import V1_ENV_ID, make_pushing_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_FREQUENCY_HZ = 20


def current_time():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def write_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def dependency_versions():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "gymnasium": package_version("gymnasium"),
        "libero": package_version("hf-libero"),
        "mujoco": package_version("mujoco"),
        "numpy": package_version("numpy"),
    }


def git_metadata():
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

    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    return {
        "sha": (
            sha_result.stdout.strip()
            if sha_result.returncode == 0
            else None
        ),
        "branch": (
            branch_result.stdout.strip()
            if branch_result.returncode == 0
            else None
        ),
        "dirty": bool(
            status_result.stdout.strip()
        ),
    }


def optional_float(value):
    if value is None:
        return None
    return float(value)


def optional_bool(value):
    if value is None:
        return None
    return bool(value)


def mean_or_none(values):
    if not values:
        return None
    return float(np.mean(values))


def std_or_none(values):
    if not values:
        return None
    return float(np.std(values))


def range_or_none(values):
    if not values:
        return None

    return [
        float(min(values)),
        float(max(values)),
    ]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--controller-variant",
        choices=CONTROLLER_VARIANTS,
        default=PI_NO_RELEASE,
        help=(
            "pi_no_release preserves the original PI baseline; "
            "hybrid_pi_settle adds an explicit contact-release phase."
        ),
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Unique output directory name under "
            "outputs/runs."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Parent directory for the baseline run.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--video-episodes",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="Evaluate one fixed friction value.",
    )

    parser.add_argument(
        "--friction-min",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--friction-max",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--velocity-kp",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--velocity-ki",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-object-speed",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--slowdown-distance",
        type=float,
        default=0.14,
    )

    return parser.parse_args()


def resolve_friction_range(args):
    has_min = args.friction_min is not None
    has_max = args.friction_max is not None

    if args.friction is not None and (
        has_min or has_max
    ):
        raise ValueError(
            "--friction cannot be combined with "
            "--friction-min/--friction-max."
        )

    if has_min != has_max:
        raise ValueError(
            "--friction-min and --friction-max "
            "must be provided together."
        )

    if args.friction is not None:
        values = (
            float(args.friction),
            float(args.friction),
        )
    elif has_min:
        values = (
            float(args.friction_min),
            float(args.friction_max),
        )
    else:
        # Default development friction range.
        values = (0.3, 0.9)

    if not np.isfinite(values).all():
        raise ValueError(
            "Friction values must be finite."
        )

    if values[0] <= 0 or values[1] <= 0:
        raise ValueError(
            "Friction values must be positive."
        )

    if values[0] > values[1]:
        raise ValueError(
            "Minimum friction cannot exceed maximum."
        )

    return values


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than zero."
        )

    if args.video_episodes < 0:
        raise ValueError(
            "--video-episodes cannot be negative."
        )

    if args.video_episodes > args.episodes:
        raise ValueError(
            "--video-episodes cannot exceed "
            "--episodes."
        )

    if (
        args.velocity_kp < 0
        or args.velocity_ki < 0
    ):
        raise ValueError(
            "PI gains cannot be negative."
        )

    if args.max_object_speed <= 0:
        raise ValueError(
            "--max-object-speed must be positive."
        )

    if args.slowdown_distance <= 0.06:
        raise ValueError(
            "--slowdown-distance must exceed "
            "the 0.06 m success radius."
        )


def resolve_run_name(value):
    run_name = value or (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_feedback_pi_v1"
    )

    if (
        not run_name
        or run_name in {".", ".."}
        or Path(run_name).name != run_name
    ):
        raise ValueError(
            "--run-name must be one "
            "directory-safe name."
        )

    return run_name


def main():
    args = parse_args()
    validate_args(args)

    friction_range = resolve_friction_range(args)
    run_name = resolve_run_name(args.run_name)

    run_dir = (
        Path(args.output_root) / run_name
    )

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: "
            f"{run_dir}"
        )

    run_dir.mkdir(parents=True)

    config_path = run_dir / "config.json"
    csv_path = run_dir / "episodes.csv"
    summary_path = run_dir / "summary.json"
    video_path = (
        run_dir / "feedback_baseline.mp4"
    )

    controller_config = (
        make_feedback_controller_config(
            args.controller_variant,
            velocity_kp=args.velocity_kp,
            velocity_ki=args.velocity_ki,
            max_object_speed=(
                args.max_object_speed
            ),
            slowdown_distance=(
                args.slowdown_distance
            ),
        )
    )

    controller = (
        ConventionalFeedbackController(
            controller_config
        )
    )

    config = {
        "schema_version": 1,
        "status": "created",
        "created_at": current_time(),
        "command": {
            "argv": list(sys.argv),
            "shell": shlex.join(sys.argv),
            "working_directory": str(
                Path.cwd().resolve()
            ),
        },
        "run": {
            "run_name": run_name,
            "run_directory": str(
                run_dir.resolve()
            ),
        },
        "environment": {
            "env_version": "v1",
            "env_id": V1_ENV_ID,
            "observation_mode": (
                "posvelforce12"
            ),
            # 控制器只读取 obs[0:9]。
            "controller_observation_slice": [
                0,
                9,
            ],
            # obs[9:11] 仅用于测量 XY force。
            "force_metric_slice": [
                9,
                11,
            ],
            "uses_contact_force_for_control": (
                False
            ),
            "uses_friction_for_control": False,
            "max_steps": 300,
            "action_scale": 0.5,
            "control_frequency_hz": (
                CONTROL_FREQUENCY_HZ
            ),
            "requested_friction_range": list(
                friction_range
            ),
        },
        "controller": {
            "family": "pi_velocity_feedback",
            "variant": args.controller_variant,
            "name": args.controller_variant,
            "parameters": (
                controller_config.to_dict()
            ),
        },
        "evaluation": {
            "episodes": args.episodes,
            "base_seed": args.seed,
            "video_episodes": (
                args.video_episodes
            ),
        },
        "artifacts": {
            "config": str(
                config_path.resolve()
            ),
            "episodes_csv": str(
                csv_path.resolve()
            ),
            "summary_json": str(
                summary_path.resolve()
            ),
            "video": (
                str(video_path.resolve())
                if args.video_episodes > 0
                else None
            ),
        },
        "git": git_metadata(),
        "dependencies": dependency_versions(),
    }

    write_json(config_path, config)

    env = None
    frames = []
    results = []

    try:
        env = make_pushing_env(
            "v1",
            max_steps=300,
            action_scale=0.5,
            observation_size=256,
            friction_range=friction_range,
            include_object_velocity=True,
            include_contact_force=True,
        )

        config["environment"]["bddl_file"] = str(
            env.base_env.bddl_file
        )

        config["environment"][
            "benchmark_parameters"
        ] = {
            "success_radius": (
                env.success_radius
            ),
            "object_speed_threshold": (
                env.object_speed_threshold
            ),
            "stable_steps_required": (
                env.stable_steps_required
            ),
            "initial_yaw_range": list(
                env.initial_yaw_range
            ),
        }

        config["status"] = "running"
        config["started_at"] = current_time()
        write_json(config_path, config)

        for episode_index in range(
            args.episodes
        ):
            episode_seed = (
                args.seed + episode_index
            )

            controller.reset()

            obs, reset_info = env.reset(
                seed=episode_seed
            )

            if np.asarray(obs).shape != (12,):
                raise RuntimeError(
                    "Expected a 12D "
                    "posvelforce observation, "
                    f"got {np.asarray(obs).shape}"
                )

            info = reset_info
            episode_return = 0.0
            episode_length = 0

            control_effort = 0.0
            max_action_norm = 0.0

            max_contact_force_xy = float(
                np.linalg.norm(obs[9:11])
            )

            min_goal_distance = optional_float(
                info.get("goal_distance")
            )

            ever_distance_success = bool(
                info.get(
                    "distance_success",
                    False,
                )
            )

            max_stable_count = int(
                info.get("stable_count", 0)
            )

            first_distance_success_step = (
                0
                if ever_distance_success
                else None
            )

            record_video = (
                episode_index
                < args.video_episodes
            )

            if record_video:
                frames.extend(
                    [env.render()] * 8
                )

            while True:
                # Controller can see positions and
                # object velocity only.
                #
                # Contact force obs[9:12] and friction
                # are excluded from the controller.
                action = controller.act(obs[:9])

                if not env.action_space.contains(
                    action
                ):
                    raise RuntimeError(
                        "Controller returned "
                        f"invalid action: {action}"
                    )

                action_norm = float(
                    np.linalg.norm(action)
                )

                control_effort += (
                    float(
                        np.dot(action, action)
                    )
                    * controller_config.control_dt
                )

                max_action_norm = max(
                    max_action_norm,
                    action_norm,
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                episode_return += float(reward)
                episode_length += 1

                # Proposal 使用 XY contact-force。
                # 这里只进行测量，不传给控制器。
                force_xy = float(
                    np.linalg.norm(
                        obs[9:11]
                    )
                )

                max_contact_force_xy = max(
                    max_contact_force_xy,
                    force_xy,
                )

                goal_distance = optional_float(
                    info.get("goal_distance")
                )

                if goal_distance is not None:
                    if min_goal_distance is None:
                        min_goal_distance = (
                            goal_distance
                        )
                    else:
                        min_goal_distance = min(
                            min_goal_distance,
                            goal_distance,
                        )

                distance_success = bool(
                    info.get(
                        "distance_success",
                        False,
                    )
                )

                if (
                    distance_success
                    and first_distance_success_step
                    is None
                ):
                    first_distance_success_step = (
                        episode_length
                    )

                ever_distance_success = (
                    ever_distance_success
                    or distance_success
                )

                max_stable_count = max(
                    max_stable_count,
                    int(
                        info.get(
                            "stable_count",
                            0,
                        )
                    ),
                )

                if record_video:
                    frames.append(env.render())

                if terminated or truncated:
                    break

            success = bool(
                info.get(
                    "stable_success",
                    info.get(
                        "is_success",
                        False,
                    ),
                )
            )

            time_from_first_entry = None

            if (
                success
                and first_distance_success_step
                is not None
            ):
                time_from_first_entry = (
                    (
                        episode_length
                        - first_distance_success_step
                        + 1
                    )
                    * controller_config.control_dt
                )

            result = {
                "episode": episode_index + 1,
                "seed": episode_seed,
                "success": success,
                "stable_success": optional_bool(
                    info.get("stable_success")
                ),
                "distance_success": optional_bool(
                    info.get(
                        "distance_success"
                    )
                ),
                "ever_distance_success": (
                    ever_distance_success
                ),
                "stable_count": info.get(
                    "stable_count"
                ),
                "max_stable_count": (
                    max_stable_count
                ),
                "termination_reason": info.get(
                    "termination_reason"
                ),
                "invalid_state": optional_bool(
                    info.get("invalid_state")
                ),
                "out_of_workspace": optional_bool(
                    info.get(
                        "out_of_workspace"
                    )
                ),
                "time_limit": optional_bool(
                    info.get("time_limit")
                ),
                "episode_return": (
                    episode_return
                ),
                "episode_length": (
                    episode_length
                ),
                "completion_time_seconds": (
                    episode_length
                    * controller_config.control_dt
                ),
                "first_distance_success_step": (
                    first_distance_success_step
                ),
                "time_from_first_entry_to_"
                "success_seconds": (
                    time_from_first_entry
                ),
                "final_goal_distance": (
                    optional_float(
                        info.get(
                            "goal_distance"
                        )
                    )
                ),
                "min_goal_distance": (
                    min_goal_distance
                ),
                "final_object_speed": (
                    optional_float(
                        info.get(
                            "object_speed"
                        )
                    )
                ),
                "control_effort": (
                    control_effort
                ),
                "max_action_norm": (
                    max_action_norm
                ),
                "max_contact_force_xy": (
                    max_contact_force_xy
                ),
                "final_controller_phase": (
                    controller.phase.value
                ),
                "initial_plate_x": (
                    optional_float(
                        reset_info.get(
                            "initial_plate_x"
                        )
                    )
                ),
                "initial_plate_y": (
                    optional_float(
                        reset_info.get(
                            "initial_plate_y"
                        )
                    )
                ),
                "initial_plate_z": (
                    optional_float(
                        reset_info.get(
                            "initial_plate_z"
                        )
                    )
                ),
                "initial_plate_yaw": (
                    optional_float(
                        reset_info.get(
                            "initial_plate_yaw"
                        )
                    )
                ),
                "initial_plate_yaw_offset": (
                    optional_float(
                        reset_info.get(
                            "initial_plate_yaw_offset"
                        )
                    )
                ),
                "requested_friction": (
                    optional_float(
                        info.get(
                            "requested_friction"
                        )
                    )
                ),
                "contact_friction": (
                    optional_float(
                        info.get(
                            "contact_friction"
                        )
                    )
                ),
            }

            results.append(result)

            goal_text = (
                f"{result['final_goal_distance']:.4f}"
                if result[
                    "final_goal_distance"
                ]
                is not None
                else "invalid"
            )

            print(
                f"episode="
                f"{result['episode']:02d} "
                f"seed={episode_seed} "
                f"success={success} "
                f"reason="
                f"{result['termination_reason']} "
                f"steps={episode_length} "
                f"final_goal={goal_text} "
                f"effort="
                f"{control_effort:.3f} "
                f"max_force_xy="
                f"{max_contact_force_xy:.3f}"
            )

            if record_video:
                frames.extend(
                    [env.render()] * 12
                )

        fieldnames = list(results[0])

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

        if frames:
            imageio.mimsave(
                video_path,
                frames,
                fps=CONTROL_FREQUENCY_HZ,
            )

        successes = [
            float(row["success"])
            for row in results
        ]

        returns = [
            row["episode_return"]
            for row in results
        ]

        lengths = [
            row["episode_length"]
            for row in results
        ]

        final_distances = [
            row["final_goal_distance"]
            for row in results
            if row["final_goal_distance"]
            is not None
        ]

        min_distances = [
            row["min_goal_distance"]
            for row in results
            if row["min_goal_distance"]
            is not None
        ]

        efforts = [
            row["control_effort"]
            for row in results
        ]

        max_forces = [
            row["max_contact_force_xy"]
            for row in results
        ]

        completion_times = [
            row["completion_time_seconds"]
            for row in results
            if row["success"]
        ]

        settling_times = [
            row[
                "time_from_first_entry_to_"
                "success_seconds"
            ]
            for row in results
            if row[
                "time_from_first_entry_to_"
                "success_seconds"
            ]
            is not None
        ]

        requested_frictions = [
            row["requested_friction"]
            for row in results
            if row["requested_friction"]
            is not None
        ]

        contact_frictions = [
            row["contact_friction"]
            for row in results
            if row["contact_friction"]
            is not None
        ]

        termination_counts = dict(
            sorted(
                Counter(
                    row["termination_reason"]
                    for row in results
                ).items()
            )
        )

        summary = {
            "schema_version": 1,
            "created_at": current_time(),
            "run_name": run_name,
            "controller": (
                config["controller"]
            ),
            "environment": (
                config["environment"]
            ),
            "evaluation": (
                config["evaluation"]
            ),
            "git": config["git"],
            "metrics": {
                "success_count": int(
                    sum(successes)
                ),
                "success_rate": (
                    mean_or_none(successes)
                ),
                "ever_distance_success_count": (
                    int(
                        sum(
                            bool(
                                row[
                                    "ever_distance_"
                                    "success"
                                ]
                            )
                            for row in results
                        )
                    )
                ),
                "ever_distance_success_rate": (
                    mean_or_none(
                        [
                            float(
                                row[
                                    "ever_distance_"
                                    "success"
                                ]
                            )
                            for row in results
                        ]
                    )
                ),
                "mean_return": (
                    mean_or_none(returns)
                ),
                "std_return": (
                    std_or_none(returns)
                ),
                "mean_episode_length": (
                    mean_or_none(lengths)
                ),
                "std_episode_length": (
                    std_or_none(lengths)
                ),
                "mean_success_completion_"
                "time_seconds": (
                    mean_or_none(
                        completion_times
                    )
                ),
                "mean_time_from_first_entry_"
                "to_success_seconds": (
                    mean_or_none(
                        settling_times
                    )
                ),
                "mean_final_goal_distance": (
                    mean_or_none(
                        final_distances
                    )
                ),
                "std_final_goal_distance": (
                    std_or_none(
                        final_distances
                    )
                ),
                "mean_min_goal_distance": (
                    mean_or_none(
                        min_distances
                    )
                ),
                "mean_control_effort": (
                    mean_or_none(efforts)
                ),
                "std_control_effort": (
                    std_or_none(efforts)
                ),
                "mean_max_contact_force_xy": (
                    mean_or_none(max_forces)
                ),
                "max_contact_force_xy": (
                    max(max_forces)
                    if max_forces
                    else None
                ),
                "requested_friction_"
                "range_observed": (
                    range_or_none(
                        requested_frictions
                    )
                ),
                "contact_friction_"
                "range_observed": (
                    range_or_none(
                        contact_frictions
                    )
                ),
                "termination_counts": (
                    termination_counts
                ),
            },
            "artifacts": config["artifacts"],
        }

        write_json(
            summary_path,
            summary,
        )

        config["status"] = "completed"
        config["finished_at"] = (
            current_time()
        )

        write_json(
            config_path,
            config,
        )

    except Exception as exc:
        config["status"] = "failed"
        config["finished_at"] = (
            current_time()
        )
        config["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

        write_json(
            config_path,
            config,
        )

        raise

    finally:
        if env is not None:
            env.close()

    success_rate = mean_or_none(
        [
            float(row["success"])
            for row in results
        ]
    )

    print(
        "\nConventional feedback "
        "baseline summary"
    )

    print(
        f"success_rate={success_rate:.3f} "
        f"({int(sum(row['success'] for row in results))}"
        f"/{len(results)})"
    )

    print(
        "mean_control_effort="
        f"{mean_or_none(efforts):.3f}"
    )

    print(
        "mean_max_contact_force_xy="
        f"{mean_or_none(max_forces):.3f}"
    )

    print(f"episodes_csv={csv_path}")
    print(f"summary_json={summary_path}")

    if frames:
        print(f"video={video_path}")


if __name__ == "__main__":
    main()
