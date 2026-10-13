import json
import sys
from types import SimpleNamespace

import pytest

from scripts import (
    evaluate_ablation_suite,
    evaluate_feedback_baseline,
    evaluate_ppo,
    screen_k4_checkpoints,
    train_normalized_ablation_suite,
    train_ppo,
)


def test_ablation_suite_defaults_to_requested_overnight_matrix(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_ablation_suite.py"],
    )
    args = evaluate_ablation_suite.parse_args()
    conditions = evaluate_ablation_suite.build_conditions(
        args,
        evaluate_ablation_suite.generated_suite_name(args),
    )

    assert args.episodes == 500
    assert args.seed == 4000
    assert args.run_set == "normalized"
    assert args.contact_force_threshold == 0.20
    assert len(conditions) == 3
    assert [item["friction_range"] for item in conditions] == [
        [0.3, 0.9],
        [0.9, 0.9],
        [1.0, 1.0],
    ]
    assert len(evaluate_ablation_suite.RUNS) * len(conditions) == 15
    assert all(
        "formal_norm_" in run_name
        for _, run_name in evaluate_ablation_suite.RUNS
    )


def test_k4_checkpoint_screen_defaults_to_paired_high_friction(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["screen_k4_checkpoints.py"],
    )
    args = screen_k4_checkpoints.parse_args()

    assert args.episodes == 20
    assert args.workers == 4
    assert args.seed == 6000
    assert screen_k4_checkpoints.DEFAULT_FRICTIONS == (0.9, 1.0)
    assert screen_k4_checkpoints.generated_screen_name(args) == (
        "k4_high_friction_n20_seeds_6000_6019"
    )


def test_k4_checkpoint_discovery_sorts_by_training_step(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "ppo_100000_steps.zip").touch()
    (checkpoint_dir / "ppo_50000_steps.zip").touch()
    (checkpoint_dir / "unrelated.zip").touch()

    checkpoints = screen_k4_checkpoints.discover_checkpoints(
        tmp_path
    )

    assert [step for step, _ in checkpoints] == [50000, 100000]


def test_normalized_training_suite_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_normalized_ablation_suite.py"],
    )
    args = train_normalized_ablation_suite.parse_args()
    jobs = [
        train_normalized_ablation_suite.expected_run_configuration(
            args,
            label,
            mode,
            history,
        )
        for label, mode, history in train_normalized_ablation_suite.RUNS
    ]

    assert args.timesteps == 500_000
    assert args.seed == 42
    assert args.device == "cpu"
    assert len(jobs) == 5
    assert all(job["normalization"] == "fixed_v1" for job in jobs)
    assert all(job["network"]["actor"] == [128, 128] for job in jobs)
    assert [job["history_length"] for job in jobs] == [1, 1, 1, 4, 4]


def test_train_defaults_to_v0(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ppo.py"])
    args = train_ppo.parse_args()
    assert args.env_version == "v0"


def test_train_accepts_v1(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_ppo.py", "--env-version", "v1"],
    )
    args = train_ppo.parse_args()
    assert args.env_version == "v1"


def test_train_accepts_formal_force_history_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--observation-mode",
            "posvel_force11",
            "--history-length",
            "4",
        ],
    )
    args = train_ppo.parse_args()
    assert args.observation_mode == "posvel_force11"
    assert args.history_length == 4


def test_fresh_training_defaults_to_fixed_normalization(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ppo.py"])
    args = train_ppo.parse_args()
    effective = args.normalization or (
        "none" if args.resume else "fixed_v1"
    )
    assert effective == "fixed_v1"


def test_stratified_metrics_include_force_safety():
    rows = [
        {
            "success": True,
            "episode_return": 2.0,
            "episode_length": 10,
            "final_goal_distance": 0.05,
            "contact_friction": 0.4,
            "initial_goal_distance": 0.19,
            "initial_plate_yaw_offset": 0.1,
            "peak_contact_force_xy": 0.3,
            "contact_force_sample_count": 10,
            "contact_force_exceedance_count": 2,
        },
        {
            "success": False,
            "episode_return": 0.0,
            "episode_length": 20,
            "final_goal_distance": 0.2,
            "contact_friction": 0.8,
            "initial_goal_distance": 0.25,
            "initial_plate_yaw_offset": -0.4,
            "peak_contact_force_xy": 0.1,
            "contact_force_sample_count": 20,
            "contact_force_exceedance_count": 0,
        },
    ]

    metrics = evaluate_ppo.build_stratified_metrics(rows)

    assert metrics["contact_friction"][0]["success_rate"] == 1.0
    assert metrics["contact_friction"][2]["success_rate"] == 0.0
    assert metrics["initial_goal_distance"][2]["episode_count"] == 1
    assert metrics["absolute_initial_yaw_offset"][2]["episode_count"] == 1


def test_feedback_evaluation_defaults_to_pi_no_release(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_feedback_baseline.py"],
    )
    args = evaluate_feedback_baseline.parse_args()
    assert args.controller_variant == "pi_no_release"


def test_feedback_evaluation_accepts_settle_variant(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_feedback_baseline.py",
            "--controller-variant",
            "hybrid_pi_settle",
        ],
    )
    args = evaluate_feedback_baseline.parse_args()
    assert args.controller_variant == "hybrid_pi_settle"


@pytest.mark.parametrize(
    "fixed,minimum,maximum,expected",
    [
        (0.6, None, None, (0.6, 0.6)),
        (None, 0.3, 0.9, (0.3, 0.9)),
        (None, None, None, None),
    ],
)
def test_evaluation_friction_resolution(
    fixed,
    minimum,
    maximum,
    expected,
):
    args = SimpleNamespace(
        friction=fixed,
        friction_min=minimum,
        friction_max=maximum,
    )
    assert evaluate_ppo.resolve_friction_range(args) == expected


@pytest.mark.parametrize(
    "fixed,minimum,maximum",
    [
        (0.6, 0.3, 0.9),
        (None, 0.3, None),
        (None, 0.9, 0.3),
        (float("nan"), None, None),
        (None, float("inf"), 0.9),
    ],
)
def test_evaluation_rejects_invalid_friction(
    fixed,
    minimum,
    maximum,
):
    args = SimpleNamespace(
        friction=fixed,
        friction_min=minimum,
        friction_max=maximum,
    )
    with pytest.raises(ValueError):
        evaluate_ppo.resolve_friction_range(args)


def test_old_run_config_defaults_to_v0(tmp_path):
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "environment": {
                    "observation_mode": "pos6",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_model.zip").touch()

    args = SimpleNamespace(
        run_dir=str(run_dir),
        model=None,
        output_dir=None,
        observation_mode=None,
        env_version=None,
    )
    context = evaluate_ppo.resolve_evaluation_context(args)

    assert context["training_env_version"] == "v0"
    assert context["evaluation_env_version"] == "v0"
    assert context["history_length"] == 1


def test_evaluation_reads_history_from_run_config(tmp_path):
    run_dir = tmp_path / "history_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "environment": {
                    "env_version": "v1",
                    "observation_mode": "posvel_force11",
                    "history_length": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_model.zip").touch()

    args = SimpleNamespace(
        run_dir=str(run_dir),
        model=None,
        output_dir=None,
        observation_mode=None,
        history_length=None,
        env_version=None,
    )
    context = evaluate_ppo.resolve_evaluation_context(args)

    assert context["observation_mode"] == "posvel_force11"
    assert context["history_length"] == 4


def test_evaluation_rejects_history_mismatch(tmp_path):
    run_dir = tmp_path / "history_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "environment": {
                    "env_version": "v1",
                    "observation_mode": "posvel9",
                    "history_length": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_model.zip").touch()

    args = SimpleNamespace(
        run_dir=str(run_dir),
        model=None,
        output_dir=None,
        observation_mode=None,
        history_length=1,
        env_version=None,
    )

    with pytest.raises(
        ValueError,
        match="history-length does not match",
    ):
        evaluate_ppo.resolve_evaluation_context(args)


def test_evaluation_version_can_override_training_version(tmp_path):
    run_dir = tmp_path / "v0_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "environment": {
                    "env_version": "v0",
                    "observation_mode": "pos6",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_model.zip").touch()

    args = SimpleNamespace(
        run_dir=str(run_dir),
        model=None,
        output_dir=None,
        observation_mode=None,
        env_version="v1",
    )
    context = evaluate_ppo.resolve_evaluation_context(args)

    assert context["training_env_version"] == "v0"
    assert context["evaluation_env_version"] == "v1"
