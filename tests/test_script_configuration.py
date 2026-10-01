import json
import sys
from types import SimpleNamespace

import pytest

from scripts import (
    evaluate_feedback_baseline,
    evaluate_ppo,
    train_ppo,
)


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
