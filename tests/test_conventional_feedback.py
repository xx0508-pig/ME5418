import numpy as np
import pytest

from controllers.conventional_feedback import (
    ControllerPhase,
    ConventionalFeedbackController,
)


def make_observation(
    plate_minus_eef=(0.05, 0.0, -0.18),
    goal_minus_plate=(0.20, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
):
    return np.asarray(
        [
            *plate_minus_eef,
            *goal_minus_plate,
            *velocity,
        ],
        dtype=np.float32,
    )


def test_action_is_finite_and_bounded():
    controller = ConventionalFeedbackController()

    action = controller.act(
        make_observation(
            plate_minus_eef=(1.0, -1.0, 1.0),
        )
    )

    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 0.70


def test_controller_transitions_to_push():
    controller = ConventionalFeedbackController()

    # 已经到达盘子后上方。
    controller.act(make_observation())

    assert (
        controller.phase
        == ControllerPhase.DESCEND
    )

    # 已经降低到推动高度。
    controller.act(
        make_observation(
            plate_minus_eef=(
                0.05,
                0.0,
                -0.01,
            )
        )
    )

    assert (
        controller.phase
        == ControllerPhase.PUSH
    )


def test_speed_feedback_reduces_lead_when_too_fast():
    slow_controller = (
        ConventionalFeedbackController()
    )

    slow_controller.phase = ControllerPhase.PUSH

    slow_controller.act(
        make_observation(
            velocity=(0.0, 0.0, 0.0),
        )
    )

    slow_lead = (
        slow_controller.last_diagnostics[
            "lead_distance"
        ]
    )

    fast_controller = (
        ConventionalFeedbackController()
    )

    fast_controller.phase = ControllerPhase.PUSH

    fast_controller.act(
        make_observation(
            velocity=(0.12, 0.0, 0.0),
        )
    )

    fast_lead = (
        fast_controller.last_diagnostics[
            "lead_distance"
        ]
    )

    assert fast_lead < slow_lead


def test_reset_clears_integral_and_phase():
    controller = ConventionalFeedbackController()
    controller.phase = ControllerPhase.PUSH

    controller.act(make_observation())

    assert controller.integral_error != 0.0

    controller.reset()

    assert controller.integral_error == 0.0
    assert (
        controller.phase
        == ControllerPhase.APPROACH
    )


def test_zero_goal_vector_is_safe():
    controller = ConventionalFeedbackController()
    controller.phase = ControllerPhase.PUSH

    action = controller.act(
        make_observation(
            goal_minus_plate=(0.0, 0.0, 0.0),
        )
    )

    assert np.isfinite(action).all()


def test_invalid_observation_is_rejected():
    controller = ConventionalFeedbackController()

    with pytest.raises(ValueError):
        controller.act(
            np.zeros(8, dtype=np.float32)
        )