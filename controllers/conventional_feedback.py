from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np


PI_NO_RELEASE = "pi_no_release"
HYBRID_PI_SETTLE = "hybrid_pi_settle"
CONTROLLER_VARIANTS = (
    PI_NO_RELEASE,
    HYBRID_PI_SETTLE,
)


class ControllerPhase(str, Enum):
    APPROACH = "approach"
    DESCEND = "descend"
    PUSH = "push"
    SETTLE = "settle"


@dataclass(frozen=True)
class FeedbackControllerConfig:
    # False 保留原始 PI baseline；True 启用人工接触释放阶段。
    enable_settle: bool = False

    # LIBERO 控制频率为 20 Hz。
    control_dt: float = 0.05

    # 接近盘子的几何参数。
    behind_distance: float = 0.05
    hover_height_offset: float = 0.18
    push_height_offset: float = 0.01

    # 各阶段的位置控制参数（SETTLE 可选）。
    approach_gain: float = 6.0
    descend_gain: float = 6.0
    push_position_gain: float = 8.0
    settle_gain: float = 6.0

    approach_max_action: float = 0.70
    descend_max_action: float = 0.50
    push_max_action: float = 0.55
    settle_max_action: float = 0.45
    vertical_action_limit: float = 0.25

    approach_tolerance: float = 0.025
    descend_tolerance: float = 0.015

    # v1 benchmark 成功及减速参数。
    success_radius: float = 0.06
    slowdown_distance: float = 0.14
    max_object_speed: float = 0.06

    # 盘子进入目标区后，机械臂后撤并抬升以释放接触。
    settle_backoff_distance: float = 0.04
    settle_height_offset: float = 0.08

    # 只有盘子明显滑出目标区才重新接近，避免在边界反复切换。
    goal_exit_hysteresis: float = 0.015

    # PI 速度反馈参数。
    velocity_feedforward_gain: float = 0.25
    velocity_kp: float = 0.20
    velocity_ki: float = 0.05
    integral_limit: float = 0.05

    # 相对于盘子中心的推动深度。
    # 负数表示机械臂略微后退。
    lead_min: float = -0.015
    lead_max: float = 0.035

    def __post_init__(self):
        if self.slowdown_distance <= self.success_radius:
            raise ValueError(
                "slowdown_distance must exceed success_radius"
            )

        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")

        if self.integral_limit <= 0:
            raise ValueError("integral_limit must be positive")

        if self.settle_backoff_distance <= 0:
            raise ValueError(
                "settle_backoff_distance must be positive"
            )

        if self.settle_height_offset <= self.push_height_offset:
            raise ValueError(
                "settle_height_offset must exceed "
                "push_height_offset"
            )

        if self.settle_gain <= 0:
            raise ValueError("settle_gain must be positive")

        if self.settle_max_action <= 0:
            raise ValueError(
                "settle_max_action must be positive"
            )

        if self.goal_exit_hysteresis <= 0:
            raise ValueError(
                "goal_exit_hysteresis must be positive"
            )

        if self.lead_min > self.lead_max:
            raise ValueError(
                "lead_min cannot exceed lead_max"
            )

    def to_dict(self):
        return asdict(self)


def make_feedback_controller_config(
    variant,
    **overrides,
):
    """Build an explicit, reproducible baseline variant."""

    if variant not in CONTROLLER_VARIANTS:
        raise ValueError(
            f"Unknown controller variant: {variant!r}. "
            f"Expected one of {CONTROLLER_VARIANTS}."
        )

    return FeedbackControllerConfig(
        enable_settle=(
            variant == HYBRID_PI_SETTLE
        ),
        **overrides,
    )


class ConventionalFeedbackController:
    """
    Geometric PI controller with an optional SETTLE phase.

    Expected observation:
        obs[0:3] = plate_position - eef_position
        obs[3:6] = goal_position - plate_position
        obs[6:9] = plate_velocity

    The controller never receives or reads friction.
    """

    def __init__(
        self,
        config: FeedbackControllerConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else FeedbackControllerConfig()
        )

        self.reset()

    def reset(self):
        """Reset state at the beginning of every episode."""

        self.phase = ControllerPhase.APPROACH
        self.integral_error = 0.0

        # 当盘子已经位于目标中心时，避免方向归一化除零。
        self.last_direction = np.array(
            [1.0, 0.0],
            dtype=np.float64,
        )

        # SETTLE 阶段沿进入目标区时的反方向后撤，避免目标方向
        # 在接近中心时快速改变而造成机械臂来回摆动。
        self.settle_direction = self.last_direction.copy()

        self.last_diagnostics = {}

    @staticmethod
    def _validate_observation(observation):
        observation = np.asarray(
            observation,
            dtype=np.float64,
        )

        if observation.ndim != 1 or observation.size < 9:
            raise ValueError(
                "Expected a flat observation with at least 9 values"
            )

        if not np.isfinite(observation[:9]).all():
            raise ValueError(
                "Observation contains non-finite values"
            )

        return observation

    def _goal_geometry(self, observation):
        goal_vector = observation[3:5]

        goal_distance = float(
            np.linalg.norm(goal_vector)
        )

        if goal_distance > 1e-8:
            direction = goal_vector / goal_distance
            self.last_direction = direction.copy()
        else:
            direction = self.last_direction.copy()

        return direction, goal_distance

    @staticmethod
    def _clip_action(
        action,
        limit,
        vertical_limit=None,
    ):
        action = np.clip(
            action,
            -limit,
            limit,
        )

        if vertical_limit is not None:
            action[2] = np.clip(
                action[2],
                -vertical_limit,
                vertical_limit,
            )

        return action.astype(np.float32)

    def _position_action(
        self,
        plate_minus_eef,
        direction,
        xy_offset,
        z_offset,
        gain,
        limit,
        vertical_limit=None,
    ):
        """
        Construct an EEF position target relative to the plate.

        xy_offset < 0:
            desired EEF position is behind the plate.

        xy_offset > 0:
            desired EEF position is slightly through the plate,
            producing a pushing action.
        """

        target_offset = np.array(
            [
                xy_offset * direction[0],
                xy_offset * direction[1],
                z_offset,
            ],
            dtype=np.float64,
        )

        # plate_minus_eef + target_offset
        # equals desired_eef_position - current_eef_position.
        position_error = (
            plate_minus_eef + target_offset
        )

        action = self._clip_action(
            gain * position_error,
            limit,
            vertical_limit,
        )

        return action, position_error

    def act(self, observation):
        observation = self._validate_observation(
            observation
        )

        plate_minus_eef = observation[:3]
        object_velocity = observation[6:9]

        direction, goal_distance = (
            self._goal_geometry(observation)
        )

        config = self.config

        # ----------------------------------------------------
        # Stage 1: move above a point behind the plate.
        # ----------------------------------------------------

        if self.phase == ControllerPhase.APPROACH:
            action, position_error = (
                self._position_action(
                    plate_minus_eef,
                    direction,
                    xy_offset=-config.behind_distance,
                    z_offset=config.hover_height_offset,
                    gain=config.approach_gain,
                    limit=config.approach_max_action,
                )
            )

            if (
                np.linalg.norm(position_error)
                <= config.approach_tolerance
            ):
                self.phase = ControllerPhase.DESCEND

            self.last_diagnostics = {
                "phase": self.phase.value,
                "goal_distance": goal_distance,
                "position_error": float(
                    np.linalg.norm(position_error)
                ),
            }

            return action

        # ----------------------------------------------------
        # Stage 2: descend behind the plate.
        # ----------------------------------------------------

        if self.phase == ControllerPhase.DESCEND:
            action, position_error = (
                self._position_action(
                    plate_minus_eef,
                    direction,
                    xy_offset=-config.behind_distance,
                    z_offset=config.push_height_offset,
                    gain=config.descend_gain,
                    limit=config.descend_max_action,
                    vertical_limit=(
                        config.vertical_action_limit
                    ),
                )
            )

            if (
                np.linalg.norm(position_error)
                <= config.descend_tolerance
            ):
                self.phase = ControllerPhase.PUSH
                self.integral_error = 0.0

            self.last_diagnostics = {
                "phase": self.phase.value,
                "goal_distance": goal_distance,
                "position_error": float(
                    np.linalg.norm(position_error)
                ),
            }

            return action

        # 盘子首次进入目标区后，停止 PI 推动并释放接触。
        if (
            config.enable_settle
            and self.phase == ControllerPhase.PUSH
            and goal_distance <= config.success_radius
        ):
            self.phase = ControllerPhase.SETTLE
            self.integral_error = 0.0
            self.settle_direction = direction.copy()

        # ----------------------------------------------------
        # Stage 4: back off and lift to let the plate settle.
        # ----------------------------------------------------

        if self.phase == ControllerPhase.SETTLE:
            # 使用回滞边界，防止距离在 0.06 m 附近时逐步抖动。
            if (
                goal_distance
                > config.success_radius
                + config.goal_exit_hysteresis
            ):
                self.phase = ControllerPhase.APPROACH
                self.integral_error = 0.0

                action, position_error = (
                    self._position_action(
                        plate_minus_eef,
                        direction,
                        xy_offset=-config.behind_distance,
                        z_offset=config.hover_height_offset,
                        gain=config.approach_gain,
                        limit=config.approach_max_action,
                    )
                )
            else:
                action, position_error = (
                    self._position_action(
                        plate_minus_eef,
                        self.settle_direction,
                        xy_offset=-config.settle_backoff_distance,
                        z_offset=config.settle_height_offset,
                        gain=config.settle_gain,
                        limit=config.settle_max_action,
                        vertical_limit=(
                            config.vertical_action_limit
                        ),
                    )
                )

            self.last_diagnostics = {
                "phase": self.phase.value,
                "goal_distance": goal_distance,
                "position_error": float(
                    np.linalg.norm(position_error)
                ),
            }

            return action

        # ----------------------------------------------------
        # Stage 3: PI velocity-feedback pushing.
        # ----------------------------------------------------

        slowdown_span = (
            config.slowdown_distance
            - config.success_radius
        )

        speed_fraction = np.clip(
            (
                goal_distance
                - config.success_radius
            )
            / slowdown_span,
            0.0,
            1.0,
        )

        # Far from the target: desired speed is high.
        # Near the target: desired speed gradually approaches 0.
        desired_speed = float(
            config.max_object_speed
            * speed_fraction
        )

        # Only use the component of velocity directed at the goal.
        forward_speed = float(
            np.dot(
                object_velocity[:2],
                direction,
            )
        )

        speed_error = (
            desired_speed - forward_speed
        )

        # Prevent positive integral wind-up after entering
        # the target radius.
        if desired_speed == 0.0:
            self.integral_error = min(
                self.integral_error,
                0.0,
            )

        self.integral_error = float(
            np.clip(
                self.integral_error
                + speed_error
                * config.control_dt,
                -config.integral_limit,
                config.integral_limit,
            )
        )

        # PI output controls how far through the plate
        # the desired EEF position is placed.
        lead_distance = (
            config.velocity_feedforward_gain
            * desired_speed
            + config.velocity_kp
            * speed_error
            + config.velocity_ki
            * self.integral_error
        )

        lead_distance = float(
            np.clip(
                lead_distance,
                config.lead_min,
                config.lead_max,
            )
        )

        action, position_error = (
            self._position_action(
                plate_minus_eef,
                direction,
                xy_offset=lead_distance,
                z_offset=config.push_height_offset,
                gain=config.push_position_gain,
                limit=config.push_max_action,
                vertical_limit=(
                    config.vertical_action_limit
                ),
            )
        )

        self.last_diagnostics = {
            "phase": self.phase.value,
            "goal_distance": goal_distance,
            "desired_speed": desired_speed,
            "forward_speed": forward_speed,
            "speed_error": speed_error,
            "integral_error": self.integral_error,
            "lead_distance": lead_distance,
            "position_error": float(
                np.linalg.norm(position_error)
            ),
        }

        return action
