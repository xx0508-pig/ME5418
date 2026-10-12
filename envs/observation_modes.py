"""Shared observation-mode definitions for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationModeSpec:
    include_object_velocity: bool
    include_contact_force: bool
    contact_force_components: int
    dimension: int
    contact_force_axes: tuple[str, ...]
    formal: bool


OBSERVATION_MODES = {
    "pos6": ObservationModeSpec(
        include_object_velocity=False,
        include_contact_force=False,
        contact_force_components=0,
        dimension=6,
        contact_force_axes=(),
        formal=True,
    ),
    "posvel9": ObservationModeSpec(
        include_object_velocity=True,
        include_contact_force=False,
        contact_force_components=0,
        dimension=9,
        contact_force_axes=(),
        formal=True,
    ),
    "posvel_force11": ObservationModeSpec(
        include_object_velocity=True,
        include_contact_force=True,
        contact_force_components=2,
        dimension=11,
        contact_force_axes=("x", "y"),
        formal=True,
    ),
    # Legacy modes remain available so existing models and configs can
    # still be evaluated without changing their observation semantics.
    "posforce9": ObservationModeSpec(
        include_object_velocity=False,
        include_contact_force=True,
        contact_force_components=3,
        dimension=9,
        contact_force_axes=("x", "y", "z"),
        formal=False,
    ),
    "posvelforce12": ObservationModeSpec(
        include_object_velocity=True,
        include_contact_force=True,
        contact_force_components=3,
        dimension=12,
        contact_force_axes=("x", "y", "z"),
        formal=False,
    ),
}

FORMAL_OBSERVATION_MODES = tuple(
    name
    for name, spec in OBSERVATION_MODES.items()
    if spec.formal
)

POSITION_CHANNEL_NAMES = (
    "plate_minus_eef_x",
    "plate_minus_eef_y",
    "plate_minus_eef_z",
    "goal_minus_plate_x",
    "goal_minus_plate_y",
    "goal_minus_plate_z",
)

VELOCITY_CHANNEL_NAMES = (
    "plate_velocity_x",
    "plate_velocity_y",
    "plate_velocity_z",
)

FORCE_CHANNEL_NAMES = (
    "contact_force_x",
    "contact_force_y",
    "contact_force_z",
)


def get_observation_mode_spec(name):
    try:
        return OBSERVATION_MODES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(OBSERVATION_MODES))
        raise ValueError(
            f"Unknown observation mode {name!r}. "
            f"Supported modes: {supported}"
        ) from exc


def get_observation_channel_names(name):
    spec = get_observation_mode_spec(name)
    names = list(POSITION_CHANNEL_NAMES)
    if spec.include_object_velocity:
        names.extend(VELOCITY_CHANNEL_NAMES)
    if spec.include_contact_force:
        names.extend(
            FORCE_CHANNEL_NAMES[: spec.contact_force_components]
        )
    if len(names) != spec.dimension:
        raise RuntimeError(
            f"Observation channel definition for {name!r} has "
            f"{len(names)} names but dimension {spec.dimension}."
        )
    return tuple(names)


def stacked_observation_dimension(
    observation_dimension,
    action_dimension,
    history_length,
):
    if history_length < 1:
        raise ValueError(
            "history_length must be at least one"
        )

    return (
        history_length * observation_dimension
        + (history_length - 1) * action_dimension
    )
