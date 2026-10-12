from gymnasium.envs.registration import register, registry

from .pushing_env import LiberoPushingEnv
from .pushing_env_v1 import LiberoPushingEnvV1
from .history_wrapper import HistoryStackWrapper
from .observation_statistics import ObservationStatisticsWrapper
from .fixed_normalization import (
    NORMALIZATION_CHOICES,
    FixedObservationNormalizationWrapper,
    apply_observation_normalization,
    normalization_metadata,
)
from .observation_modes import (
    FORMAL_OBSERVATION_MODES,
    OBSERVATION_MODES,
    ObservationModeSpec,
    get_observation_channel_names,
    get_observation_mode_spec,
    stacked_observation_dimension,
)


V0_ENV_ID = "ContactAwarePushing-LIBERO-v0"
V1_ENV_ID = "ContactAwarePushing-LIBERO-v1"

# 保留旧名字，避免旧代码失效。
ENV_ID = V0_ENV_ID

ENV_IDS = {
    "v0": V0_ENV_ID,
    "v1": V1_ENV_ID,
}

ENV_CLASSES = {
    "v0": LiberoPushingEnv,
    "v1": LiberoPushingEnvV1,
}


def make_pushing_env(version, **kwargs):
    try:
        env_class = ENV_CLASSES[version]
    except KeyError as exc:
        supported = ", ".join(sorted(ENV_CLASSES))
        raise ValueError(
            f"Unknown environment version {version!r}. "
            f"Supported versions: {supported}"
        ) from exc

    return env_class(**kwargs)


if V0_ENV_ID not in registry:
    register(
        id=V0_ENV_ID,
        entry_point="envs.pushing_env:LiberoPushingEnv",
    )

if V1_ENV_ID not in registry:
    register(
        id=V1_ENV_ID,
        entry_point=(
            "envs.pushing_env_v1:LiberoPushingEnvV1"
        ),
    )


__all__ = [
    "ENV_ID",
    "V0_ENV_ID",
    "V1_ENV_ID",
    "ENV_IDS",
    "ENV_CLASSES",
    "FORMAL_OBSERVATION_MODES",
    "OBSERVATION_MODES",
    "ObservationModeSpec",
    "HistoryStackWrapper",
    "ObservationStatisticsWrapper",
    "NORMALIZATION_CHOICES",
    "FixedObservationNormalizationWrapper",
    "apply_observation_normalization",
    "normalization_metadata",
    "get_observation_channel_names",
    "get_observation_mode_spec",
    "stacked_observation_dimension",
    "make_pushing_env",
    "LiberoPushingEnv",
    "LiberoPushingEnvV1",
]
