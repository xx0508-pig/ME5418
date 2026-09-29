from gymnasium.envs.registration import register, registry

from .pushing_env import LiberoPushingEnv
from .pushing_env_v1 import LiberoPushingEnvV1


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
    "make_pushing_env",
    "LiberoPushingEnv",
    "LiberoPushingEnvV1",
]
