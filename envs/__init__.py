from gymnasium.envs.registration import register, registry

from .pushing_env import LiberoPushingEnv
from .pushing_env_v1 import LiberoPushingEnvV1


V0_ENV_ID = "ContactAwarePushing-LIBERO-v0"
V1_ENV_ID = "ContactAwarePushing-LIBERO-v1"

# 保留旧名字，避免已有代码失效。
ENV_ID = V0_ENV_ID


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
    "LiberoPushingEnv",
    "LiberoPushingEnvV1",
]