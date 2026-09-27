from gymnasium.envs.registration import register, registry
from .pushing_env import LiberoPushingEnv

ENV_ID = "ContactAwarePushing-LIBERO-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="envs.pushing_env:LiberoPushingEnv")

__all__ = ["ENV_ID", "LiberoPushingEnv"]
