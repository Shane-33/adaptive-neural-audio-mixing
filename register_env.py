# register_env.py

from gymnasium.envs.registration import register

# Register the custom adaptive mixing environment
register(
    id="MixingEnv-v0",
    entry_point="mixing_env:MixingEnv",
)
