from gym.envs.registration import register

register(
    id="dac4carl-v0",
    entry_point="rlenv.RLEnv:RLEnv",
)

__all__ = ["generators"]
