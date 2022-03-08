from gym.envs.registration import register

register(
    id="rl-v0",
    entry_point="rlenv:RLEnv",
)

__all__ = ["generators"]
