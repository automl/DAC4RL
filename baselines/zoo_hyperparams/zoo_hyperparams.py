"""
This is an example submission that can give participants a reference 
"""

# from dac4automlcomp.run_experiments import run_experiment

from dac4automlcomp.policy import DACPolicy
from rlenv.generators import DefaultRLGenerator, RLInstance

from carl.envs import *

import gym
import pdb

import time


class ZooHyperparams(DACPolicy):
    """
    A policy which checks the instance and applies fixed parameters for PPO
    to the model. The parameters are based on the ones specified in stable_baselines
    zoo (https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)

    """

    def __init__(self):
        """
        Initialize all the aspects needed for the policy.
        """

        # Set the algorithm that will be used
        self.algorithm = "PPO"

    def _get_zoo_params(self, env: str):
        """
        Return a set of hyperparameters for the given environment.

        Args:
            env: str
                The name of the environment.

        Returns:
            params: Dict
                The hyperparameters for the environment.
        """

        if env == "CARLPendulumEnv":
            params = {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "batch_size": 64,
                "n_steps": 2048,
                "n_epochs": 10,
            }
        elif env == "CARLCartPoleEnv":
            params = {
                "learning_rate": 0.001,  # should be a schedule, but for this example we use a constant
                "gamma": 0.98,
                "gae_lambda": 0.8,
                "ent_coef": 0.0,
                "n_steps": 32,
                "n_epochs": 10,
                "batch_size": 256,
            }
        elif env == "CARLAcrobotEnv":
            params = {
                "gamma": 0.99,
                "gae_lambda": 0.94,
                "ent_coef": 0.0,
                "n_epochs": 4,
            }
        elif env == "CARLMountainCarContinuousEnv":
            params = {
                "learning_rate": 7.77e-05,
                "gamma": 0.9999,
                "gae_lambda": 0.9,
                "vf_coef": 0.19,
                "ent_coef": 0.00429,
                "policy_kwargs": dict(log_std_init=-3.29, ortho_init=False),
                "max_grad_norm": 5,
                "n_epochs": 10,
                "batch_size": 256,
                "use_sde": True,
            }
        elif env == "CARLLunarLanderEnv":
            params = {
                "n_steps": 1024,
                "batch_size": 64,
                "gae_lambda": 0.98,
                "gamma": 0.999,
                "n_epochs": 4,
                "ent_coef": 0.01,
            }

        return params

    def act(self, obs):
        """
        Generate an action in the form of the hyperparameters based on
        the given instance

        Args:
            state: Dict
                The state of the environment.

        Returns:
            action: Dict
        """
        # Get the environment from the state

        # Get the zoo parameters for hte environment
        zoo_params = self._get_zoo_params(self.env)

        # Create the action dictionary
        action = {"algorithm": self.algorithm}
        action = {**action, **zoo_params}

        return action

    def reset(self, instance):
        """Reset a policy's internal state.

        The reset method is there to support 'stateful' policies (e.g., LSTM),
        i.e., whose actions are a function not only of the current
        observations, but of the entire observation history from the
        current episode/execution. It is called at the beginning of the
        target algorithm execution (before the first call to act()) and also provides the policy
        with information about the target problem instance being solved.

        Args:
            instance: The problem instance the target algorithm to be configured is currently solving
        """
        self.env = instance.env_type

    def seed(self, seed):
        """Sets random state of the policy.
        Subclasses should implement this method if their policy is stochastic
        """
        pass

    def save(self, path):
        """Saves the policy to given folder path."""
        pass

    @classmethod
    def load(cls, path):
        """Loads the policy from given folder path."""
        pass


if __name__ == "__main__":

    start_time = time.time()

    policy = ZooHyperparams()
    env = gym.make("dac4carl-v0")
    done = False

    state = env.reset()

    env_type = state["env"]
    policy.env = env_type
    
    print(policy.env)
    pdb.set_trace()

    reward_history = []
    while not done:
        # get the default stat at reset

        init_time = time.time()

        # generate an action
        action = policy.act(env_type)

        # Apply the action t get hte reward, next state and done
        state, reward, done, _ = env.step(action)

        # save the reward
        reward_history.append(reward)
        print("--- %s seconds per instance---" % (time.time() - init_time))

    print(reward_history)

    print("--- %s seconds ---" % (time.time() - start_time))
