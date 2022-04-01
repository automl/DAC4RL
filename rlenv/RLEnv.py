from locale import normalize
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces

from stable_baselines3.common.evaluation import evaluate_policy

from dac4automlcomp.dac_env import DACEnv
from carl.envs import *

from functools import partial
import os
import stable_baselines3
import json

import gym
import sys
import inspect

import time

from typing import Dict, Union, Optional, Tuple

from dac4automlcomp.generator import Generator

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(os.getcwd())

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG, PPO, SAC

from carl.envs import *
from carl.envs.carl_env import CARLEnv
from carl.utils.json_utils import lazy_json_dump

import pdb

import carl.training.trial_logger

importlib.reload(carl.training.trial_logger)

from carl.context.sampling import sample_contexts

from rlenv.generators import DefaultRLGenerator, RLInstance

# TODO Check the sampling frequency for contexts
class RLEnv(DACEnv[RLInstance], instance_type=RLInstance):
    def __init__(
        self,
        generator: Generator[RLInstance] = DefaultRLGenerator(),
        device: str = "cpu",
        seed=123456,
        total_timesteps = 1e5,
        n_epochs = 10
    ):
        """
        RL Env that wraps the CARL environment and specifically allows for dynamnically setting hyperparameters
        in a DAC fashion.

        Args:
            generator: Generator object that generates the hyperparameter space
            device: Device to use for the agent
            seed: Seed for the environment
            total_timesteps: Total number of timesteps to train for
            n_epochs: Number of instances to train for
        """

        super().__init__(generator)
        self.device = device

        self.total_timesteps = total_timesteps
        self.n_epochs = n_epochs

        self.ref_seed = self.seed(seed)[0]

        self.allowed_models = ["PPO", "DDPG", "SAC"]

        self.env_multipliers = {
            "CARLPendulumEnv": 0.6,
            "CARLAcrobotEnv" : 0.3,
            "CARLMountainCarContinuousEnv": 0.3 ,
            "CARLLunarLanderEnv" : 1,
            "CARLCartPoleEnv" : 0.3
        }

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise ValueError(
                "Observation space changes for every instance. "
                "It is set after every reset. "
                "Use a provided wrapper or handle it manually. "
                "If batch size is fixed for every instance, "
                "observation space will stay fixed."
            )
        return self._observation_space

    @DACEnv._to_instance.register
    def _(self, instance: RLInstance):
        return instance

    def step(self, action):
        """
        Take a step by applying a set of hyperparameters to the model,
        training that model for the per_instance_steps and then evaluating
        its metrics, which are returned as a state
        """
        print(f"Stepping with action : {action}")

        # create the model based on the action
        self.create_model(action)

        # Train for specified timesteps per epoch
        self.model.learn(total_timesteps=self.per_epoch_steps)

        # Get episode metrics
        episode_rewards = self.env.envs[0].get_episode_rewards()
        episode_lengths = self.env.envs[0].get_episode_lengths()

        # Evaluate Policy for 100 episodes
        mean_reward, std_reward = evaluate_policy(
            model=self.model,
            env=self.eval_env,
            n_eval_episodes=100,
            deterministic=True,
            render=False,
        )

        # Update the trained model in the model_dict
        self.model_dict[self.algorithm] = self.model

        self.epoch_counter += 1

        if self.epoch_counter == self.n_epochs:
            done = True
        else:
            done = False
            

        print(f"Done : {done}")
        print(f"epoch counter : {self.epoch_counter}")

        state = {
            "step": self.epoch_counter,
            "std_reward": std_reward,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        return state, mean_reward, done, {}

    def create_model(self, action):
        """
        Create a model based on the specified algorithm and hyperparameters

        Args:
            algo: Algorithm name

        """
        algo = action["algorithm"]
        action.pop("algorithm")
        hyperparams = action

        if algo not in self.allowed_models:
            raise ValueError(f"Algorithm {algo} not allowed")

        self.algorithm = algo
        self.agent = getattr(stable_baselines3, self.algorithm)

        if self.model_dict[algo] is not None:
            # Load a checkpointed model if it exists
            self.model = self.model_dict[algo]
            for key in hyperparams:
                setattr(self.model, key, hyperparams[key])

            print(f"Loaded model from {self.model_dict[algo]}")
        else:
            # Create a new model otherwise
            self.model = self.agent(
                env=self.env,
                policy="MlpPolicy",
                verbose=1,
                seed=self.ref_seed,
                **hyperparams,
            )

            print(f"Created model for {algo}")

        # Create an eval environment for this agent
        eval_env = make_vec_env(self.EnvCls, n_envs=1, vec_env_cls=DummyVecEnv)

        self.eval_env = self.agent._wrap_env(eval_env)

        eval_env.seed(self.ref_seed)  # env is seeded in agent

        return self.model

    def reset(
        self,
        algorithm: Optional[str] = None,
        instance: Optional[Union[RLInstance, int]] = None,
    ):
        """
        Reset the Instance

        Args:
            instance:   The instance to reset, which already includes environment and
                        Context distribution for this instance

        """

        super().reset(instance)
        
        assert isinstance(self.current_instance, RLInstance)


        # Sample environment, context_features and context_std from the instance
        (self.env_type, context_features, context_std) = self.current_instance

        # The total time for which a schedule is allowed to train on 
        # a sampled environment is divided into epochs of training. Thus, 
        # the hyperparameters are reset after each epoch
        
        # Set the total number of timesteps based on the environment
        total_timesteps = self.env_multipliers[self.env_type] * self.total_timesteps
        self.per_epoch_steps = int(total_timesteps / self.n_epochs)

        print(f"Selected Environment is {self.env_type}")
        print(f"Total timesteps : {total_timesteps}")

        # Sample 1 context based on the instance
        self.contexts = sample_contexts(
            env_name=self.env_type,
            context_feature_args=context_features,
            num_contexts=1,
            default_sample_std_percentage=context_std,
        )

        # Create training environments
        self.env_kwargs = dict(
            contexts=self.contexts,
            hide_context=False,
            state_context_features="changing_context_features",
        )

        self.EnvCls = partial(getattr(carl.envs, self.env_type), **self.env_kwargs)

        self.env = make_vec_env(self.EnvCls, n_envs=1, vec_env_cls=DummyVecEnv)
        self.env.seed(self.ref_seed)

        # Counter to track epochs
        self.epoch_counter = 0

        # Create a dictionary of allowed models
        self.model_dict = {}
        for key in self.allowed_models:
            self.model_dict[key] = None

        ret = {
            "env": self.env_type,
            "context_features": context_features,
            "context_values" : self.contexts,
            "context_std": context_std,
        }

        return ret

    def seed(self, seed=None):
        """
        Standardize seeds
        """
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return super().seed(seed)


if __name__ == "__main__":
    env = gym.make("dac4carl-v0")

    done = False
    algo_schedule = [
        "SAC"
    ]  # Can even include mode algorithms, but it might depend on the environment

    i = 0

    # reset the instance to get all the metrics related to it
    state = env.reset()

    # Some parameters can only be set at the start,
    # since the model makes them into a schedule. Thus,
    # they need to be removed after the first epoch
    flag = True

    algo_action = {
        "buffer_size": 50000,
        "batch_size": 512,
        "ent_coef": 0.1,
        "train_freq": 32,
        "gradient_steps": 32,
        "tau": 0.01,
        "gamma": 0.999,
        "policy_kwargs": dict(log_std_init=-3.67, net_arch=[64, 64]),
        "use_sde": True,
    }

    reward_history = []
    while not done:

        # Add the algorithms to the hyperparams to
        # be used in the next epoch
        algo_action["algorithm"] = algo_schedule[i]

        time_start = time.time()

        # Apply the action
        state, reward, done, _ = env.step(algo_action)

        # Remove the initial parameters
        if flag:
            algo_action.pop("train_freq")
            flag = False

        # Specifically for algo schedules
        i = i + 1
        if i == len(algo_schedule):
            i = 0

        # track rewards
        reward_history.append(reward)

        print("--- %s seconds ---" % (time.time() - time_start))

    print(reward_history)
