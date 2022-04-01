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


class RLEnv(DACEnv[RLInstance], instance_type=RLInstance):
    def __init__(
        self,
        generator: Generator[RLInstance] = DefaultRLGenerator(),
        device: str = "cpu",
        seed=123456,
        total_timesteps=1e6,
        n_intervals=20,  # Should be really low for evaluations
    ):
        """
        RL Env that wraps the CARL environment and specifically allows for dynamnically setting hyperparameters
        in a DAC fashion.

        Args:
            generator: Generator object that generates the hyperparameter space
            device: Device to use for the agent
            seed: Seed for the environment
            total_timesteps: Total number of timesteps to train for
            n_instances: Number of instances to train for
        """

        super().__init__(generator)
        self.device = device

        self.total_timesteps = total_timesteps
        self.n_intervals = n_intervals
        self.per_interval_steps = int(total_timesteps / n_intervals)

        print(f"Total timesteps : {self.total_timesteps}")
        print(f"Per interval steps : {self.per_interval_steps}")

        self.ref_seed = self.seed(seed)[0]

        self.allowed_models = ["PPO", "DDPG", "SAC"]

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

        # Train for specified timesteps per interval
        self.model.learn(total_timesteps=self.per_interval_steps)

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

        if self.interval_counter == self.n_intervals:
            done = True
        else:
            done = False
            self.interval_counter += 1

        print(f"Done : {done}")
        print(f"Interval counter : {self.interval_counter}")

        state = {
            "step": self.interval_counter,
            "std_reward": std_reward,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        return state, mean_reward, done, {}

    def _set_hps(self, action: Dict):
        """
        Set the hyperparameters based on the action

        Args:
            action: Dict of hyperparameters (Exhaustive list for all algorithms)

        Returns:
            algo: Algorithm name
            hyperparams: Dict of hyperparameters
        """

        hyperparams = action

    def create_model(self, action):
        """
        Create a model based on the specified algorithm

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

        print(f"Selected Environment is {self.env_type}")

        # Sample contexts based on the instance
        self.contexts = sample_contexts(
            env_name=self.env_type,
            context_feature_args=context_features,
            num_contexts=100,
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

        # Counter to track intervals
        self.interval_counter = 0

        # Create a dictionary of allowed models
        self.model_dict = {}
        for key in self.allowed_models:
            self.model_dict[key] = None

        return {
            "Env": self.env_type,
            "Context_Features": context_features,
            "instance": self.current_instance,
        }

    def seed(self, seed=None):
        """
        Standardize seeds
        """
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return super().seed(seed)


if __name__ == "__main__":
    env = gym.make("dac4carl-v0", total_timesteps=1e5, n_intervals=10)

    done = False
    algo_schedule = [
        "SAC"
    ]  # Can even include mode algorithms, but it might depend on the environment

    i = 0

    # reset the instance to get all the metrics related to it
    state = env.reset()

    # Some parameters can only be set at the start,
    # since the model makes them into a schedule. Thus,
    # they need to be removed after the first interval
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
        # be used in the next interval
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

    print(f"I've got the magic stuff")
