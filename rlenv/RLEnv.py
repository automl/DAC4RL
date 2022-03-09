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

from typing import Dict, Union, Optional, Type, Callable, Tuple

from dac4automlcomp.generator import Generator

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(os.getcwd())

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import DDPG, PPO, SAC


from carl.envs import *
from carl.envs.carl_env import CARLEnv
from carl.utils.json_utils import lazy_json_dump

import torch as th
import pdb

import carl.training.trial_logger

importlib.reload(carl.training.trial_logger)
from carl.training.trial_logger import TrialLogger

from carl.context.sampling import sample_contexts
from carl.utils.hyperparameter_processing import preprocess_hyperparams

from collections import namedtuple
from dataclasses import dataclass, InitVar

from rlenv.generators import DefaultRLGenerator, RLInstance


# TODO: 
# - Control the number of contexts sampled per instance
# - Control the distribution of contexts in the generator


# NOTE: Design choices to do
# Training step intervals
# Rewar histories in different states 


class RLEnv(DACEnv[RLInstance], instance_type=RLInstance):
    def __init__(
        self,
        env,
        generator: Generator[RLInstance] = DefaultRLGenerator(),
        # outdir='/tmp/RL_test',
        # parser,
        # args,
        
        device: str = "cpu",
        agent= 'PPO',
        seed = 123456,
        eval_freq = 500, 
        total_timesteps = 1e6,
        n_intervals = 20,    # Should be really low for evaluations
                
    ):
        """
        RL Env that wraps the CARL environment and specifically allows for dynamnically setting hyperparameters 
        in a DAC fashion. 

        Args:
            generator: Generator object that generates the hyperparameter space
            outdir: Path to the output directory
            parser: ArgumentParser object that contains the arguments for the experiment
            env: CARL environment
            device: Device to use for the agent
            agent: Agent to use for training
            seed: Seed for the environment
            eval_freq: Frequency at which to evaluate the agent
            total_timesteps: Total number of timesteps to train for
            n_instances: Number of instances to train for
        """

        super().__init__(generator)
        self.device = device

        self.total_timesteps = total_timesteps
        self.n_intervals = n_intervals
        self.env_type = env
        self.eval_freq = eval_freq
        self.ref_seed = self.seed(seed)[0]

        self.per_interval_steps = int(total_timesteps/n_intervals) 

    
        # set up logger TODO check if required for tracking reward history
        # self.logger = TrialLogger(
        #     outdir,
        #     parser=parser,
        #     trial_setup_args=args,
        #     add_context_feature_names_to_logdir=False,
        #     init_sb3_tensorboard=False,  # set to False if using SubprocVecEnv
        # )
    
        # Get the agent class
        try:
            self.agent_cls = eval(agent)

        except ValueError:
            print(
                f"{agent} is an unknown agent class. Please use a classname from stable baselines 3"
            )

        # Counter to track instances
        self.interval_counter = 0

    
    def get_env(
        self,
        env_name,
        n_envs: int = 1,
        env_kwargs: Optional[Dict] = None,
        wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
        wrapper_kwargs=None,
        normalize_kwargs: Optional[Dict] = None,
        agent_cls: Optional[stable_baselines3.common.base_class.BaseAlgorithm] = None,  # only important for eval env to appropriately wrap
        eval_seed: Optional[int] = None,  # env is seeded in agent
        return_vec_env: bool = True,
        vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
        return_eval_env: bool = False,
    ) -> Union[CARLEnv, Tuple[CARLEnv]]:
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if env_kwargs is None:
            env_kwargs = {}
        EnvCls = partial(getattr(carl.envs, env_name), **env_kwargs)

        make_vec_env_kwargs = dict(
            wrapper_class=wrapper_class,
            vec_env_cls=vec_env_cls,
            wrapper_kwargs=wrapper_kwargs,
        )

        # Wrap, Seed and Normalize Env
        if return_vec_env:
            env = make_vec_env(EnvCls, n_envs=n_envs, **make_vec_env_kwargs)
        else:
            env = EnvCls()
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
        n_eval_envs = 1

        # Eval policy works with more than one eval envs, but the number of contexts/instances must be divisible
        # by the number of eval envs without rest in order to catch all instances.
        if return_eval_env:
            if return_vec_env:
                eval_env = make_vec_env(EnvCls, n_envs=n_eval_envs, **make_vec_env_kwargs)
            else:
                eval_env = EnvCls()
                if wrapper_class is not None:
                    eval_env = wrapper_class(env, **wrapper_kwargs)
            if agent_cls is not None:
                eval_env = agent_cls._wrap_env(eval_env)
            else:
                warnings.warn(
                    "agent_cls is None. Should be provided for eval_env to ensure that the correct wrappers are used."
                )
            if eval_seed is not None:
                eval_env.seed(eval_seed)  # env is seeded in agent

        if normalize_kwargs is not None and normalize_kwargs["normalize"]:
            del normalize_kwargs["normalize"]
            env = VecNormalize(env, **normalize_kwargs)

            if return_eval_env:
                eval_normalize_kwargs = normalize_kwargs.copy()
                eval_normalize_kwargs["norm_reward"] = False
                eval_normalize_kwargs["training"] = False
                eval_env = VecNormalize(eval_env, **eval_normalize_kwargs)

        ret = env
        if return_eval_env:
            ret = (env, eval_env)
        return ret
    
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

    # TODO: 
    # - Algorithms can be changed between intervals, so just 
    # create a new every time
    def step(self, action):
        """
        Take a step by applying a set of hyperparameters to the model,
        training that model for the per_instance_steps and then evaluating 
        its metric, which are returned
        """
        print(f'Stepping with action : {action}')

        # Generate hyperparams
        algo, hyperparams = self._set_hps(action)

        # Create a new model 
        model = eval(algo)(
                            env=self.env, 
                            verbose=1, 
                            seed=self.ref_seed, 
                            **hyperparams
                    )  


        # TODO Check if this is necessary, if not for training performance
        # self.model.set_logger(
        #     self.logger.stable_baselines_logger
        # )

        # Train for specified timesteps per interval
        model.learn(total_timesteps=self.per_interval_steps)


        # Evaluate Policy for 100 episodes
        mean_reward, std_reward = evaluate_policy(
                                        model = model, 
                                        env = self.eval_env, 
                                        n_eval_episodes=100,
                                        deterministic=True,
                                        render=False,
                                    )

        if self.interval_counter == self.n_intervals:
            done = True
        else:
            done = False
            self.interval_counter += 1

        # TODO Check if reward history is required
        state = {
            "step": self.interval_counter,
            "std_reward" : std_reward,
            'Instance': None
        }

        return state, mean_reward, done, {}

    # TODO check if the kwargs make sense for the algorithm
    # Policy Architecture keeps to default
    def _set_hps(self, action: Dict):
        """
        Set the hyperparameters based on the action 

        Args:
            action: Dict of hyperparameters (Exhaustive list for all algorithms)
        """

        hyperparams = action
        hyperparams["policy"] = "MlpPolicy"

        algo  = hyperparams['algorithm']
        hyperparams.pop('algorithm')

        return algo, hyperparams

    def reset(
        self, 
        instance: Optional[Union[RLInstance, int]] = None
    ):
        """
        Reset the Instance

        Args:
            instance:   The instance to reset, which already includes environment and 
                        Context distribution for this instance


        """
        print('Resetting the environment')
        super().reset(instance)
        self.interval_counter = 0
        
        assert isinstance(self.current_instance, RLInstance)
        
        (self.env_type, context_features, context_std) = self.current_instance

        print(f'New Environment is {self.env_type}')

        # Sample contexts based on the instance
        self.contexts = sample_contexts(
                            env_name=self.env_type, 
                            context_feature_args= context_features, 
                            num_contexts=100,
                            default_sample_std_percentage=context_std
                        )


        # Get training and evaluation environments
        self.env_kwargs = dict(
            contexts=self.contexts,
            logger=None,
            hide_context=False,
            state_context_features="changing_context_features", # Only the features that change are appended to the state
        )
        

        self.env, self.eval_env = self.get_env(
            env_name=self.env_type,
            n_envs=1,
            env_kwargs=self.env_kwargs,
            wrapper_class=None,
            vec_env_cls=DummyVecEnv,
            return_eval_env=True,
            normalize_kwargs=None,
            agent_cls=self.agent_cls,
            eval_seed=self.ref_seed,
        )


 

    def seed(self, seed=None):
        """
        Standardize seeds
        """
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return super().seed(seed)


if __name__ == "__main__":
    env = gym.make( "dac4carl-v0", 
                    env=CARLPendulumEnv, 
                    total_timesteps=1e2, 
                    n_intervals=20
                )
    
    done = False
    while not done:
        env.reset()
        ppo_action = {
            "algorithm": "PPO",
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "clip_range": 0.2,
        }
        state, reward, done, _ = env.step(ppo_action)
        

    print(f'I\'ve got the magic stuff')

