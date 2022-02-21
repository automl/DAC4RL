from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces

from dac4automlcomp.dac_env import DACEnv
from carl.envs import *

from functools import partial
import os
import stable_baselines3
from xvfbwrapper import Xvfb
import configargparse
import yaml
import json

import gym
import sys
import inspect

from typing import Dict, Union, Optional, Type, Callable, Tuple

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(os.getcwd())

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    EveryNTimesteps,
    CheckpointCallback,
)

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


# TODO: 
# - Integrate steps from an RL DAC agent - instances 
# - Control number of contexts on which it is trained per instance step
# - 
 

class RLEnv(DACEnv):
    def __init__(
        self,
        generator,
        outdir,
        parser,
        env,

        device: str = "cpu",
        agent= 'PPO',
        seed = 123456,
        eval_freq = 500
        
        
    ):
        super().__init__(generator)
        self.device = device

        # TODO Set-up CarlEnv in this argument
    
        # set up logger
        self.logger = TrialLogger(
            outdir,
            parser=parser,
            trial_setup_args=args,
            add_context_feature_names_to_logdir=False,
            init_sb3_tensorboard=False,  # set to False if using SubprocVecEnv
        )
    
        # Get the agent class
        try:
            agent_cls = eval(agent)

        except ValueError:
            print(
                f"{agent} is an unknown agent class. Please use a classname from stable baselines 3"
            )


        # Set contexts
        self.contexts = {}



        # Get training and evaluation environments
        env_kwargs = dict(
            contexts=self.contexts,
            logger=None,
            hide_context=False,
            state_context_features="changing_context_features", # Only the features that change are appended to the state
        )


        self.env, self.eval_env = self.get_env(
            env_name=env,
            n_envs=1,
            env_kwargs=env_kwargs,
            wrapper_class=None,
            vec_env_cls=DummyVecEnv,
            return_eval_env=True,
            normalize_kwargs=None,
            agent_cls=agent_cls,
            eval_seed=seed,
        )

        
        # Handle all calbacks
        eval_callback = EvalCallback(
            self.eval_env,
            log_path=logger.logdir,
            eval_freq=1,  # args.eval_freq,
            n_eval_episodes=len(self.contexts),
            deterministic=True,
            render=False,
        )

        callbacks = [eval_callback]
        everynstep_callback = EveryNTimesteps(
            n_steps=eval_freq, 
            callback=eval_callback
        )

        chkp_cb = CheckpointCallback(
            save_freq=eval_freq, 
            save_path=os.path.join(logger.logdir, "models")
        )
        if callbacks is None:
            callbacks = [chkp_cb]
        else:
            callbacks.append(chkp_cb)

    
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
        context_encoder: Optional[th.nn.Module] = None,
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

    @DACEnv.get_instance.register
    def _(self, instance):
        return instance

    def step(self, action: float):
        """
        Take a step by applying a set of hyperparameters to the model.
        """

        model_path = os.path.join(self.logger.logdir, "model.zip")

        # Generate hyperpüarams
        hyperparams = self._set_hps(action)
        
        # Create a new model
        self.model = self.agent_cls(
                            env=self.env, 
                            verbose=1, 
                            seed=self.seed, 
                            **hyperparams
                    )  #

        # Load weights if a model has been saved
        if os.path.exists(model_path):
            self.model.load(os.path.join(self.logger.logdir, "model.zip"))
        

        # TODO Check if this is necessary
        self.model.set_logger(
            self.logger.stable_baselines_logger
        )

        # Train
        self.model.learn(
                total_timesteps=self.steps, 
                callback=self.callbacks
            )

        #TODO: Figure out a way to get rewards

        self.model.save(model_path)




        pass 


    def _set_hps(self, action: Dict):
        
        hyperparams = {}
        env_wrapper = None
        normalize_kwargs = None
        schedule_kwargs = None

        hyperparams["policy"] = "MlpPolicy"



        
        pass
       
    def get_contexts(self, context_file: None):
        """
        Generate Contexts by either sampling them or getting them 
        from  file 
        """


        if not context_file:
            contexts = sample_contexts(
                self.env,
                self.context_feature_args,
                self.num_contexts,
                default_sample_std_percentage=args.default_sample_std_percentage,
            )
        else:
            with open(context_file, "r") as file:
                contexts = json.load(file)
        return contexts


    def reset(self):
        pass 

        # return {
        #     "step": 0,
        #     "loss": loss,
        #     "validation_loss": None,
        #     "crashed": False,
        # }

    def seed(self, seed=None):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return super().seed(seed)
