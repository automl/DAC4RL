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

from dac4automlcomp.generator import Generator, GeneratorIterator, InstanceType

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

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
                                        UniformFloatHyperparameter, Hyperparameter


# TODO: 
# - run this whole thing with the DAC instance


# NOTE: Design choices to do
# Context distributions between instances and also at test time
# Model Saving
# Training step intervals
# Fixed instance set vs generators 
# Final state and action spaces

#TODO: if we want to change the center of the context distribution, 
# we need to change the sampling method in CARL
RLInstance = namedtuple(
    "RLInstance",
    [
        "env_type",
        "context_features",
        "context_dist_std",
    ],
)

@dataclass
class DefaultRLGenerator(Generator[RLInstance]):
    env_type: InitVar[Hyperparameter] = CategoricalHyperparameter("env_type", choices=[CARLPendulumEnv, CARLAcrobotEnv, CARLMountainCarContinuousEnv, CARLLunarLanderEnv])
    context_features_pendulum: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features", choices=[]
    )
    context_features_acrobot: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features", choices=[]
    )
    context_features_mountaincar: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features", choices=[]
    )
    context_features_lunarlander: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features", choices=[]
    )
    context_dist_std: InitVar[Hyperparameter] = UniformFloatHyperparameter(
        "context_dist_std", 0.01, 0.99, log=True, default_value=0.1
    )

    def __post_init__(self, *args):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameters(args)

    def random_instance(self, rng):
        default_rng_state = torch.get_rng_state()
        seed = rng.randint(1, 4294967295, dtype=np.int64)
        self.cs.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        config = self.cs.sample_configuration()
        if config.env_type == CARLPendulumEnv:
            features = config.context_features_pendulum
        elif config.env_type == CARLMountainCarContinuousEnv:
            features = config.context_features_mountaincar
        elif config.env_type == CARLAcrobotEnv:
            features = config.context_features_acrobot
        else:
            features = config.context_features_lunarlander
        torch.set_rng_state(default_rng_state)
        return RLInstance(config.env_type, features, config.context_dist_std)


class RLEnv(DACEnv[RLInstance]):
    def __init__(
        self,
        generator: Generator[RLInstance],
        outdir,
        parser,
        args,
        env,

        device: str = "cpu",
        agent= 'PPO',
        seed = 123456,
        eval_freq = 500, 
        total_timesteps = 1e6,
        n_intervals = 5,    # Should be really low for evaluations
                
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
        self.seed = seed
        self.eval_freq = eval_freq

        self.per_interval_steps = int(total_timesteps/n_intervals) 

        # TODO Set-up CarlEnv in this argument
    
        # set up logger TODO check if required for tracking reward history
        self.logger = TrialLogger(
            outdir,
            parser=parser,
            trial_setup_args=args,
            add_context_feature_names_to_logdir=False,
            init_sb3_tensorboard=False,  # set to False if using SubprocVecEnv
        )
    
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
        Take a step by applying a set of hyperparameters to the model,
        training that model for the per_instance_steps and then evaluating 
        its metric, which are returned
        """

        model_path = os.path.join(
                            self.logger.logdir, 
                            f"model_instance_{self.instance_counter}.zip"
                        )

        # Generate hyperparams
        hyperparams = self._set_hps(action)
        
        
        # Load weights if a model has been saved 
        # -- The case where we are not at the starting instance
        if os.path.exists(model_path):
            self.model.load(os.path.join(self.logger.logdir, "model.zip"))
        else:
            # Create a new model
            # TODO agente class should be iun hyperparams, so just ensure 
            # if this part is not needed
            # TODO Remove algorithm key form the hyperparams 
            self.model = self.agent_cls(
                                env=self.env, 
                                verbose=1, 
                                seed=self.seed, 
                                **hyperparams
                        )  


        # TODO Check if this is necessary, if not for training performance
        self.model.set_logger(
            self.logger.stable_baselines_logger
        )

        # Train for specified timesteps per interval
        self.model.learn(total_timesteps=self.per_interval_steps)


        # Evaluate Policy for 100 episodes
        mean_reward, std_reward = evaluate_policy(
                                        model = self.model, 
                                        env = self.eval_env, 
                                        n_eval_episodes=100,
                                        deterministic=True,
                                        render=False,
                                    )

        # Save the model used in the instance
        # TODO Change if we have too many intervals -- save every certain steps
        self.model.save(model_path)


        if self.interval_counter == self.n_intervals:
            done = True
        else:
            done = False
            self.interval_counter += 1

        # TODO: Add other configurable metrics 
        # -- What else defines a state?
        # -- Return the whole DAC instance 
        # -- Training Reward history : TODO check how to get it easily from the logger
        state = {
            "step": self.instance_counter,
            "std_reward" : std_reward,
            'training_reward_histoy': None,
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
                    - Algorithm
                    - Learning Rate
                    - Discount Factor
                    - Tau
                    - Action Noise
                    - GAE Lambda
                    - Replay Buffer Size
                    - Replay buffer Class
                    - Entropy coefficient
                    - Value Function Coefficients
                    - Clip Coefficient
        """
        action = {
            "algorithm": "PPO",
            "lr": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "clip_range": 0.2,
        }

        hyperparams = action
        hyperparams["policy"] = "MlpPolicy"
        
        return hyperparams
       
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
        super().reset(instance)
        self.interval_counter = 0
        
        assert isinstance(self.current_instance, RLInstance)
        
        (self.env_type, context_features, context_std) = self.current_instance

        # Sample contexts
        self.contexts = sample_contexts(
                            env_name=self.env_type, 
                            context_feature_args= context_features, 
                            num_contexts=self.num_contexts,
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
            env_name=self.env,
            n_envs=1,
            env_kwargs=self.env_kwargs,
            wrapper_class=None,
            vec_env_cls=DummyVecEnv,
            return_eval_env=True,
            normalize_kwargs=None,
            agent_cls=self.agent_cls,
            eval_seed=self.seed,
        )
 

    def seed(self, seed=None):
        """
        Standardize seeds
        """
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return super().seed(seed)



