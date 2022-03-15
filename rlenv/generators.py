from collections import namedtuple
from dataclasses import dataclass, InitVar

import pdb

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
                                        UniformFloatHyperparameter, Hyperparameter

from ConfigSpace import ConfigurationSpace
from dac4automlcomp.generator import Generator

from carl.envs import *

import torch 
import numpy as np
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
    
    def __init__(self):

        super().__init__()

        self.env_type: InitVar[Hyperparameter] = CategoricalHyperparameter(
                                                "env_type", 
                                                choices=[
                                                    'CARLPendulumEnv', 
                                                    'CARLAcrobotEnv', 
                                                    'CARLMountainCarContinuousEnv', 
                                                    'CARLLunarLanderEnv',
                                                    'CARLCartPoleEnv'
                                                ]
                                            )
        self.context_features_pendulum: InitVar[Hyperparameter] = CategoricalHyperparameter(
            "context_features_pendulum", choices=["max_speed", "dt", "g", "m", "l"]
        )
        self.context_features_acrobot: InitVar[Hyperparameter] = CategoricalHyperparameter(
            "context_features_acrobot", choices=[
                "link_length_1", "link_length_2", "link_mass_1", "link_mass_2",
                "link_com_1", "link_com_2", "link_moi", "max_velocity_1",
                "max_velocity_2", "torque_noise_max"
            ]
        )
        self.context_features_mcc: InitVar[Hyperparameter] = CategoricalHyperparameter(
            "context_features_mcc", choices=[
                "min_position", "max_position", "max_speed", "goal_position", "goal_velocity",
                "power", "min_position_start", "max_position_start", "min_velocity_start",
                "max_velocity_start",
            ]
        )
        self.context_features_lunarlander: InitVar[Hyperparameter] = CategoricalHyperparameter(
            "context_features_lunarlander", choices=[
                "FPS", "SCALE", "MAIN_ENGINE_POWER", "SIDE_ENGINE_POWER",
                "INITIAL_RANDOM", "GRAVITY_X", "GRAVITY_Y", "LEG_AWAY",
                "LEG_DOWN", "LEG_W", "LEG_H", "LEG_SPRING_TORQUE", 
                "SIDE_ENGINE_HEIGHT", "SIDE_ENGINE_AWAY", "VIEWPORT_W",
                "VIEWPORT_H"
            ]
        )

        self.context_features_cartpole: InitVar[Hyperparameter] = CategoricalHyperparameter(
            "context_features_cartpole", choices=[
                "gravity", "masscart", "masspole", "pole_length", 
                "force_magnifier", "update_interval"
            ]
        )

        self.context_dist_std: InitVar[Hyperparameter] = UniformFloatHyperparameter(
            "context_dist_std", 0.01, 0.99, log=True, default_value=0.1
        )

        self.max_context: int = 2

        self.env_space = ConfigurationSpace()
        self.env_space.add_hyperparameter(self.env_type)
        self.env_space.add_hyperparameter(self.context_dist_std)
        
        self.pendulum_context = ConfigurationSpace()
        self.pendulum_context.add_hyperparameter(self.context_features_pendulum)
        
        self.acrobot_context = ConfigurationSpace()
        self.acrobot_context.add_hyperparameter(self.context_features_acrobot)

        self.mcc_context = ConfigurationSpace()
        self.mcc_context.add_hyperparameter(self.context_features_mcc)

        self.cartpole_context = ConfigurationSpace()
        self.cartpole_context.add_hyperparameter(self.context_features_cartpole)
        
        self.lunarlander_context = ConfigurationSpace()
        self.lunarlander_context.add_hyperparameter(self.context_features_lunarlander)


    def random_instance(self, rng):
        default_rng_state = torch.get_rng_state()
        seed = rng.randint(1, 4294967295, dtype=np.int64)
        
        # Seed all he config spaces
        self.env_space.seed(seed)
        self.pendulum_context.seed(seed)
        self.acrobot_context.seed(seed)
        self.mcc_context.seed(seed)
        self.lunarlander_context.seed(seed)
        self.cartpole_context.seed(seed)

        # Seed the torch backend
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Sample an environment
        env_config = self.env_space.sample_configuration()

        # Sample a context vector based on the environment
        features = []
        if env_config['env_type'] == 'CARLPendulumEnv':
            for _ in range(self.max_context):
                context = self.pendulum_context.sample_configuration()
                features.append(context['context_features_pendulum'])

        elif env_config['env_type'] == 'CARLMountainCarContinuousEnv':
            for _ in range(self.max_context):
                context = self.mcc_context.sample_configuration()
                features.append(context['context_features_mcc'])
        elif env_config['env_type'] == 'CARLAcrobotEnv':
            for _ in range(self.max_context):
                context = self.acrobot_context.sample_configuration()
                features.append(context['context_features_acrobot'])
        elif env_config['env_type'] == 'CARLLunarLanderEnv':
            for _ in range(self.max_context):
                context = self.lunarlander_context.sample_configuration()
                features.append(context['context_features_lunarlander'])
        elif env_config['env_type'] == 'CARLCartPoleEnv':
            for _ in range(self.max_context):
                context = self.cartpole_context.sample_configuration()
                features.append(context['context_features_cartpole'])
        else:
            raise ValueError(f"Unknown env_type {env_config['env_type']}")
        torch.set_rng_state(default_rng_state)

        return RLInstance(
                    env_config['env_type'], 
                    features, 
                    env_config['context_dist_std']
                )
