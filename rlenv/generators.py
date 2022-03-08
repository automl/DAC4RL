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
    env_type: InitVar[Hyperparameter] = CategoricalHyperparameter(
                                            "env_type", 
                                            choices=[
                                                'CARLPendulumEnv', 
                                                'CARLAcrobotEnv', 
                                                'CARLMountainCarContinuousEnv', 
                                                'CARLLunarLanderEnv'
                                            ]
                                        )
    context_features_pendulum: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features_pendulum", choices=["max_speed", "dt", "g", "m", "l"]
    )
    context_features_acrobot: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features_acrobot", choices=[
            "link_length_1", "link_length_2", "link_mass_1", "link_mass_2",
            "link_com_1", "link_com_2", "link_moi", "max_velocity_1",
            "max_velocity_2", "torque_noise_max"
        ]
    )
    context_features_mountaincar: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features_mountaincar", choices=[
            "min_position", "max_position", "max_speed",
            "goal_position", "goal_velocity", "force",
            "gravity", "start_position", "start_position_std",
            "start_velocity", "start_velocity_std"
        ]
    )
    # context_features_lunarlander: InitVar[Hyperparameter] = CategoricalHyperparameter(
    #     "context_features_lunarlander", choices=[
    #         "FPS", "SCALE", "MAIN_ENGINE_POWER", "SIDE_ENGINE_POWER",
    #         "INITIAL_RANDOM", "GRAVITY_X", "GRAVITY_Y", "LEG_AWAY",
    #         "LEG_DOWN", "LEG_W", "LEG_H", "LEG_SPRING_TORQUE", 
    #         "SIDE_ENGINE_HEIGHT", "SIDE_ENGINE_AWAY", "VIEWPORT_W",
    #         "VIEWPORT_H"
    #     ]
    # )
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

        # Sample a configuration
        config = self.cs.sample_configuration()

        if config['env_type'] == 'CARLPendulumEnv':
            features = config['context_features_pendulum']
        elif config['env_type'] == 'CARLMountainCarContinuousEnv':
            features = config['context_features_mountaincar']
        elif config['env_type'] == 'CARLAcrobotEnv':
            features = config['context_features_acrobot']
        # elif config['env_type'] == 'CARLLunarLanderEnv':
        #     features = config['context_features_lunarlander']
        else:
            raise ValueError(f"Unknown env_type {config['env_type']}")
        torch.set_rng_state(default_rng_state)


        return RLInstance(
                    config['env_type'], 
                    features, 
                    config['context_dist_std']
                )

