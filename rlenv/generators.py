from collections import namedtuple
from dataclasses import dataclass, InitVar

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
                                        UniformFloatHyperparameter, Hyperparameter

from ConfigSpace import ConfigurationSpace
from dac4automlcomp.generator import Generator

from carl.envs import *

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
                                                CARLPendulumEnv, 
                                                CARLAcrobotEnv, 
                                                CARLMountainCarContinuousEnv, 
                                                CARLLunarLanderEnv
                                            ]
                                        )
    context_features_pendulum: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features", choices=[1, 2]
    )
    context_features_acrobot: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features", choices=[1,2]
    )
    context_features_mountaincar: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features", choices=[1,2]
    )
    context_features_lunarlander: InitVar[Hyperparameter] = CategoricalHyperparameter(
        "context_features", choices=[1,2]
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
        return RLInstance(
                    config.env_type, 
                    features, 
                    config.context_dist_std
                )

