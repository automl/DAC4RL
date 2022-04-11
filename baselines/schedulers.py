import dataclasses
import json
from typing import List, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from dac4automlcomp.policy import DACPolicy, DeterministicPolicy


class Serializable:
    """
    Subclass providing a generic way to serialize a dataclass DACPolicy object as a json
    """

    def save(self, path):
        file_path = path / f"{self.__class__.__name__}.json"
        with file_path.open(mode="w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, path):
        file_path = path / f"{cls.__name__}.json"
        with file_path.open(mode="r") as f:
            return cls(**json.load(f))


class Configurable:
    """
    Subclass providing a generic way to specify a DACPolicy's configuration space
    """

    @staticmethod
    def config_space() -> ConfigurationSpace:
        """Return a configuration space object"""
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg):
        """Return an instance of the class corresponding to the given configuration"""
        return cls(**cfg)


@dataclasses.dataclass
class ConstantPolicy(Configurable, Serializable, DeterministicPolicy, DACPolicy):
    algorithm: str
    learning_rate: float
    gamma: float
    gae_lambda: float
    vf_coef: float
    ent_coef: float
    clip_range: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    train_freq: int
    gradient_steps: int

    def act(self, _):
        if self.algorithm == "PPO":
            action = {
                "algorithm": self.algorithm,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "vf_coef": self.vf_coef,
                "ent_coef": self.ent_coef,
                "clip_range": self.clip_range,
            }
        else:
            action = {
                "algorithm": self.algorithm,
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "gamma": self.gamma,
                "train_freq": self.train_freq,
                "gradient_steps": self.gradient_steps,
            }
        return action

    def reset(self, instance):
        pass

    @staticmethod
    def config_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter(
                "learning_rate", lower=0.000001, upper=10, log=True
            )
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("gamma", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("gae_lambda", lower=0.000001, upper=0.99)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("vf_coef", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("ent_coef", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("clip_range", lower=0.0, upper=1.0)
        )
        cs.add_hyperparameter(UniformFloatHyperparameter("tau", lower=0.0, upper=1.0))
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("buffer_size", lower=1000, upper=1e8)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("learning_starts", lower=1, upper=1e4)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("batch_size", lower=8, upper=1024)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("train_freq", lower=1, upper=1e4)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("gradient_steps", lower=-1, upper=1e3)
        )
        cs.add_hyperparameter(
            CategoricalHyperparameter("algorithm", choices=["PPO", "SAC", "DDPG"])
        )
        return cs


@dataclasses.dataclass
class SchedulePolicy(Configurable, Serializable, DeterministicPolicy, DACPolicy):
    algorithm: str
    learning_rates: List[float]
    gammas: List[float]
    gae_lambdas: List[float]
    vf_coefs: List[float]
    ent_coefs: List[float]
    clip_ranges: List[float]
    buffer_sizes: List[int]
    learning_starts: List[int]
    batch_sizes: List[int]
    taus: List[float]
    train_freqs: List[int]
    gradient_steps: List[int]

    def act(self, state):
        if self.t < len(self.learning_rates):
            self.t += 1

        if self.algorithm == "PPO":
            action = {
                "algorithm": self.algorithm,
                "learning_rate": self.learning_rates[min(self.t, len(self.learning_rates)-1)],
                "gamma": self.gammas[min(self.t, len(self.gammas)-1)],
                "gae_lambda": self.gae_lambdas[min(self.t, len(self.gae_lambdas)-1)],
                "vf_coef": self.vf_coefs[min(self.t, len(vf_coefs)-1)],
                "ent_coef": self.ent_coefs[min(self.t, len(ent_coefs)-1)],
                "clip_range": self.clip_ranges[min(self.t, len(self.clip_ranges)-1)],
            }
        else:
            action = {
                "algorithm": self.algorithm,
                "learning_rate": self.learning_rates[min(self.t, len(self.learning_rates)-1)],
                "buffer_size": self.buffer_sizes[min(self.t, len(self.buffer_sizes)-1)],
                "learning_starts": self.learning_starts[min(self.t, len(self.learning_starts)-1)],
                "batch_size": self.batch_sizes[min(self.t, len(self.batch_sizes)-1)],
                "tau": self.taus[min(self.t, len(self.taus)-1)],
                "gamma": self.gammas[min(self.t, len(self.gammas)-1)],
                "train_freq": self.train_freqs[min(self.t, len(self.train_freqs)-1)],
                "gradient_steps": self.gradient_steps[min(self.t, len(self.gradient_steps)-1)],
            }
        return action

    def reset(self, instance):
        self.t = 0

    @staticmethod
    def config_space():
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            UniformFloatHyperparameter(
                "learning_rate", lower=0.000001, upper=10, log=True
            )
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("gamma", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("gae_lambda", lower=0.000001, upper=0.99)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("vf_coef", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("ent_coef", lower=0.000001, upper=1.0)
        )
        cs.add_hyperparameter(
            UniformFloatHyperparameter("clip_range", lower=0.0, upper=1.0)
        )
        cs.add_hyperparameter(UniformFloatHyperparameter("tau", lower=0.0, upper=1.0))
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("buffer_size", lower=1000, upper=1e8)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("learning_starts", lower=1, upper=1e4)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("batch_size", lower=8, upper=1024)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("train_freq", lower=1, upper=1e4)
        )
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("gradient_steps", lower=-1, upper=1e3)
        )
        cs.add_hyperparameter(
            CategoricalHyperparameter("algorithm", choices=["PPO", "SAC", "DDPG"])
        )
        return cs
