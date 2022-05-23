from pathlib import Path
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
from baselines.schedulers import Configurable, Serializable


@dataclasses.dataclass
class MultiSchedulePolicy(Configurable, Serializable, DeterministicPolicy, DACPolicy):
    algorithms: List[str]
    learning_rates: List[List[float]]
    gammas: List[List[float]]
    gae_lambdas: List[List[float]]
    vf_coefs: List[List[float]]
    ent_coefs: List[List[float]]
    clip_ranges: List[List[float]]
    buffer_sizes: List[List[int]]
    learning_starts: List[List[int]]
    batch_sizes: List[List[int]]
    taus: List[List[float]]
    train_freqs: List[List[int]]
    gradient_steps: List[List[int]]

    def act(self, state):
        algorithm = self.algorithms[self.instance]
        learning_rates = self.learning_rates[self.instance]
        gammas = self.gammas[self.instance]
        gae_lambdas = self.gae_lambdas[self.instance]
        vf_coefs = self.vf_coefs[self.instance]
        ent_coefs = self.ent_coefs[self.instance]
        clip_ranges = self.clip_ranges[self.instance]
        buffer_sizes = self.buffer_sizes[self.instance]
        learning_starts = self.learning_starts[self.instance]
        batch_sizes = self.batch_sizes[self.instance]
        taus = self.taus[self.instance]
        train_freqs = self.train_freqs[self.instance]
        gradient_steps = self.gradient_steps[self.instance]

        if self.t < len(learning_rates):
            self.t += 1

        if self.algorithm == "PPO":
            action = {
                "algorithm": algorithm,
                "learning_rate": learning_rates[min(self.t, len(learning_rates)-1)],
                "gamma": gammas[min(self.t, len(gammas)-1)],
                "gae_lambda": gae_lambdas[min(self.t, len(gae_lambdas)-1)],
                "vf_coef": vf_coefs[min(self.t, len(vf_coefs)-1)],
                "ent_coef": ent_coefs[min(self.t, len(ent_coefs)-1)],
                "clip_range": clip_ranges[min(self.t, len(clip_ranges)-1)],
            }
            if self.t > 0:
                del action["clip_range"]
        else:
            action = {
                "algorithm": algorithm,
                "learning_rate": learning_rates[min(self.t, len(learning_rates)-1)],
                "buffer_size": buffer_sizes[min(self.t, len(buffer_sizes)-1)],
                "learning_starts": learning_starts[min(self.t, len(learning_starts)-1)],
                "batch_size": batch_sizes[min(self.t, len(batch_sizes)-1)],
                "tau": taus[min(self.t, len(taus)-1)],
                "gamma": gammas[min(self.t, len(gammas)-1)],
                "train_freq": train_freqs[min(self.t, len(train_freqs)-1)],
                "gradient_steps": gradient_steps[min(self.t, len(gradient_steps)-1)],
            }
            if self.t > 0:
                del action["train_freq"]

        return action

    def reset(self, instance):
        #TODO: get instance id
        self.instance = 0
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


def load_solution(
    policy_cls=SchedulePolicy, path=Path(".")
) -> DACPolicy:
    """
    Load Solution.
    Serves as an entry point for the competition evaluation.
    By default (the submission) it loads a saved SMAC optimized configuration for the ConstantLRPolicy.
    Args:
        policy_cls: The DACPolicy class object to load
        path: Path pointing to the location the DACPolicy is stored
    Returns
    -------
    DACPolicy
    """
    path = Path(path, 'logs', 'pb2_seed0', 'saved_configs')
    return policy_cls.load(path)
