import argparse
from pathlib import Path

import gym
from baselines import schedulers

import ray
from ray import tune
from ray.tune.schedulers.pb2 import PB2
from ray.tune.examples.pbt_function import pbt_function


def evaluate_cost(cfg, **kwargs):
    #TODO: pickle won't like this...
    global args
    global train_env
    global done
    if done:
        train_env.reset()
    if round(cfg["algorithm"]) == 0:
        cfg["algorithm"] = "PPO"
    elif round(cfg["algorithm"]) == 1:
        cfg["algorithm"] = "SAC"
    else:
        cfg["algorithm"] = "DDPG"

    if cfg["algorithm"] == "PPO":
        action = {"algorithm": cfg["algorithm"],
                  "learning_rate": cfg["learning_rate"],
                  "gamma": cfg["gamma"],
                  "gae_lambda": cfg["gae_lambda"],
                  "vf_coef": cfg["vf_coef"],
                  "ent_coef": cfg["ent_coef"],
                  "clip_range": cfg["clip_range"]
                  }
    else:
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg["buffer_size"] = int(cfg["buffer_size"])
        cfg["learning_starts"] = int(cfg["learning_starts"])
        cfg["train_freq"] = int(cfg["train_freq"])
        cfg["gradient_steps"] = int(cfg["gradient_steps"])
        action = {"algorithm": cfg["algorithm"],
                  "learning_rate": cfg["learning_rate"],
                  "buffer_size": cfg["buffer_size"],
                  "learning_starts": cfg["learning_starts"],
                  "batch_size": cfg["batch_size"],
                  "tau": cfg["tau"],
                  "gamma": cfg["gamma"],
                  "train_freq": cfg["train_freq"],
                  "gradient_steps": cfg["gradient_steps"],
                  }
    obs, reward, done, _ = train_env.step(action)
    return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Run PB2 for AutoRL"
    )
    parser.add_argument("--n_instances", type=int, default=1000, help="Number of instances in training environment")
    parser.add_argument("--env_seed", type=int, default=42, help="Random seed for the training environment")
    parser.add_argument("--outdir", type=str, default="tmp", help="Directory where to save trained models and logs.")
    args, _ = parser.parse_known_args()

    logdir = Path(args.outdir)
    logdir.mkdir(parents=True, exist_ok=True)

    train_env = gym.make("dac4carl-v0", n_instances=args.n_instances)
    train_env.seed(args.env_seed)
    done = False

    pbt = PB2(
        perturbation_interval=1,
        hyperparam_bounds={
            "algorithm": [0.0, 2.0],
            "learning_rate": [0.0001, 1.0],
            "gamma": [0.0001, 1.0],
            "gae_lambda": [0.0001, 1.0],
            "vf_coef": [0.0001, 1.0],
            "ent_coef": [0.0001, 1.0],
            "clip_range": [0.0001, 1.0],
            "buffer_size": [1000.0, 100000000.0],
            "learning_starts": [1.0, 10000.0],
            "batch_size": [8.0, 1024.0],
            "tau": [0.0001, 1.0],
            "train_freq": [1.0, 10000.0],
            "gradient_steps": [-1.0, 1000.0],
        })

    analysis = tune.run(
        evaluate_cost,
        name="pb2_baseline",
        scheduler=pbt,
        metric="mean_reward",
        mode="max",
        verbose=False,
        stop={
            "training_iteration": 30,
        },
        num_samples=8,
        fail_fast=True,
        config={
            "algorithm": 0.0,
            "learning_rate": 0.0001,
            "gamma": 0.9,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.5,
            "clip_range": 0.2,
            "buffer_size": 100000.0,
            "learning_starts": 100.0,
            "batch_size": 100.0,
            "tau": 0.1,
            "train_freq": 1.0,
            "gradient_steps": -1.0,
        })

    print("Best hyperparameters found were: ", analysis.best_config)

    #TODO: make dac policy out of schedule
    #question is if the best config is the schedule and if not how to get the schedule out of the analysis
    stuff = analysis.trial.config
    config_save_dir = logdir / "saved_configs"
    config_save_dir.mkdir(parents=True, exist_ok=True)
    incumbent_policy.save(config_save_dir)