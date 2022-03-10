import argparse
from pathlib import Path

import gym
from DAC4RL import rlenv
from DAC4RL.baselines import schedulers

import ray
from ray import tune
from ray.tune.schedulers.pb2 import PB2
from ray.tune.examples.pbt_function import pbt_function


def evaluate_cost(cfg, **kwargs):
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
    return {"mean_reward": reward}


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

    train_env = gym.make("dac4carl-v0")
    train_env.seed(args.env_seed)
    done = True

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
            "training_iteration": 120,
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

    results = analysis.dataframe()
    best_process = analysis.best_dataframe["pid"].values[0]
    best_schedule = results[results["pid"]==best_process].sort_values(by=["training_iteration"])
    hyperparams = {}
    if round(analysis.best_config["algorithm"]) == 0:
        hyperparams["algorithm"] = "PPO"
    elif round(analysis.best_config["algorithm"]) == 1:
        hyperparams["algorithm"] = "SAC"
    else:
        hyperparams["algorithm"] = "DDPG"
    hyperparams["learning_rates"] = list(best_schedule["config/learning_rate"].values)
    hyperparams["gammas"] = list(best_schedule["config/gamma"].values)
    hyperparams["gae_lambdas"] = list(best_schedule["config/gae_lambda"].values)
    hyperparams["vf_coefs"] = list(best_schedule["config/vf_coef"].values)
    hyperparams["ent_coefs"] = list(best_schedule["config/ent_coef"].values)
    hyperparams["clip_ranges"] = list(best_schedule["config/clip_range"].values)
    hyperparams["batch_sizes"] = [int(i) for i in list(best_schedule["config/batch_size"].values)]
    hyperparams["taus"] = list(best_schedule["config/tau"].values)
    hyperparams["learning_starts"] = [int(i) for i in list(best_schedule["config/learning_starts"].values)]
    hyperparams["train_freqs"] = [int(i) for i in list(best_schedule["config/train_freq"].values)]
    hyperparams["gradient_steps"] = [int(i) for i in list(best_schedule["config/gradient_steps"].values)]
    hyperparams["buffer_sizes"] = [int(i) for i in list(best_schedule["config/buffer_size"].values)]
    policy = schedulers.SchedulePolicy(**hyperparams)
    
    config_save_dir = logdir / "saved_configs"
    config_save_dir.mkdir(parents=True, exist_ok=True)
    policy.save(config_save_dir)
