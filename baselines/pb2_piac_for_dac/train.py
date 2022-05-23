import argparse
from pathlib import Path
import json

# import gym
# from DAC4RL import rlenv
from DAC4RL.baselines.pb2_piac_for_dac.solution import MultiSchedulePolicy

import ray
from ray import tune
from ray.tune.schedulers.pb2 import PB2
from ray.tune.examples.pbt_function import pbt_function


def evaluate_env(cfg, checkpoint_dir=None):
    import gym
    from DAC4RL import rlenv

    global args
    global logdir
    train_env = gym.make("dac4carl-v0")
    train_env.seed(args.env_seed)
    state = None
    while not state["env"] == cfg["env"]:
        state = train_env.reset()

    done = False
    while not done:
        if round(cfg["algorithm"]) == 0:
            algorithm = "PPO"
        elif round(cfg["algorithm"]) == 1:
            algorithm = "SAC"
        else:
            algorithm = "DDPG"

        if algorithm == "PPO":
            action = {
                "algorithm": algorithm,
                "learning_rate": float(cfg["learning_rate"]),
                "gamma": float(cfg["gamma"]),
                # "gae_lambda": float(cfg["gae_lambda"]),
                "vf_coef": float(cfg["vf_coef"]),
                "ent_coef": float(cfg["ent_coef"]),
                "clip_range": float(cfg["clip_range"]),
                }
        else:
            action = {
                "learning_rate": float(cfg["learning_rate"]),
                "buffer_size": int(cfg["buffer_size"]),
                "learning_starts": int(cfg["learning_starts"]),
                "batch_size": int(cfg["batch_size"]),
                "tau": float(cfg["tau"]),
                "gamma": float(cfg["gamma"]),
                "train_freq": int(cfg["train_freq"]),
                "gradient_steps": int(cfg["gradient_steps"]),
            }
        try:
            obs, reward, done, _ = train_env.step(action)
            trial_id = tune.get_trial_id()
            config = cfg.copy()
            config["learning_starts"] = int(cfg["learning_starts"])
            config["buffer_size"] = int(cfg["buffer_size"])
            config["batch_size"] = int(cfg["batch_size"])
            config["train_freq"] = int(cfg["train_freq"])
            config["gradient_steps"] = int(cfg["gradient_steps"])
            config["tau"] = float(cfg["tau"])
            config["gae_lambda"] = float(cfg["gae_lambda"])
            config["cf_coef"] = float(cfg["vf_coef"])
            config["ent_coef"] = float(cfg["ent_coef"])
            config["clip_range"] = float(cfg["clip_range"])

            with open(f"{logdir}/trial_{trial_id}.jsonl", "a+") as f:
                f.write(json.dumps(config))
                f.write("\n")
            tune.report(reward=reward)  # {"mean_reward": reward}
        except:
            done = True
            tune.report(reward=-50000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Run PB2 for each AutoRL env")
    parser.add_argument(
        "--n_instances",
        type=int,
        default=1000,
        help="Number of instances in training environment",
    )
    parser.add_argument(
        "--env_seed",
        type=int,
        default=42,
        help="Random seed for the training environment",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="tmp",
        help="Directory where to save trained models and logs.",
    )
    args, _ = parser.parse_known_args()

    logdir = Path(f"{args.outdir}/pb2_seed{args.env_seed}")
    logdir.mkdir(parents=True, exist_ok=True)

    envs = ['CARLPendulumEnv', 'CARLAcrobotEnv', 'CARLMountainCarContinuousEnv', 'CARLLunarLanderEnv', 'CARLCartPoleEnv',]
    analyses = []
    for env in envs:
        pbt = PB2(
            time_attr="training_iteration",
            perturbation_interval=1,
            hyperparam_bounds={
                "algorithm": [0.0, 2.0],
                "learning_rate": [0.0001, 1.0],
                "gamma": [0.0001, 1.0],
                "gae_lambda": [0.01, 0.99],
                "vf_coef": [0.0001, 0.999],
                "ent_coef": [0.0001, 0.999],
                "clip_range": [0.0001, 1.0],
                "buffer_size": [1000.0, 100000000.0],
                "learning_starts": [1.0, 10000.0],
                "batch_size": [8.0, 1024.0],
                "tau": [0.0001, 1.0],
                "train_freq": [1.0, 10000.0],
                "gradient_steps": [-1.0, 1000.0],
            },
        )

        analysis = tune.run(
            evaluate_env,
            name="pb2_baseline",
            scheduler=pbt,
            metric="reward",
            mode="max",
            verbose=True,
            stop={
                "training_iteration": 50,
            },
            num_samples=8,
            fail_fast=True,
            config={
                "instance": env,
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
            },
        )
        analyses.append(analysis)

    schedules = []
    for analysis in analyses:
        results = analysis.dataframe()
        best_process = analysis.best_dataframe["trial_id"].values[0]
        schedule = []
        with open(f"{logdir}/trial_{best_process}.jsonl", "r") as f:
            line = f.readline()
            schedule.append(json.loads(line))
        hyperparams = {}
        if schedule[0]["algorithm"] == 0:
            hyperparams["algorithm"] = "PPO"
        elif schedule[0]["algorithm"] == 1:
            hyperparams["algorithm"] = "SAC"
        else:
            hyperparams["algorithm"] = "DDPG"
        hyperparams["learning_rates"] = [t["learning_rate"] for t in schedule]
        hyperparams["gammas"] = [t["gamma"] for t in schedule]
        hyperparams["gae_lambdas"] = [t["gae_lambda"] for t in schedule]
        hyperparams["vf_coefs"] = [t["vf_coef"] for t in schedule]
        hyperparams["ent_coefs"] = [t["ent_coef"] for t in schedule]
        hyperparams["clip_ranges"] = [t["clip_range"] for t in schedule]
        hyperparams["batch_sizes"] = [t["batch_size"] for t in schedule]
        hyperparams["taus"] = [t["tau"] for t in schedule]
        hyperparams["learning_starts"] = [t["learning_starts"] for t in schedule]
        hyperparams["train_freqs"] = [t["train_freq"] for t in schedule]
        hyperparams["gradient_steps"] = [t["gradient_steps"] for t in schedule]
        hyperparams["buffer_sizes"] = [t["buffer_size"] for t in schedule]
        schedules.append(hyperparams)

    policy = MultiSchedulePolicy(**schedules)

    config_save_dir = logdir / "saved_configs"
    config_save_dir.mkdir(parents=True, exist_ok=True)
    policy.save(config_save_dir)
