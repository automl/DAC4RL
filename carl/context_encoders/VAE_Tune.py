from numpy.core.fromnumeric import std
import torch as th
import numpy as np
import os
import pdb
from tqdm import tqdm
import pickle
from mle_hyperopt import HyperbandSearch

from carl.context_encoders.Context_Beta_VAE import ContextBVAE
import json

import shutil
import hydra
from omegaconf import DictConfig


# @hydra.main(config_path='config.yaml')
def main() -> None:
    model = ContextBVAE(6, 2, [4, 3])

    out_dir = os.path.join("/home/mohan/git/CARL/tmp/VAE_Stuff/CartPole/VAE_Beta")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    context_db = os.path.join(
        "/home/mohan/git/CARL/tmp", "AE_stuff/AE_database/CartPole", "new_60k.npy"
    )

    dataset = np.load(context_db)
    np.random.shuffle(dataset)

    train_set = dataset[: int(0.8 * len(dataset))]
    val_set = dataset[int(-0.2 * len(dataset)) :]

    step = 0

    # Blackbox objective
    def train_model(**kwargs) -> None:

        for key in kwargs:
            print(key, kwargs[key])
            print("\t")

        iter_dir = os.path.join(out_dir, f"iter_{step}")
        if not os.path.exists(iter_dir):
            os.mkdir(os.path.join(iter_dir))
        else:
            shutil.rmtree(os.path.join(iter_dir))
            os.mkdir(os.path.join(iter_dir))

        optimizer = th.optim.Adam(
            model.parameters(), lr=kwargs["rate"], weight_decay=kwargs["decay"]
        )

        loader = th.utils.data.DataLoader(
            dataset=train_set, batch_size=kwargs["batch"], shuffle=False
        )

        losses = []
        recon = []
        kl = []

        epochs = 100

        for _ in tqdm(range(epochs)):

            for vector in loader:

                # Output of Autoencoder
                results = model(vector)

                # Calculating the loss function

                loss_dict = model.loss_function(*results, M_N=5e-3)
                loss = loss_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Storing the losses in a list for plotting
                losses.append(loss.item())
                recon.append(loss_dict["recon_loss"].item())
                kl.append(loss_dict["kl_loss"].item())

        print(
            f"Final Training Losses: {losses[-1]}\tFinal Training Recon: {recon[-1]}\tFinal Training KL:: {kl[-1]}"
        )

        ## Validation
        loader = th.utils.data.DataLoader(
            dataset=val_set, batch_size=kwargs["batch"], shuffle=False
        )

        val_losses = []
        val_recon = []
        val_kl = []

        for _ in range(10):
            for vector in loader:

                # Output of Autoencoder
                val_results = model(vector)

                # Calculating the loss function
                val_loss_dict = model.loss_function(*val_results, M_N=5e-3)

                val_loss = val_loss_dict["loss"]

                # Storing the losses in a list for plotting
                val_losses.append(val_loss.item())
                val_recon.append(val_loss_dict["recon_loss"].item())
                val_kl.append(val_loss_dict["kl_loss"].item())

        print(
            f"Mean Val Losses: {np.mean(val_losses)}\tMean Val Recon: {np.mean(val_recon)}\tMean Val KL {np.mean(val_kl)}"
        )

        ## Save the stuff

        np.save(os.path.join(iter_dir, "losses.npy"), losses)
        np.save(os.path.join(iter_dir, "recon_losses.npy"), recon)
        np.save(os.path.join(iter_dir, "kl_losses.npy"), kl)

        with open(os.path.join(iter_dir, "representations.pkl"), "wb") as f:
            pickle.dump(model.get_representation(), f)

        with open(os.path.join(iter_dir, "opt.pkl"), "wb") as f:
            pickle.dump(optimizer.state_dict(), f)

        th.save(model.state_dict(), os.path.join(iter_dir, "model.zip"))

        step = step + 1

    strategy = HyperbandSearch(
        real={
            "rate": {"begin": 0.001, "end": 0.5, "prior": "uniform"},
            "decay": {"begin": 1e-16, "end": 1e-4, "prior": "uniform"},
        },
        integer={"batch": {"begin": 30, "end": 128, "prior": "log-uniform"}},
        search_config={"max_resource": 14, "eta": 3},
        seed_id=42,
    )

    configs = strategy.ask()

    with open(os.path.join(out_dir, "configs.json"), "w") as f:
        json.dump(configs, f, indent=4)

    for c in configs:
        train_model(model, out_dir, **c["params"])


if __name__ == "__main__":
    main()
