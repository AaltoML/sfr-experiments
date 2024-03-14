#!/usr/bin/env python3
import logging
import os
import random
import shutil

import hydra
import omegaconf
import src
import torch
import wandb
from experiments.sl.cfgs.schema import TrainConfig
from experiments.sl.utils import (
    checkpoint,
    compute_metrics,
    EarlyStopper,
    set_seed_everywhere,
    train_val_split,
)
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train")
def train(cfg: TrainConfig):
    """Trains a NN with the loss configured via SFR's likelihood & prior"""

    ##### Make experiment reproducible #####
    try:
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    if "cuda" in cfg.device:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(cfg.device))

    ##### Optionally train in double precision #####
    if cfg.double_train:
        logger.info("Using float64")
        torch.set_default_dtype(torch.double)

    ##### Ensure all operations are deterministic on GPU #####
    eval('setattr(torch.backends.cudnn, "determinstic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    ##### Load the data with train/val/test split #####
    ds_train, ds_val, ds_test, _ = hydra.utils.instantiate(
        cfg.dataset, dir=os.path.join(get_original_cwd(), "data")
    )
    cfg.output_dim = ds_train.output_dim
    print(f"output_dim: {cfg.output_dim}")
    print(cfg.dataset.name)

    ##### Instantiate the neural network #####
    network = hydra.utils.instantiate(cfg.network, ds_train=ds_train)

    ##### Create data loaders #####
    train_loader = DataLoader(dataset=ds_train, shuffle=True, batch_size=cfg.batch_size)
    val_loader = DataLoader(dataset=ds_val, shuffle=False, batch_size=cfg.batch_size)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    ##### Instantiate SFR #####
    sfr = hydra.utils.instantiate(cfg.sfr, model=network)

    ##### Initialise W&B #####
    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            shutil.copytree(
                os.path.abspath(".hydra"),
                os.path.join(os.path.join(get_original_cwd(), wandb.run.dir), "hydra"),
            )
            wandb.save("hydra")
        except FileExistsError:
            pass

    optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=cfg.lr)

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x.to(cfg.device))
        return sfr.likelihood.inv_link(f)

    @torch.no_grad()
    def loss_fn(data_loader: DataLoader):
        cum_loss = 0
        for X, y in data_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            cum_loss += loss
        return cum_loss / len(data_loader.dataset)

    early_stopper = EarlyStopper(
        patience=int(cfg.early_stop.patience / cfg.logging_epoch_freq),
        min_prior_precision=cfg.early_stop.min_prior_precision,
    )

    best_val_loss = float("inf")
    for epoch in tqdm(list(range(cfg.n_epochs))):
        for X, y in train_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = sfr.loss(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if wandb.run is not None:
                wandb.log({"loss": loss})

        if epoch % cfg.logging_epoch_freq == 0:
            val_loss = loss_fn(val_loader)
            if wandb.run is not None:
                wandb.log({"val_loss": val_loss})
            train_metrics = compute_metrics(
                pred_fn=map_pred_fn,
                data_loader=train_loader,
                device=cfg.device,
            )
            val_metrics = compute_metrics(
                pred_fn=map_pred_fn,
                data_loader=val_loader,
                device=cfg.device,
            )
            test_metrics = compute_metrics(
                pred_fn=map_pred_fn,
                data_loader=test_loader,
                device=cfg.device,
            )
            if wandb.run is not None:
                wandb.log({"train/": train_metrics})
                wandb.log({"val/": val_metrics})
                wandb.log({"test/": test_metrics})
                wandb.log({"epoch": epoch})

            #### Save a checkpoint if sfr has best val loss #####
            if val_loss < best_val_loss:
                best_ckpt_fname = checkpoint(
                    sfr=sfr, optimizer=optimizer, save_dir="./"
                )
                best_val_loss = val_loss
                if wandb.run is not None:
                    wandb.log({"best_test/": test_metrics})
                    wandb.log({"best_val/": val_metrics})

            #### Break training loop if val loss has stopped decreasing #####
            if early_stopper(val_loss):
                logger.info("Early stopping criteria met, stopping training...")
                break

    logger.info("Finished training")

    #### Load best checkpoint #####
    ckpt = torch.load(best_ckpt_fname)
    sfr.load_state_dict(ckpt["model"])
    return sfr


if __name__ == "__main__":
    train()  # pyright: ignore
    # train_on_cluster()  # pyright: ignore
