#!/usr/bin/env python3
import copy
import logging
import os
import time
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import hydra
import numpy as np
import pandas as pd
import src
import torch
import wandb
from experiments.sl.train import checkpoint
from experiments.sl.utils import (
    compute_metrics,
    EarlyStopper,
    set_seed_everywhere,
    compute_metrics_regression,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from tqdm import tqdm


class TableLogger:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # Data dictionary used to make pd.DataFrame
        self.data = {
            "dataset": [],
            "model": [],
            "seed": [],
            "num_inducing": [],
            "acc": [],
            "nlpd": [],
            "ece": [],
            "mse": [],
            "prior_prec": [],
            "time": [],
            "method": [],
        }

    def add_data(
        self,
        model_name: str,
        metrics: dict,
        prior_prec: float,
        num_inducing: Optional[int] = None,
        time: float = None,
        method: str = None,
    ):
        "Add NLL to data dict and wandb table"
        if isinstance(prior_prec, torch.Tensor):
            prior_prec = prior_prec.item()
        self.data["dataset"].append(self.cfg.dataset.name)
        self.data["model"].append(model_name)
        self.data["seed"].append(self.cfg.random_seed)
        self.data["num_inducing"].append(num_inducing)

        print(f"meterics {metrics}")
        self.data["nlpd"].append(metrics["nll"])
        try:
            self.data["acc"].append(metrics["acc"])
        except:
            self.data["acc"].append("")
        try:
            self.data["mse"].append(metrics["mse"])
        except:
            self.data["mse"].append("")
        try:
            self.data["ece"].append(metrics["acc"])
        except:
            self.data["ece"].append("")

        self.data["prior_prec"].append(prior_prec)
        self.data["time"].append(time)
        self.data["method"].append(method)
        if self.cfg.wandb.use_wandb:
            wandb.log({"Metrics": wandb.Table(data=pd.DataFrame(self.data))})


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="dual_updates")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd

    ##### Make experiment reproducible #####
    try:
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    ##### Use CUDA if it's requested AND available #####
    if cfg.device in "cuda":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = "cpu"
    logger.info(f"Using device: {cfg.device}")

    table_logger = TableLogger(cfg)

    ##### Use double precision #####
    torch.set_default_dtype(torch.double)

    ##### Load train/val/test/update data sets #####
    ds_train, ds_val, ds_test, ds_update = hydra.utils.instantiate(
        cfg.dataset, dir=os.path.join(get_original_cwd(), "data"), double=True
    )
    output_dim = ds_train.output_dim
    logger.info(f"D: {len(ds_train)}")
    logger.info(f"F: {ds_train.output_dim}")
    try:
        logger.info(f"D: {ds_train.data.shape[-1]}")
    except:
        pass
    ds_train = ConcatDataset([ds_train, ds_val])
    ds_train.output_dim = output_dim

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)
    update_loader = DataLoader(ds_update, batch_size=cfg.batch_size, shuffle=True)
    train_and_update_loader = DataLoader(
        ConcatDataset([ds_train, ds_update]), batch_size=cfg.batch_size, shuffle=True
    )

    ##### Init Weight and Biases #####
    cfg.output_dim = ds_train.output_dim
    print(OmegaConf.to_yaml(cfg))
    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
        run_dir = run.dir
    else:
        run_dir = "./"

    ##### Instantiate the neural network #####
    network = hydra.utils.instantiate(cfg.network, ds_train=ds_train)

    ##### Instantiate SFR #####
    sfr = hydra.utils.instantiate(cfg.sfr, model=network)
    sfr.double()
    logger.info(f"SFR: \n {sfr}")
    if isinstance(sfr.likelihood, src.likelihoods.Gaussian):
        likelihood = "regresssion"
    else:
        likelihood = "classification"

    ##### Sample Z from train and update #####
    ds_train_and_update = ConcatDataset([ds_train, ds_update])
    indices = torch.randperm(len(ds_train_and_update))[: sfr.num_inducing]
    Z_ds = torch.utils.data.Subset(ds_train_and_update, indices)
    Z_ds = DataLoader(Z_ds, batch_size=len(Z_ds))
    Z = next(iter(Z_ds))[0].to(sfr.device)
    sfr.Z = Z.to(sfr.device)

    @torch.no_grad()
    def loss_fn(data_loader: DataLoader, model: src.SFR = None):
        if model is None:
            model = sfr
        losses = []
        for X, y in data_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            loss = model.loss(X, y)
            losses.append(loss)
        losses = torch.stack(losses, 0)
        cum_loss = torch.mean(losses, 0)
        return cum_loss

    def train_loop(sfr, data_loader: DataLoader, val_loader: DataLoader):
        params = [{"params": sfr.parameters()}]
        if isinstance(sfr.likelihood, src.likelihoods.Gaussian):
            params.append({"params": sfr.likelihood.log_sigma_noise})
        optimizer = torch.optim.Adam(params, lr=cfg.lr)

        early_stopper = EarlyStopper(
            patience=int(cfg.early_stop.patience / cfg.logging_epoch_freq),
            min_prior_precision=cfg.early_stop.min_prior_precision,
        )

        @torch.no_grad()
        def map_pred_fn(x, idx=None):
            f = sfr.network(x.to(cfg.device))
            return sfr.likelihood.inv_link(f)

        best_loss = float("inf")
        for epoch in tqdm(list(range(cfg.n_epochs))):
            for X, y in data_loader:
                X = X.to(cfg.device)
                y = y.to(cfg.device)
                loss = sfr.loss(X, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if cfg.wandb.use_wandb:
                    wandb.log({"loss": loss})
                    if isinstance(sfr.likelihood, src.likelihoods.Gaussian):
                        wandb.log({"log_sigma_noise": sfr.likelihood.sigma_noise})

            val_loss = loss_fn(val_loader, model=sfr)
            if epoch % cfg.logging_epoch_freq == 0 and cfg.wandb.use_wandb:
                wandb.log({"val_loss": val_loss})
                if likelihood == "classification":
                    train_metrics = compute_metrics(
                        pred_fn=map_pred_fn, data_loader=train_loader, device=cfg.device
                    )
                    val_metrics = compute_metrics(
                        pred_fn=map_pred_fn, data_loader=val_loader, device=cfg.device
                    )
                    test_metrics = compute_metrics(
                        pred_fn=map_pred_fn, data_loader=test_loader, device=cfg.device
                    )
                else:
                    train_metrics = compute_metrics_regression(
                        model=sfr, data_loader=train_loader, device=cfg.device, map=True
                    )
                    val_metrics = compute_metrics_regression(
                        model=sfr, data_loader=val_loader, device=cfg.device, map=True
                    )
                    test_metrics = compute_metrics_regression(
                        model=sfr, data_loader=test_loader, device=cfg.device, map=True
                    )
                wandb.log({"train/": train_metrics})
                wandb.log({"val/": val_metrics})
                wandb.log({"test/": test_metrics})
                wandb.log({"epoch": epoch})

            if val_loss < best_loss:
                best_ckpt_fname = checkpoint(
                    sfr=sfr, optimizer=optimizer, save_dir=run_dir
                )
                best_loss = val_loss
            if early_stopper(val_loss):
                logger.info("Early stopping criteria met, stopping training...")
                break

        ##### Load checkpoint #####
        ckpt = torch.load(best_ckpt_fname)
        sfr.load_state_dict(ckpt["model"])
        return sfr

    def train_and_log(
        sfr: src.SFR,
        train_loader: DataLoader,
        name: str,
        inference_loader: Optional[DataLoader] = None,
        train_val_split: float = 0.7,
    ):
        ds_train, ds_val = torch.utils.data.random_split(
            train_loader.dataset, [train_val_split, 1 - train_val_split]
        )
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=True)

        ##### Train NN #####
        sfr.train()
        start_time = time.time()
        sfr = train_loop(sfr, data_loader=train_loader, val_loader=val_loader)
        train_time = time.time() - start_time

        ##### Log MAP before updates #####
        sfr.double()
        sfr.eval()
        log_map_metrics(
            sfr,
            test_loader,
            name=name,
            table_logger=table_logger,
            device=cfg.device,
            time=train_time,
        )

        ##### Fit SFR #####
        logger.info("Fitting SFR...")
        if inference_loader is None:
            inference_loader = train_loader
        start_time = time.time()

        sfr.Z = Z
        all_train = DataLoader(
            inference_loader.dataset, batch_size=len(inference_loader.dataset)
        )
        sfr.train_data = next(iter(all_train))
        sfr._build_sfr()

        inference_time = time.time() - start_time
        logger.info("Finished fitting SFR")
        log_sfr_metrics(
            sfr,
            name=name,
            test_loader=test_loader,
            table_logger=table_logger,
            device=cfg.device,
            time=inference_time + train_time,
        )
        return sfr

    ##### Train on D1 and log #####
    sfr = train_and_log(sfr, train_loader=train_loader, name="Train D1")

    ##### Make a copy of SFR #####
    network_ = hydra.utils.instantiate(cfg.network, ds_train=ds_train)
    network_.double()
    network_.load_state_dict(sfr.network.state_dict())  # copy weights and stuff
    sfr_copy = hydra.utils.instantiate(cfg.sfr, model=network_)
    sfr_copy.double()

    ##### Dual updates on D2 and log #####
    start_time = time.time()
    sfr.update(data_loader=update_loader)
    update_inference_time = time.time() - start_time
    log_sfr_metrics(
        sfr,
        name="Train D1 -> Update D2",
        test_loader=test_loader,
        table_logger=table_logger,
        device=cfg.device,
        time=update_inference_time,
    )

    ##### Continue training on D1+D2 and log #####
    sfr = train_and_log(
        sfr, train_loader=train_and_update_loader, name="Train D1 -> Train D1+D2"
    )

    ##### Train on D1+D2 (from scratch) and log #####
    network_new = hydra.utils.instantiate(cfg.network, ds_train=ds_train)
    network_new.double()
    sfr_new = hydra.utils.instantiate(cfg.sfr, model=network_new)
    sfr_new.double()
    sfr_new = train_and_log(
        sfr_new, train_loader=train_and_update_loader, name="Train D1+D2"
    )

    ##### Continue training on just D2 and log #####
    logger.info("Continue training on just D2 and log")
    _ = train_and_log(
        sfr_copy,
        train_loader=update_loader,
        name="Train D1 -> Train D2",
        inference_loader=train_and_update_loader,
    )

    return table_logger


def log_map_metrics(
    sfr, test_loader, name: str, table_logger, device, time: float = None
):
    from experiments.sl.utils import compute_metrics

    @torch.no_grad()
    def map_pred_fn(x, idx=None):
        f = sfr.network(x.to(device))
        return sfr.likelihood.inv_link(f)

    if isinstance(sfr.likelihood, src.likelihoods.Gaussian):
        map_metrics = compute_metrics_regression(
            model=sfr, data_loader=test_loader, device=device, map=True
        )
    else:
        map_metrics = compute_metrics(
            pred_fn=map_pred_fn, data_loader=test_loader, device=device
        )
    table_logger.add_data(
        "NN MAP",
        metrics=map_metrics,
        num_inducing=None,
        prior_prec=sfr.prior.prior_precision,
        time=time,
        method=name,
    )
    logger.info(f"map_metrics: {map_metrics}")


def log_sfr_metrics(
    sfr,
    test_loader,
    name: str,
    table_logger: TableLogger,
    device="cuda",
    num_samples=100,
    time: float = None,
):
    from experiments.sl.utils import sfr_pred

    if isinstance(sfr.likelihood, src.likelihoods.Gaussian):
        gp_metrics = compute_metrics_regression(
            model=sfr, pred_type="gp", data_loader=test_loader, device=device
        )
    else:
        gp_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
    table_logger.add_data(
        "SFR (GP)",
        metrics=gp_metrics,
        num_inducing=sfr.num_inducing,
        prior_prec=sfr.prior.prior_precision,
        time=time,
        method=name,
    )
    logger.info(f"SFR metrics: {gp_metrics}")


if __name__ == "__main__":
    main()  # pyright: ignore
