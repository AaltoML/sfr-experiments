#!/usr/bin/env python3
import copy
import logging
import os
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


class TableLogger:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # Data dictionary used to make pd.DataFrame
        self.data = {
            "dataset": [],
            "model": [],
            "seed": [],
            "num_inducing": [],
            "num_inducing_percent": [],
            "acc": [],
            "nlpd": [],
            "ece": [],
            "prior_prec": [],
            "ood_auc": [],
        }
        self.tbl = wandb.Table(
            columns=[
                "dataset",
                "model",
                "seed",
                "num_inducing",
                "num_inducing_percent",
                "acc",
                "nlpd",
                "ece",
                "prior_prec",
                "ood_auc",
            ]
        )

    def add_data(
        self,
        model_name: str,
        metrics: dict,
        prior_prec: float,
        num_inducing: Optional[int] = None,
        num_inducing_percent: Optional[int] = None,
    ):
        "Add NLL to data dict and wandb table"
        if isinstance(prior_prec, torch.Tensor):
            prior_prec = prior_prec.item()
        self.data["dataset"].append(self.cfg.dataset.name)
        self.data["model"].append(model_name)
        self.data["seed"].append(self.cfg.random_seed)
        self.data["num_inducing"].append(num_inducing)
        self.data["num_inducing_percent"].append(num_inducing_percent)
        self.data["acc"].append(metrics["acc"])
        self.data["nlpd"].append(metrics["nll"])
        self.data["ece"].append(metrics["ece"])
        self.data["prior_prec"].append(prior_prec)
        self.data["ood_auc"].append(metrics["ood_auc"])
        self.tbl.add_data(
            self.cfg.dataset.name,
            model_name,
            self.cfg.random_seed,
            num_inducing,
            num_inducing_percent,
            metrics["acc"],
            metrics["nll"],
            metrics["ece"],
            prior_prec,
            metrics["ood_auc"],
        )
        if wandb.run is not None:
            wandb.log({"Metrics": wandb.Table(data=pd.DataFrame(self.data))})

        data_dict = {
            "id": pd.Series(metrics["id_entropies"]),
            "ood": pd.Series(metrics["ood_entropies"]),
        }
        table = wandb.Table(dataframe=pd.DataFrame(data_dict))
        if wandb.run is not None:
            if num_inducing is not None:
                wandb.log({f"entropies/{model_name}_M={num_inducing}": table})
            else:
                wandb.log({f"entropies/{model_name}": table})


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train_and_inference")
def train_and_inference(cfg: DictConfig):
    import experiments
    from experiments.sl.cluster_train import train
    from hydra.utils import get_original_cwd

    if cfg.device in "cuda":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = "cpu"
    print("Using device: {}".format(cfg.device))

    table_logger = TableLogger(cfg)

    torch.set_default_dtype(torch.float)

    ##### Load train/val/test data sets #####
    ds_train, ds_val, ds_test, _ = hydra.utils.instantiate(
        cfg.dataset, dir=os.path.join(get_original_cwd(), "data")
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

    ##### Train #####
    sfr = train(cfg)  # Train the NN

    ##### Make everything double for inference #####
    torch.set_default_dtype(torch.double)
    sfr.double()
    sfr.eval()

    ds_train, ds_val, ds_test, _ = hydra.utils.instantiate(
        cfg.dataset,
        dir=os.path.join(get_original_cwd(), "data"),
        double=cfg.double_inference,
    )
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True)

    if cfg.dataset.name in "CIFAR10":
        ood_ds = experiments.sl.bnn_predictive.preds.datasets.SVHN(
            root=os.path.join(get_original_cwd(), "data"),
            train=False,
            download=True,
            # transform=CIFAR10_transform,
        )
        # ood_ds = torchvision.datasets.SVHN(
        #     "data", split="test", transform=None, download=True
        # )
        ood_loader = DataLoader(ood_ds, batch_size=cfg.batch_size, shuffle=True)
    elif cfg.dataset.name in "FMNIST":
        import experiments

        _, _, ood_ds, _ = experiments.sl.utils.get_image_dataset(
            dir=os.path.join(get_original_cwd(), "data"),
            name="MNIST",
            double=cfg.double_inference,
            device=cfg.device,
            debug=cfg.dataset.debug,
            val_from_test=False,
            val_split=0.5,
        )
        ood_loader = DataLoader(ood_ds, batch_size=cfg.batch_size, shuffle=True)
    else:
        ood_loader = None

    ##### Log MAP NLPD #####
    log_map_metrics(
        sfr,
        test_loader,
        ood_loader=ood_loader,
        table_logger=table_logger,
        device=cfg.device,
    )

    ##### Log Laplace BNN/GLM NLPD/ACC/ECE #####
    if cfg.run_laplace_flag:
        print("starting laplace")
        torch.cuda.empty_cache()
        for hessian_structure in cfg.hessian_structures:
            torch.cuda.empty_cache()
            log_la_metrics(
                network=sfr.network,
                prior_precision=sfr.prior.prior_precision,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                ood_loader=ood_loader,
                table_logger=table_logger,
                device=cfg.device,
                posthoc_prior_opt=cfg.posthoc_prior_opt_laplace,
                hessian_structure=hessian_structure,
            )

    if cfg.run_sfr_flag:
        num_data = len(ds_train)
        print(f"NUM_DATA {num_data}")
        logger.info(f"num_data: {num_data}")
        for num_inducing in cfg.num_inducings:
            if cfg.num_inducings_as_percent:
                print(f"NUM_INDUCING PERCENT {num_inducing}")
                num_inducing_percent = num_inducing
                num_inducing = int(num_data * num_inducing / 100)
                print(f"num_inducing {num_inducing}")
            else:
                num_inducing_percent = int(num_inducing / num_data * 100)
            torch.cuda.empty_cache()
            ##### Log SFR GP/NN NLPD/ACC/ECE #####
            log_sfr_metrics(
                network=sfr.network,
                output_dim=ds_train.output_dim,
                prior_precision=sfr.prior.prior_precision,  # TODO how to set this?
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                ood_loader=ood_loader,
                table_logger=table_logger,
                num_inducing=num_inducing,
                num_inducing_percent=num_inducing_percent,
                dual_batch_size=cfg.dual_batch_size,
                device=cfg.device,
                posthoc_prior_opt_grid=cfg.posthoc_prior_opt_grid,
                posthoc_prior_opt_bo=cfg.posthoc_prior_opt_bo,
                num_bo_trials=cfg.num_bo_trials,
                EPS=cfg.EPS,
                jitter=cfg.jitter,
                log_sfr_nn_flag=cfg.run_sfr_nn_flag,
            )

            ##### Log GP GP/NN NLPD/ACC/ECE #####
            log_gp_metrics(
                network=sfr.network,
                output_dim=ds_train.output_dim,
                prior_precision=sfr.prior.prior_precision,  # TODO how to set this?
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                ood_loader=ood_loader,
                table_logger=table_logger,
                num_inducing=num_inducing,
                num_inducing_percent=num_inducing_percent,
                dual_batch_size=cfg.dual_batch_size,
                device=cfg.device,
                posthoc_prior_opt_grid=cfg.posthoc_prior_opt_grid,
                posthoc_prior_opt_bo=cfg.posthoc_prior_opt_bo,
                num_bo_trials=cfg.num_bo_trials,
                EPS=cfg.EPS,
                jitter=cfg.jitter,
            )

    ##### Log table on W&B and save latex table as .tex #####
    df = pd.DataFrame(table_logger.data)
    print(df)
    if wandb.run is not None:
        wandb.log({"Metrics": wandb.Table(data=df)})

    df_latex = df.to_latex(escape=False)
    print(df_latex)

    with open("uci_table.tex", "w") as file:
        file.write(df_latex)
        if wandb.run is not None:
            wandb.save("uci_table.tex")


def log_map_metrics(sfr, test_loader, ood_loader, table_logger, device):
    from experiments.sl.utils import compute_metrics

    @torch.no_grad()
    def map_pred_fn(x):
        f = sfr.network(x.to(device))
        return sfr.likelihood.inv_link(f)

    map_metrics = compute_metrics(
        pred_fn=map_pred_fn,
        data_loader=test_loader,
        ood_loader=ood_loader,
        device=device,
    )
    table_logger.add_data(
        "NN MAP",
        metrics=map_metrics,
        num_inducing=None,
        prior_prec=sfr.prior.prior_precision,
    )
    logger.info(f"map_metrics: {map_metrics}")


def log_sfr_metrics(
    network,
    output_dim,
    prior_precision,
    train_loader,
    val_loader,
    test_loader,
    ood_loader,
    table_logger: TableLogger,
    num_inducing: int = 128,
    num_inducing_percent: Optional[int] = None,
    dual_batch_size: int = 1000,
    device="cuda",
    posthoc_prior_opt_grid: bool = True,
    posthoc_prior_opt_bo: bool = True,
    num_bo_trials: int = 20,
    num_samples: int = 100,
    EPS=0.01,
    # EPS=0.0,
    jitter: float = 1e-6,
    log_sfr_nn_flag: bool = False,
):
    import src
    from experiments.sl.utils import (
        compute_metrics,
        init_SFR_with_gaussian_prior,
        sfr_pred,
    )

    likelihood = src.likelihoods.CategoricalLh(EPS=EPS)
    sfr = init_SFR_with_gaussian_prior(
        model=network,
        prior_precision=prior_precision,  # TODO what should this be
        likelihood=likelihood,
        output_dim=output_dim,
        num_inducing=num_inducing,
        dual_batch_size=dual_batch_size,
        jitter=jitter,
        device=device,
    )
    sfr.double()
    sfr.eval()
    logger.info("Fitting SFR...")
    sfr.fit(train_loader=train_loader)
    logger.info("Finished fitting SFR")

    ##### Metrics for prediction with NN as mean #####
    if log_sfr_nn_flag:
        nn_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="nn", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            ood_loader=ood_loader,
            device=device,
        )
        table_logger.add_data(
            "SFR (NN)",
            metrics=nn_metrics,
            num_inducing=num_inducing,
            num_inducing_percent=num_inducing_percent,
            prior_prec=sfr.prior.prior_precision,
        )

    ##### Metrics for prediction with GP mean #####
    gp_metrics = compute_metrics(
        pred_fn=sfr_pred(
            model=sfr, pred_type="gp", num_samples=num_samples, device=device
        ),
        data_loader=test_loader,
        ood_loader=ood_loader,
        device=device,
    )
    table_logger.add_data(
        "SFR (GP)",
        metrics=gp_metrics,
        num_inducing=num_inducing,
        num_inducing_percent=num_inducing_percent,
        prior_prec=sfr.prior.prior_precision,
    )

    if posthoc_prior_opt_bo:
        # Copy prior_prec so it can be reset for GP prediction
        prior_prec = copy.copy(sfr.prior.prior_precision)

        ##### Run with NN as mean #####
        if log_sfr_nn_flag:
            sfr.prior_precision = copy.copy(prior_prec)
            sfr.optimize_prior_precision(
                pred_type="nn",
                val_loader=val_loader,
                method="bo",
                prior_prec_min=1e-8,
                prior_prec_max=1.0,
                num_trials=num_bo_trials,
            )
            gp_metrics_bo = compute_metrics(
                pred_fn=sfr_pred(
                    model=sfr, pred_type="nn", num_samples=num_samples, device=device
                ),
                data_loader=test_loader,
                ood_loader=ood_loader,
                device=device,
            )
            table_logger.add_data(
                "SFR (NN) BO",
                metrics=gp_metrics_bo,
                num_inducing=num_inducing,
                num_inducing_percent=num_inducing_percent,
                prior_prec=sfr.prior.prior_precision,
            )

        ##### Run with GP mean prediction #####
        sfr.prior_precision = copy.copy(prior_prec)
        sfr.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="bo",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=num_bo_trials,
        )
        gp_metrics_bo = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            ood_loader=ood_loader,
            device=device,
        )
        table_logger.add_data(
            "SFR (GP) BO",
            metrics=gp_metrics_bo,
            num_inducing=num_inducing,
            num_inducing_percent=num_inducing_percent,
            prior_prec=sfr.prior.prior_precision,
        )

    if posthoc_prior_opt_grid:
        sfr.prior_precision = copy.copy(prior_prec)
        sfr.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="grid",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=50,
        )
        gp_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            ood_loader=ood_loader,
            device=device,
        )
        table_logger.add_data(
            "SFR (GP) GRID",
            metrics=gp_metrics,
            num_inducing=num_inducing,
            num_inducing_percent=num_inducing_percent,
            prior_prec=sfr.prior.prior_precision,
        )

    if posthoc_prior_opt_grid:
        sfr.prior_precision = copy.copy(prior_prec)
        sfr.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="grid",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=50,
        )
        nn_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=sfr, pred_type="nn", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            device=device,
        )
        table_logger.add_data(
            "SFR (NN) GRID",
            metrics=nn_metrics,
            num_inducing=num_inducing,
            num_inducing_percent=num_inducing_percent,
            prior_prec=sfr.prior.prior_precision,
        )
        # logger.info(f"map_metrics: {map_metrics}")


def log_gp_metrics(
    network,
    output_dim,
    prior_precision,
    train_loader,
    val_loader,
    test_loader,
    ood_loader,
    table_logger: TableLogger,
    num_inducing: int = 128,
    num_inducing_percent: Optional[int] = None,
    dual_batch_size: int = 1000,
    # device="cpu",
    device="cuda",
    posthoc_prior_opt_grid: bool = True,
    posthoc_prior_opt_bo: bool = True,
    num_bo_trials: int = 20,
    num_samples: int = 100,
    EPS=0.01,
    jitter: float = 1e-6,
):
    import src
    from experiments.sl.utils import (
        compute_metrics,
        init_NN2GPSubset_with_gaussian_prior,
        sfr_pred,
    )

    likelihood = src.likelihoods.CategoricalLh(EPS=EPS)
    gp = init_NN2GPSubset_with_gaussian_prior(
        model=network,
        prior_precision=prior_precision,  # TODO what should this be
        likelihood=likelihood,
        output_dim=output_dim,
        subset_size=num_inducing,
        dual_batch_size=dual_batch_size,
        jitter=jitter,
        device=device,
    )
    gp = gp.double()
    gp.eval()
    logger.info("Fitting GP...")
    gp.fit(train_loader=train_loader)
    logger.info("Finished fitting GP")

    nn_metrics = compute_metrics(
        pred_fn=sfr_pred(
            model=gp, pred_type="nn", num_samples=num_samples, device=device
        ),
        data_loader=test_loader,
        ood_loader=ood_loader,
        device=device,
    )
    table_logger.add_data(
        "GP Subset (NN)",
        metrics=nn_metrics,
        num_inducing=num_inducing,
        num_inducing_percent=num_inducing_percent,
        prior_prec=gp.prior.prior_precision,
    )

    gp_metrics = compute_metrics(
        pred_fn=sfr_pred(
            model=gp, pred_type="gp", num_samples=num_samples, device=device
        ),
        data_loader=test_loader,
        ood_loader=ood_loader,
        device=device,
    )
    table_logger.add_data(
        "GP Subset (GP)",
        metrics=gp_metrics,
        num_inducing=num_inducing,
        num_inducing_percent=num_inducing_percent,
        prior_prec=gp.prior.prior_precision,
    )

    if posthoc_prior_opt_bo:
        # Copy prior_prec so it can be reset for GP prediction
        prior_prec = copy.copy(gp.prior.prior_precision)

        ##### Run with NN mean #####
        gp.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="bo",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=num_bo_trials,
        )
        nn_metrics_bo = compute_metrics(
            pred_fn=sfr_pred(
                model=gp, pred_type="nn", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            ood_loader=ood_loader,
            device=device,
        )
        table_logger.add_data(
            "GP Subset (NN) BO",
            metrics=nn_metrics_bo,
            num_inducing=num_inducing,
            num_inducing_percent=num_inducing_percent,
            prior_prec=gp.prior.prior_precision,
        )

        gp.prior_precision = copy.copy(prior_prec)
        gp.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="bo",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=num_bo_trials,
        )
        gp_metrics_bo = compute_metrics(
            pred_fn=sfr_pred(
                model=gp, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            ood_loader=ood_loader,
            device=device,
        )
        table_logger.add_data(
            "GP Subset (GP) BO",
            metrics=gp_metrics_bo,
            num_inducing=num_inducing,
            num_inducing_percent=num_inducing_percent,
            prior_prec=gp.prior.prior_precision,
        )

    if posthoc_prior_opt_grid:
        gp.prior_precision = copy.copy(prior_prec)
        gp.optimize_prior_precision(
            pred_type="gp",
            val_loader=val_loader,
            method="grid",
            prior_prec_min=1e-8,
            prior_prec_max=1.0,
            num_trials=40,
        )
        gp_metrics = compute_metrics(
            pred_fn=sfr_pred(
                model=gp, pred_type="gp", num_samples=num_samples, device=device
            ),
            data_loader=test_loader,
            ood_loader=ood_loader,
            device=device,
        )
        table_logger.add_data(
            "GP Subset (GP) GRID",
            metrics=gp_metrics,
            num_inducing=num_inducing,
            num_inducing_percent=num_inducing_percent,
            prior_prec=gp.prior.prior_precision,
        )

        if posthoc_prior_opt_grid:
            gp.prior_precision = copy.copy(prior_prec)
            gp.optimize_prior_precision(
                pred_type="nn",
                val_loader=val_loader,
                method="grid",
                prior_prec_min=1e-8,
                prior_prec_max=1.0,
                num_trials=50,
            )
            nn_metrics = compute_metrics(
                pred_fn=sfr_pred(
                    model=gp, pred_type="nn", num_samples=num_samples, device=device
                ),
                data_loader=test_loader,
                ood_loader=ood_loader,
                device=device,
            )
            table_logger.add_data(
                "GP Subset (NN) GRID",
                metrics=nn_metrics,
                num_inducing=num_inducing,
                num_inducing_percent=num_inducing_percent,
                prior_prec=gp.prior.prior_precision,
            )


def log_la_metrics(
    network,
    prior_precision: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    ood_loader: DataLoader,
    table_logger: TableLogger,
    device: str,
    posthoc_prior_opt: bool = True,
    num_samples: int = 100,
    hessian_structure: str = "kron",
):
    import laplace
    from experiments.sl.utils import compute_metrics, la_pred

    la = laplace.Laplace(
        likelihood="classification",
        subset_of_weights="all",
        hessian_structure=hessian_structure,
        sigma_noise=1,
        backend=laplace.curvature.asdl.AsdlGGN,
        model=network,
    )
    print(f"la {la}")
    la.prior_precision = prior_precision
    logger.info("Fitting Laplace...")
    la.fit(train_loader)
    logger.info("Finished fitting Laplace")

    bnn_pred_fn = la_pred(
        model=la,
        pred_type="nn",
        link_approx="mc",
        num_samples=num_samples,
        device=device,
    )
    bnn_metrics = compute_metrics(
        pred_fn=bnn_pred_fn,
        data_loader=test_loader,
        ood_loader=ood_loader,
        device=device,
    )
    table_logger.add_data(
        f"BNN {hessian_structure}", metrics=bnn_metrics, prior_prec=la.prior_precision
    )
    torch.cuda.empty_cache()
    glm_pred_fn = la_pred(
        model=la,
        pred_type="glm",
        link_approx="mc",
        num_samples=num_samples,
        device=device,
    )
    glm_metrics = compute_metrics(
        pred_fn=glm_pred_fn,
        data_loader=test_loader,
        ood_loader=ood_loader,
        device=device,
    )
    table_logger.add_data(
        f"GLM {hessian_structure}", metrics=glm_metrics, prior_prec=la.prior_precision
    )

    ##### Get NLL for BNN predict #####
    if posthoc_prior_opt:
        la.optimize_prior_precision(
            pred_type="nn",
            val_loader=val_loader,
            method="CV",  # "marglik"
            log_prior_prec_min=-6,
            log_prior_prec_max=2,
            grid_size=40,
        )
        bnn_pred_fn = la_pred(
            model=la,
            pred_type="nn",
            link_approx="mc",
            num_samples=num_samples,
            device=device,
        )
        bnn_metrics = compute_metrics(
            pred_fn=bnn_pred_fn,
            data_loader=test_loader,
            ood_loader=ood_loader,
            device=device,
        )
        table_logger.add_data(
            f"BNN {hessian_structure} GRID",
            metrics=bnn_metrics,
            prior_prec=la.prior_precision,
        )

    ##### Get NLL for GLM predict #####
    if posthoc_prior_opt:
        la.optimize_prior_precision(
            pred_type="glm",
            val_loader=val_loader,
            method="CV",  # "marglik"
            log_prior_prec_min=-6,
            log_prior_prec_max=2,
            grid_size=40,
        )
        glm_pred_fn = la_pred(
            model=la,
            pred_type="glm",
            link_approx="mc",
            num_samples=num_samples,
            device=device,
        )
        glm_metrics = compute_metrics(
            pred_fn=glm_pred_fn,
            data_loader=test_loader,
            ood_loader=ood_loader,
            device=device,
        )
        table_logger.add_data(
            f"GLM {hessian_structure} GRID",
            metrics=glm_metrics,
            prior_prec=la.prior_precision,
        )


if __name__ == "__main__":
    train_and_inference()  # pyright: ignore
