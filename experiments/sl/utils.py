#!/usr/bin/env python3
import os
import random
from typing import List, Optional, Union

import laplace
import numpy as np
import pandas as pd
import scipy.io as sio
import src
import torch
import torch.distributions as dists
import torch.nn as nn
import wandb
from experiments.sl.bnn_predictive.experiments.scripts.imgclassification import (
    get_dataset,
    get_model,
    QuickDS,
)
from experiments.sl.bnn_predictive.preds.datasets import UCIClassificationDatasets
from experiments.sl.bnn_predictive.preds.models import SiMLP
from laplace import BaseLaplace
from netcal.metrics import ECE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from src import SFR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import Subset


def set_seed_everywhere(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def checkpoint(
    sfr: src.SFR, optimizer: torch.optim.Optimizer, save_dir: str, verbose: bool = False
):
    if verbose:
        logger.info("Saving SFR and optimiser...")
    state = {"model": sfr.state_dict(), "optimizer": optimizer.state_dict()}
    fname = "best_ckpt_dict.pt"
    save_name = os.path.join(save_dir, fname)
    torch.save(state, save_name)
    if verbose:
        logger.info("Finished saving model and optimiser etc")
    return save_name


class EarlyStopper:
    def __init__(self, patience=1, min_prior_precision=0):
        self.patience = patience
        self.min_prior_precision = min_prior_precision
        self.counter = 0
        self.min_val_nll = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_val_nll:
            self.min_val_nll = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_val_nll + self.min_prior_precision):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def init_SFR_with_gaussian_prior(
    model: torch.nn.Module,
    prior_precision: float,
    likelihood: src.likelihoods.Likelihood,
    output_dim: int,
    num_inducing: int = 30,
    dual_batch_size: Optional[int] = None,
    jitter: float = 1e-6,
    device: str = "cpu",
) -> src.SFR:
    prior = src.priors.Gaussian(
        params=model.parameters, prior_precision=prior_precision
    )
    return src.SFR(
        network=model,
        prior=prior,
        likelihood=likelihood,
        output_dim=output_dim,
        num_inducing=num_inducing,
        dual_batch_size=dual_batch_size,
        jitter=jitter,
        device=device,
    )


def init_NN2GPSubset_with_gaussian_prior(
    model: torch.nn.Module,
    prior_precision: float,
    likelihood: src.likelihoods.Likelihood,
    output_dim: int,
    subset_size: int = 30,
    dual_batch_size: Optional[int] = None,
    jitter: float = 1e-6,
    device: str = "cpu",
) -> src.NN2GPSubset:
    prior = src.priors.Gaussian(
        params=model.parameters, prior_precision=prior_precision
    )
    return src.NN2GPSubset(
        network=model,
        prior=prior,
        likelihood=likelihood,
        output_dim=output_dim,
        subset_size=subset_size,
        dual_batch_size=dual_batch_size,
        jitter=jitter,
        device=device,
    )


def train_val_split(
    ds_train: Dataset,
    ds_test: Dataset,
    val_from_test: bool = True,
    val_split: float = 1 / 2,
):
    dataset = ds_test if val_from_test else ds_train
    len_ds = len(dataset)
    perm_ixs = torch.randperm(len_ds)
    val_ixs, ds_ixs = (
        perm_ixs[: int(len_ds * val_split)],
        perm_ixs[int(len_ds * val_split) :],
    )
    ds_val = Subset(dataset, val_ixs)
    if val_from_test:
        ds_test = Subset(dataset, ds_ixs)
    else:
        ds_train = Subset(dataset, ds_ixs)
    return ds_train, ds_val, ds_test


@torch.no_grad()
def compute_metrics(
    pred_fn,
    data_loader: DataLoader,
    ood_loader: DataLoader = None,
    device: str = "cpu",
    inference_strategy: str = "sfr",
) -> dict:
    py, targets, id_max_probs = [], [], []
    for _, (x, y) in enumerate(data_loader):
        p = pred_fn(x.to(device))
        py.append(p)
        targets.append(y.to(device))
        max_prob = torch.softmax(p, dim=-1)
        id_max_probs.append(torch.max(max_prob, axis=-1).values)

    targets = torch.cat(targets, dim=0).cpu().numpy()
    probs = torch.cat(py).cpu().numpy()

    if probs.shape[-1] == 1:
        bernoulli = True
    else:
        bernoulli = False

    if bernoulli:
        y_pred = probs >= 0.5
        acc = np.sum((y_pred[:, 0] == targets)) / len(probs)
    else:
        acc = (probs.argmax(-1) == targets).mean()
    ece = ECE(bins=15).measure(probs, targets)  # TODO does this work for bernoulli?

    if bernoulli:
        dist = dists.Bernoulli(torch.Tensor(probs[:, 0]))
    else:
        dist = dists.Categorical(torch.Tensor(probs))
    nll = -dist.log_prob(torch.Tensor(targets)).mean().numpy()

    metrics = {"acc": acc, "nll": nll, "ece": ece}

    # Compute ID/OOD entropies
    if ood_loader is not None:
        ood_probs, ood_targets, ood_max_probs = [], [], []
        for x, y in ood_loader:
            prob = pred_fn(x.to(device))
            ood_probs.append(prob)
            ood_targets.append(y)
            max_prob = torch.softmax(prob, dim=-1)
            ood_max_probs.append(torch.max(max_prob, axis=-1).values)

        ood_probs = torch.cat(ood_probs).cpu().numpy()
        ood_dist = torch.distributions.Categorical(torch.Tensor(ood_probs))
        id_entropies = dist.entropy().cpu().numpy()
        ood_entropies = ood_dist.entropy().cpu().numpy()

        ood_targets = torch.cat(ood_targets, dim=0).cpu().numpy()
        all_targets = np.concatenate([ood_targets, targets], 0)
        all_probs = np.concatenate([ood_probs, probs], 0)

        id_ood_labels = np.array([1] * len(probs) + [0] * len(ood_probs))
        # labels = np.array([0] * len(probs) + [1] * len(ood_probs))
        max_probs = torch.cat(id_max_probs + ood_max_probs).cpu().numpy()
        ood_auc = roc_auc_score(id_ood_labels, max_probs)

        metrics.update(
            {
                "id_entropies": id_entropies,
                "ood_entropies": ood_entropies,
                "ood_auc": ood_auc,
            }
        )
    else:
        metrics.update({"id_entropies": None, "ood_entropies": None, "ood_auc": None})

        # if wandb.run is not None:
        #     data_dict = {"id": pd.Series(id_entropies), "ood": pd.Series(ood_entropies)}
        #     table = wandb.Table(dataframe=pd.DataFrame(data_dict))
        #     wandb.log({f"{wandb_log_name}/entropies": table})

    return metrics


@torch.no_grad()
def compute_metrics_regression(
    model: Union[SFR, BaseLaplace, torch.nn.Module],
    data_loader: DataLoader,
    pred_type: str = "nn",
    device: str = "cpu",
    map: bool = False,
) -> dict:
    nlpd = []
    num_data = len(data_loader.dataset)
    mse = 0.0
    nlpd = 0.0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        input_dim = x.shape[-1]
        if not map:
            if isinstance(model, SFR):
                # print("Calculating SFR NLPD")
                y_mean, y_var = model(x.to(device), pred_type=pred_type)
                # y_var -= model.likelihood.sigma_noise**2
            elif isinstance(model, BaseLaplace):
                # print("Calculating LA NLPD")
                y_mean, f_var = model(x.to(device), pred_type=pred_type)
                y_var = f_var + model.sigma_noise**2
        else:
            # print("Calculating MAP NLPD")
            y_mean = model.network(x.to(device))
            y_var = torch.ones_like(y_mean) * model.likelihood.sigma_noise**2
            # TODO should this be ones???

        # y_mean = y_mean.detach().cpu()
        # y_std = y_var.sqrt()
        # y_std = y_std.detach().cpu()
        y_mean = y_mean
        y_std = y_var.sqrt()
        if y.ndim == 1:
            y = torch.unsqueeze(y, -1)
        mse += torch.nn.MSELoss(reduction="sum")(y_mean, y)
        # print(f"mse {mse.shape}")
        # log_prob = -torch.distributions.Normal(loc=y_mean, scale=y_std).log_prob(y)
        # print(f"log_prob {log_prob.shape}")
        # log_prob = torch.mean(
        #     -torch.distributions.Normal(loc=y_mean, scale=y_std).log_prob(y), -1
        # )
        # print(f"log_prob prod {log_prob.shape}")
        # print(f"y_mean {y_mean.shape}")
        # print(f"y_std {y_std.shape}")
        # print(f"y {y.shape}")
        # nlpd.append(
        #     torch.sum(  # TODO should this be sum?
        #         -torch.distributions.Normal(
        #             loc=torch.zeros_like(y_mean), scale=y_std
        #         ).log_prob(y_mean - y),
        #         -1
        #         # -torch.distributions.Normal(loc=y_mean, scale=y_std).log_prob(y), -1
        #     )
        # )
        nlpd -= torch.sum(
            torch.distributions.Normal(
                loc=torch.zeros_like(y_mean), scale=y_std
            ).log_prob(y_mean - y)
        )
        # nlpd += torch.distributions.Normal(
        #     loc=torch.zeros_like(y_mean), scale=y_std
        # ).log_prob(y_mean - y)
        # print(f"nlpd {nlpd.shape}")

    # nlpd = torch.concat(nlpd, 0)
    # print(f"nlpd {nlpd.shape}")
    # nlpd = torch.mean(nlpd, 0)
    nlpd = nlpd / (num_data * input_dim)
    # print(f"nlpd {nlpd.shape}")
    # print(f"mse {len(mse)}")
    # mse = torch.stack(mse, 0)
    mse = mse / (num_data * input_dim)
    # print(f"mse {mse.shape}")
    # mse = torch.mean(mse, 0)
    # print(f"mse {mse.shape}")

    metrics = {"mse": mse.cpu().numpy().item(), "nll": nlpd.cpu().numpy().item()}
    return metrics


def get_image_dataset(
    name: str,
    double: bool,
    dir: str,
    device: str,
    debug: bool,
    val_from_test: bool,
    val_split: float,
    train_update_split: Optional[float] = None,
):
    ds_train, ds_test = get_dataset(
        dataset=name, double=double, dir=dir, device=None, debug=debug
    )
    if debug:
        ds_train.data = ds_train.data[:500]
        ds_train.targets = ds_train.targets[:500]
        ds_test.data = ds_test.data[:500]
        ds_test.targets = ds_test.targets[:500]
    if double:
        print("MAKING DATASET DOUBLE")
        ds_train.data = ds_train.data.to(torch.double)
        ds_test.data = ds_test.data.to(torch.double)
        ds_train.targets = ds_train.targets.long()
        ds_test.targets = ds_test.targets.long()
    # if device is not None:
    #     ds_train.data = ds_train.data.to(device)
    #     ds_test.data = ds_test.data.to(device)
    #     ds_train.targets = ds_train.targets.to(device)
    #     ds_test.targets = ds_test.targets.to(device)
    output_dim = ds_train.K  # set network output dim
    pixels = ds_train.pixels
    channels = ds_train.channels
    ds_train = QuickDS(ds_train, device)
    # ds_val = QuickDS(ds_val, device)
    ds_test = QuickDS(ds_test, device)
    # Split train data set into train and validation
    print("Original num train {}".format(len(ds_train)))
    print("Original num test {}".format(len(ds_test)))
    ds_train, ds_val, ds_test = train_val_split(
        ds_train=ds_train,
        ds_test=ds_test,
        val_from_test=val_from_test,
        val_split=val_split,
    )
    ds_train.K = output_dim
    ds_train.output_dim = output_dim
    ds_train.pixels = pixels
    ds_train.channels = channels
    print("Final num train {}".format(len(ds_train)))
    print("Final num val {}".format(len(ds_val)))
    print("Final num test {}".format(len(ds_test)))
    ds_update = None  # TODO implement this properly
    return ds_train, ds_val, ds_test, ds_update


def get_uci_dataset(
    name: str,
    random_seed: int,
    dir: str,
    double: bool,
    train_update_split: Optional[float] = None,
    **kwargs,
):
    ds_train = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=True,
        double=double,
    )
    if train_update_split:
        output_dim = ds_train.C  # set network output dim
        ds_train, ds_update, _ = train_val_split(
            ds_train=ds_train,
            ds_test=None,
            val_from_test=False,
            val_split=train_update_split,
        )
        ds_train.C = output_dim
        ds_train.output_dim = output_dim
        ds_update.C = output_dim
        ds_update.output_dim = output_dim
    else:
        ds_update = None

    ds_test = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=False,
        valid=False,
        double=double,
    )
    ds_val = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=False,
        valid=True,
        double=double,
    )
    print(f"dataset={name}")
    if ds_train.C > 2:  # set network output dim
        output_dim = ds_train.C
    else:
        output_dim = 1
    if double:
        try:
            ds_train.data = ds_train.data.to(torch.double)
            ds_train.targets = ds_train.targets.long()
        except:
            ds_train.dataset.data = ds_train.dataset.data.to(torch.double)
            ds_train.dataset.targets = ds_train.dataset.targets.to(torch.double)
            # ds_val.dataset.data = ds_val.dataset.data.to(torch.double)
        ds_val.data = ds_val.data.to(torch.double)
        ds_val.targets = ds_val.targets.to(torch.double)
        ds_test.data = ds_test.data.to(torch.double)
        ds_val.targets = ds_val.targets.long()
        ds_test.targets = ds_test.targets.long()
        if train_update_split:
            ds_update.dataset.data = ds_update.dataset.data.to(torch.double)
            ds_update.dataset.targets = ds_update.dataset.targets.long()

    # always use Softmax instead of Bernoulli
    output_dim = ds_train.C
    if name in ["australian", "breast_cancer", "ionosphere"]:
        # ds_train.targets = ds_train.targets.to(torch.double)
        # ds_val.targets = ds_val.targets.to(torch.double)
        # ds_test.targets = ds_test.targets.to(torch.double)
        try:
            ds_train.targets = ds_train.targets.long()
        except:
            ds_train.dataset.targets = ds_train.dataset.targets.long()
        ds_val.targets = ds_val.targets.long()
        ds_test.targets = ds_test.targets.long()

    print(f"output_dim={output_dim}")
    print(f"ds_train.C={ds_train.C}")
    # ds_train.K = output_dim
    ds_train.output_dim = output_dim
    return ds_train, ds_val, ds_test, ds_update


def get_image_network(name: str, ds_train, device: str):
    network = get_model(model_name=name, ds_train=ds_train).to(device)
    return network


def get_uci_network(name, output_dim, ds_train, device: str):
    try:
        input_size = ds_train.data.shape[1]
    except:
        try:
            input_size = ds_train.dataset.data.shape[1]
        except:
            input_size = ds_train[0][0].shape[0]
    network = SiMLP(
        input_size=input_size,
        output_size=output_dim,
        n_layers=2,
        n_units=50,
        activation="tanh",
    ).to(device)
    return network


def get_large_uci_network(name, output_dim, ds_train, device: str):
    try:
        input_size = ds_train.data.shape[1]
    except:
        try:
            input_size = ds_train.dataset.data.shape[1]
        except:
            input_size = ds_train[0][0].shape[0]
    network = SiMLP(
        input_size=input_size,
        output_size=output_dim,
        n_layers=3,
        n_units=512,
        activation="tanh",
    ).to(device)
    # network.apply(orthogonal_init)
    return network


def get_boston_network(name, output_dim, ds_train, device: str):
    try:
        input_size = ds_train.data.shape[1]
    except:
        try:
            input_size = ds_train.dataset.data.shape[1]
        except:
            input_size = ds_train[0][0].shape[0]
    network = SiMLP(
        input_size=input_size,
        output_size=output_dim,
        n_layers=2,
        n_units=128,
        activation="tanh",
    ).to(device)
    # network.apply(orthogonal_init)
    return network


class Sin(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


def get_stationary_mlp(
    ds_train, output_dim: int, hidden_size: int = 50, device: str = "cpu"
):
    try:
        input_size = ds_train.data.shape[1]
    except:
        # input_size = ds_train.dataset.data.shape[1]
        try:
            input_size = ds_train.dataset.data.shape[1]
        except:
            input_size = ds_train[0][0].shape[0]
    network = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, 16),
        Sin(),
        # torch.nn.Tanh(),
        # torch.nn.Tanh(),
        torch.nn.Linear(16, output_dim),
    )
    return network.to(device)


class UCIDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, device: str = "cpu", name: str = "boston"):
        self.name = name
        self.device = device

        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def get_UCIreg_dataset(
    name: str,  # ["boston", "concrete", "airfoil", "elevators"]
    random_seed: int,
    double: bool = False,
    data_split: Optional[list] = [70, 15, 15, 0],
    order_dim: Optional[int] = None,  # if int order along X[:, order_dim]
    normalize=True,
    device: str = "cpu",
    **kwargs,
):
    file_path = os.path.dirname(os.path.realpath(__file__))

    full_path = os.path.join(file_path, "data/uci_regression/" + name)
    X, Y = load_UCIreg_dataset(full_path=full_path, name=name, normalize=normalize)

    ds = UCIDataset(data=X, targets=Y)

    data_split_1 = [data_split[0] + data_split[1], data_split[2] + data_split[3]]
    # Order data set along input dimension
    if order_dim is not None:
        idxs = np.argsort(ds.data, 0)[:, 0]
        ds.data = ds.data[idxs]  # order inputs
        ds.targets = ds.targets[idxs]  # order outputs
        split_idx = round(data_split_1[0] / 100 * len(idxs))
        ds_train = UCIDataset(
            data=ds.data[0:split_idx], targets=ds.targets[0:split_idx]
        )
        ds_new = UCIDataset(
            data=ds.data[split_idx:-1], targets=ds.targets[split_idx:-1]
        )
    else:
        # print(f"data_split_1 {data_split_1}")
        ds_train, ds_new = split_dataset(
            dataset=ds, random_seed=random_seed, data_split=data_split_1
        )

    # print(f"data_split[0:2] {data_split[0:2]}")
    ds_train, ds_val = split_dataset(
        dataset=ds_train,
        random_seed=random_seed,
        # double=double,
        data_split=data_split[0:2],
    )
    # print(f"data_split[2:] {data_split[2:]}")
    ds_test, ds_update = split_dataset(
        dataset=ds_new,
        random_seed=random_seed,
        # double=double,
        data_split=data_split[2:],
    )
    # print(f"data_split[0:2] {data_split[0:2]}")
    # # ds_train, ds_val, ds_update_1 = split_dataset(
    # ds_train, ds_val, ds_test_1 = split_dataset(
    #     dataset=ds_train,
    #     random_seed=random_seed,
    #     double=double,
    #     data_split=data_split[0:2] + [data_split[2] / 2],
    # )
    # print(f"data_split[2:] {data_split[2:]}")
    # # ds_test, ds_update_2 = split_dataset(
    # ds_test_2, ds_update = split_dataset(
    #     dataset=ds_new,
    #     random_seed=random_seed,
    #     double=double,
    #     # data_split=data_split[2:],
    #     data_split=[data_split[2] / 2] + data_split[3:4],
    #     # data_split=data_split[2:3] + [data_split[2] / 2],
    # )
    # ds_test = ConcatDataset([ds_test_1, ds_test_2])
    # # ds_update = ConcatDataset([ds_update_1, ds_update_2])

    if double:
        ds_train.data.to(torch.double)
        ds_val.data.to(torch.double)
        ds_test.data.to(torch.double)
        ds_update.data.to(torch.double)
        ds_train.targets.to(torch.double)
        ds_val.targets.to(torch.double)
        ds_test.targets.to(torch.double)
        ds_update.targets.to(torch.double)
        # Y = torch.tensor(Y, dtype=torch.double)
    else:
        ds_train.data.to(torch.float)
        ds_val.data.to(torch.float)
        ds_test.data.to(torch.float)
        ds_update.data.to(torch.float)
        ds_train.targets.to(torch.float)
        ds_val.targets.to(torch.float)
        ds_test.targets.to(torch.float)
        ds_update.targets.to(torch.float)

    output_dim = 1
    # ds_train, ds_val, ds_test, ds_update = split_dataset(
    #     dataset=ds, random_seed=random_seed, double=double, data_split=data_split
    # )
    ds_train.output_dim = output_dim
    # breakpoint()
    return ds_train, ds_val, ds_test, ds_update


def split_dataset(
    dataset: torch.utils.data.Dataset,
    random_seed: int,
    # double: bool = False,
    data_split: Optional[list] = [70, 30],
    device: str = "cpu",
):
    if random_seed:
        random.seed(random_seed)
    num_data = len(dataset)
    # print(f"num_data {num_data}")
    idxs = np.random.permutation(num_data)
    datasets = []
    idx_start = 0
    for split in data_split:
        idx_end = idx_start + int(num_data * split / 100)
        idxs_ = idxs[idx_start:idx_end]
        # print(f"idxs_ {np.sort(idxs_)}")
        if isinstance(dataset.data, torch.Tensor):
            X = dataset.data[idxs_]
            y = dataset.targets[idxs_]
        else:
            X = torch.from_numpy(dataset.data[idxs_])
            y = torch.from_numpy(dataset.targets[idxs_])
        # if double:
        #     X = X.to(torch.double)
        #     y = y.to(torch.double)
        # else:
        #     X = X.to(torch.float)
        #     y = y.to(torch.float)
        X = X.to(device)
        y = y.to(device)
        ds = torch.utils.data.TensorDataset(X, y)
        ds.data = X
        ds.targets = y
        datasets.append(ds)
        idx_start = idx_end
    return datasets


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # elif isinstance(m, EnsembleLinear):
    #     for w in m.weight.data:
    #         nn.init.orthogonal_(w)
    #     if m.bias is not None:
    #         for b in m.bias.data:
    #             nn.init.zeros_(b)
    elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        # nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# UCI regression dataset loading function
def load_UCIreg_dataset(full_path, name, normalize=True):
    # load the dataset as a numpy array
    if name == "boston":
        df = pd.read_csv(r"" + full_path + ".csv")
        Xo = df[
            [
                "crim",
                "zn",
                "indus",
                "chas",
                "nox",
                "rm",
                "age",
                "dis",
                "tax",
                "ptratio",
                "black",
                "lstat",
            ]
        ].to_numpy()
        Yo = df["medv"].to_numpy().reshape((-1, 1))
    elif name == "concrete":
        df = pd.read_csv(r"" + full_path + ".csv")
        Xo = df[["cement", "water", "coarse_agg", "fine_agg", "age"]].to_numpy()
        Yo = df["compressive_strength"].to_numpy().reshape((-1, 1))
    elif name == "airfoil":
        df = pd.read_csv(r"" + full_path + ".csv")
        Xo = df[
            [
                "Frequency",
                "AngleAttack",
                "ChordLength",
                "FreeStreamVelocity",
                "SuctionSide",
            ]
        ].to_numpy()
        Yo = df["Sound"].to_numpy().reshape((-1, 1))
    elif name == "elevators":
        # Load all the data

        data = np.array(loadmat(full_path + ".mat")["data"])
        Xo = data[:, :-1]
        Yo = data[:, -1].reshape(-1, 1)

    if normalize == True:
        X_scaler = StandardScaler().fit(Xo)
        Y_scaler = StandardScaler().fit(Yo)
        Xo = X_scaler.transform(Xo)
        Yo = Y_scaler.transform(Yo)

    return Xo, Yo


def sfr_pred(
    model: src.SFR,
    pred_type: str = "gp",  # "gp" or "nn"
    num_samples: int = 100,
    device: str = "cuda",
):
    @torch.no_grad()
    def pred_fn(x):
        return model(x.to(device), pred_type=pred_type, num_samples=num_samples)[0]

    return pred_fn


def la_pred(
    model: laplace.BaseLaplace,
    pred_type: str = "glm",  # "glm" or "nn"
    link_approx: str = "probit",  # 'mc', 'probit', 'bridge', 'bridge_norm'
    num_samples: int = 100,  # num_samples for link_approx="mc"
    device: str = "cuda",
):
    @torch.no_grad()
    def pred_fn(x):
        return model(
            x.to(device),
            pred_type=pred_type,
            link_approx=link_approx,
            n_samples=num_samples,
        )

    return pred_fn


def get_airline_dataset(
    random_seed: int,
    double: bool = False,
    data_split: Optional[list] = [70, 15, 15, 0],
    order_dim: Optional[int] = None,  # if int order along X[:, order_dim]
    normalize=True,
    device: str = "cpu",
    **kwargs,
):
    # Import the data
    file_path = os.path.dirname(os.path.realpath(__file__))
    # print(f"file_path {file_path}")
    # full_path = os.path.join(file_path, "data/airline.pickle")
    full_path = os.path.join(file_path, "data/airline.mat")
    mat_contents = sio.loadmat(full_path)
    X = mat_contents["X"]
    Y = mat_contents["Y"].T
    # print(f"full_path {full_path}")

    # Convert time of day from hhmm to minutes since midnight
    # data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(data.ArrTime, 100)
    # data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(data.DepTime, 100)

    # Normalize Y scale and offset
    if normalize == True:
        X_scaler = StandardScaler().fit(X)
        Y_scaler = StandardScaler().fit(Y)
        X = X_scaler.transform(X)
        Y = Y_scaler.transform(Y)

    data_split_1 = [data_split[0] + data_split[1], data_split[2] + data_split[3]]
    # Order data set along input dimension
    if order_dim is not None:
        idxs = np.argsort(X, 0)[:, 0]
        X = X[idxs]  # order inputs
        Y = Y[idxs]  # order outputs
        split_idx = round(data_split_1[0] / 100 * len(idxs))
        ds_train = UCIDataset(data=X[0:split_idx], targets=Y[0:split_idx])
        ds_new = UCIDataset(data=X[split_idx:-1], targets=Y[split_idx:-1])
    else:
        # print(f"data_split_1 {data_split_1}")
        ds_train, ds_new = split_dataset(
            dataset=ds, random_seed=random_seed, data_split=data_split_1
        )

    # print(f"data_split[0:2] {data_split[0:2]}")
    ds_train, ds_val = split_dataset(
        dataset=ds_train,
        random_seed=random_seed,
        # double=double,
        data_split=data_split[0:2],
    )
    # print(f"data_split[2:] {data_split[2:]}")
    ds_test, ds_update = split_dataset(
        dataset=ds_new,
        random_seed=random_seed,
        # double=double,
        data_split=data_split[2:],
    )
    if double:
        ds_train.data.to(torch.double)
        ds_val.data.to(torch.double)
        ds_test.data.to(torch.double)
        ds_update.data.to(torch.double)
        ds_train.targets.to(torch.double)
        ds_val.targets.to(torch.double)
        ds_test.targets.to(torch.double)
        ds_update.targets.to(torch.double)
        # Y = torch.tensor(Y, dtype=torch.double)
    else:
        ds_train.data.to(torch.float)
        ds_val.data.to(torch.float)
        ds_test.data.to(torch.float)
        ds_update.data.to(torch.float)
        ds_train.targets.to(torch.float)
        ds_val.targets.to(torch.float)
        ds_test.targets.to(torch.float)
        ds_update.targets.to(torch.float)
        # Y = torch.tensor(Y, dtype=torch.float)
    # print(f"data_split[0:2] {data_split[0:2]}")
    # breakpoint()
    output_dim = 1
    ds_train.output_dim = output_dim
    return ds_train, ds_val, ds_test, ds_update


def get_uci_dataset_for_fast_updates(
    name: str,
    random_seed: int,
    double: bool,
    dir: str,
    data_split: List[float] = [35, 15, 15, 35],
    order_dim: Optional[int] = None,
    **kwargs,
):
    ds_train = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=True,
        double=double,
    )
    ds_test = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=False,
        valid=False,
        double=double,
    )
    ds_val = UCIClassificationDatasets(
        name,
        random_seed=random_seed,
        root=dir,
        stratify=True,
        train=False,
        valid=True,
        double=double,
    )
    output_dim = ds_train.C
    # print(f"ds_train {len(ds_train)}")
    # print(f"ds_test {len(ds_test)}")
    # print(f"ds_val {len(ds_val)}")
    ds_1 = ConcatDataset([ds_train, ds_test])
    # print(f"ds_1 {len(ds_1)}")
    ds = ConcatDataset([ds_1, ds_val])
    # print(f"ds {len(ds)}")
    loader = DataLoader(ds, batch_size=len(ds))
    data = next(iter(loader))
    X, Y = data
    for i in range(X.shape[-1]):
        print(f"dim {i} unique: {torch.unique(X[:, i]).shape}")
    # breakpoint()

    if double:
        X = torch.tensor(X, dtype=torch.double)
        Y = torch.tensor(Y, dtype=torch.long)
    else:
        X = torch.tensor(X, dtype=torch.float)
        # Y = torch.tensor(Y, dtype=torch.float)

    data_split_1 = [data_split[0] + data_split[1], data_split[2] + data_split[3]]
    # Order data set along input dimension
    if order_dim is not None:
        idxs = np.argsort(X, 0)[:, 0]
        X = X[idxs]  # order inputs
        Y = Y[idxs]  # order outputs
        split_idx = round(data_split_1[0] / 100 * len(idxs))
        ds_train = UCIDataset(data=X[0:split_idx], targets=Y[0:split_idx])
        ds_new = UCIDataset(data=X[split_idx:-1], targets=Y[split_idx:-1])
    else:
        # print(f"data_split_1 {data_split_1}")
        ds_train, ds_new = split_dataset(
            dataset=ds, random_seed=random_seed, double=double, data_split=data_split_1
        )

    # print(f"data_split[0:2] {data_split[0:2]}")
    ds_train, ds_val = split_dataset(
        dataset=ds_train,
        random_seed=random_seed,
        # double=double,
        data_split=data_split[0:2],
    )
    # print(f"data_split[2:] {data_split[2:]}")
    ds_test, ds_update = split_dataset(
        dataset=ds_new,
        random_seed=random_seed,
        # double=double,
        data_split=data_split[2:],
    )
    if double:
        ds_train.data.to(torch.double)
        ds_val.data.to(torch.double)
        ds_test.data.to(torch.double)
        ds_update.data.to(torch.double)
    # print(f"data_split[0:2] {data_split[0:2]}")
    # breakpoint()
    ds_train.output_dim = output_dim
    return ds_train, ds_val, ds_test, ds_update


def get_uci_dataset_from_repo(
    name: str,
    random_seed: int,
    double: bool,
    data_split: List[float] = [35, 15, 15, 35],
    order_dim: Optional[int] = None,
    **kwargs,
):
    from uci_datasets import Dataset

    ds = Dataset(name)
    X = ds.x
    Y = ds.y
    for i in range(X.shape[-1]):
        print(f"dim {i} unique: {np.unique(X[:, i]).shape}")
    # breakpoint()

    data_split_1 = [data_split[0] + data_split[1], data_split[2] + data_split[3]]
    # Order data set along input dimension
    if order_dim is not None:
        idxs = np.argsort(X, 0)[:, 0]
        X = X[idxs]  # order inputs
        Y = Y[idxs]  # order outputs
        split_idx = round(data_split_1[0] / 100 * len(idxs))
        ds_train = UCIDataset(data=X[0:split_idx], targets=Y[0:split_idx])
        ds_new = UCIDataset(data=X[split_idx:-1], targets=Y[split_idx:-1])
    else:
        # print(f"data_split_1 {data_split_1}")
        ds_train, ds_new = split_dataset(
            dataset=ds, random_seed=random_seed, double=double, data_split=data_split_1
        )

    # print(f"data_split[0:2] {data_split[0:2]}")
    ds_train, ds_val = split_dataset(
        dataset=ds_train,
        random_seed=random_seed,
        # double=double,
        data_split=data_split[0:2],
    )
    # print(f"data_split[2:] {data_split[2:]}")
    ds_test, ds_update = split_dataset(
        dataset=ds_new,
        random_seed=random_seed,
        # double=double,
        data_split=data_split[2:],
    )
    if double:
        ds_train.data.to(torch.double)
        ds_val.data.to(torch.double)
        ds_test.data.to(torch.double)
        ds_update.data.to(torch.double)
    # print(f"data_split[0:2] {data_split[0:2]}")

    # breakpoint()
    ds_train.output_dim = Y.shape[-1]
    return ds_train, ds_val, ds_test, ds_update
