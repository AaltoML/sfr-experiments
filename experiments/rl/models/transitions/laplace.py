#!/usr/bin/env python3
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import laplace
import numpy as np
import src
import torch
import wandb
from experiments.rl.custom_types import Action, State, StatePrediction
from experiments.rl.utils import EarlyStopper
from experiments.rl.utils.buffer import ReplayBuffer
from torch.utils.data import DataLoader, TensorDataset

from .base import TransitionModel


class LaplaceTransitionModel(TransitionModel):
    def __init__(
        self,
        network: torch.nn.Module,
        state_dim: int,
        learning_rate: float = 1e-2,
        num_iterations: int = 1000,
        batch_size: int = 64,
        # num_workers: int = 1,
        prior_precision: float = 0.0001,  # weight decay
        sigma_noise: float = 1.0,
        wandb_loss_name: str = "Transition model loss",
        early_stopper: EarlyStopper = None,
        device: str = "cuda",
        prediction_type: str = "LA",  # "LA" or "NN" or TODO
        logging_freq: int = 500,
        hessian_structure: str = "full",
        subset_of_weights: str = "all",
        backend=laplace.curvature.BackPackGGN,
        train_sigma_noise: bool = False,
        separate_sigma_noise: bool = False,
    ):
        network.to(device)

        self.network = network
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        # self.num_inducing = num_inducing
        self.prior_precision = prior_precision
        self.sigma_noise = sigma_noise
        self.wandb_loss_name = wandb_loss_name
        self.early_stopper = early_stopper
        self.device = device
        self.prediction_type = prediction_type
        self.logging_freq = logging_freq
        self.train_sigma_noise = train_sigma_noise
        self.separate_sigma_noise = separate_sigma_noise

        self.subset_of_weights = subset_of_weights
        self.hessian_structure = hessian_structure

        # Build SFR to get loss fn
        if separate_sigma_noise:
            raise NotImplementedError(
                "Laplace redux library can't handle non-scalar sigma_noise..."
            )
            log_sigma_noise = torch.tensor([np.log(sigma_noise)] * state_dim).to(device)
        else:
            log_sigma_noise = torch.tensor([np.log(sigma_noise)]).to(device)
        if self.train_sigma_noise:
            log_sigma_noise.requires_grad = True
        likelihood = src.likelihoods.Gaussian(log_sigma_noise=log_sigma_noise)

        prior = src.priors.Gaussian(
            params=network.parameters, prior_precision=prior_precision
        )
        self.sfr = src.SFR(
            network=network,
            prior=prior,
            likelihood=likelihood,
            output_dim=state_dim,
            num_inducing=None,
            # dual_batch_size=dual_batch_size,
            # dual_batch_size=None,
            # jitter=jitter,
            device=device,
        )

        self.la = laplace.Laplace(
            self.network,
            "regression",
            sigma_noise=sigma_noise,
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
            prior_precision=prior_precision,
            backend=backend,
        )

    @torch.no_grad()
    def predict(self, state: State, action: Action) -> StatePrediction:
        state_action_input = torch.concat([state, action], -1)
        if "NN" in self.prediction_type:
            delta_state_mean = self.network.forward(state_action_input)
            delta_state_var = torch.zeros_like(delta_state_mean)
        elif "LA" in self.prediction_type:
            delta_state_mean, delta_state_var = self.la(state_action_input)
            delta_state_var = torch.diagonal(delta_state_var, dim1=-1, dim2=-2)
        else:
            raise NotImplementedError("prediction_type should be one of LA or NN")
        # delta_state_mean = sfr.predict_mean(state_action_input)
        # delta_state_mean, delta_state_var, noise_var = svgp_predict_fn(
        return StatePrediction(
            state_mean=state + delta_state_mean,
            state_var=delta_state_var,
            noise_var=self.sfr.likelihood.sigma_noise**2,
        )

    def train(self, replay_buffer: ReplayBuffer):
        if self.early_stopper is not None:
            self.early_stopper.reset()
        # self.network.apply(weights_init_normal)
        self.network.train()
        params = [{"params": self.sfr.parameters()}]
        if self.train_sigma_noise:
            params.append({"params": self.sfr.likelihood.log_sigma_noise})
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        for i in range(self.num_iterations):
            samples = replay_buffer.sample(batch_size=self.batch_size)
            state_action_inputs = torch.concat(
                [samples["state"], samples["action"]], -1
            )
            state_diff = samples["next_state"] - samples["state"]

            # pred = network(state_action_inputs)
            # loss = loss_fn(pred, state_diff)
            loss = self.sfr.loss(x=state_action_inputs, y=state_diff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"sigma_noise {self.sfr.likelihood.log_sigma_noise}")
            if self.wandb_loss_name is not None:
                if wandb.run is not None:
                    wandb.log({self.wandb_loss_name: loss})

            if i % self.logging_freq == 0:
                logger.info("Iteration : {} | Loss: {}".format(i, loss))
            if self.early_stopper is not None:
                stop_flag = self.early_stopper(loss)
                if stop_flag:
                    logger.info("Early stopping criteria met, stopping training")
                    logger.info("Breaking out loop")
                    break

        data = replay_buffer.sample(batch_size=len(replay_buffer))
        state_action_inputs_all = torch.concat([data["state"], data["action"]], -1)
        state_diff_all = data["next_state"] - data["state"]

        train_loader = DataLoader(
            TensorDataset(*(state_action_inputs_all, state_diff_all)),
            batch_size=self.batch_size,
            # shuffle=False,
        )
        print("made train_loader {}".format(train_loader))
        # self.sfr.set_data((state_action_inputs_all, state_diff_all))
        # la = laplace.Laplace(
        #     self.network,
        #     "regression",
        #     subset_of_weights=self.subset_of_weights,
        #     hessian_structure=self.hessian_structure,
        # )
        self.la._sigma_noise = self.sfr.likelihood.sigma_noise
        self.la.fit(train_loader)

    def update(self, data_new):
        pass
        # return self.sfr.update(x=data_new[0], y=data_new[1])
