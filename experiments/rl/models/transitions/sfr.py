#!/usr/bin/env python3
import logging
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import src
import torch
import wandb
from experiments.rl.custom_types import Action, State, StatePrediction
from experiments.rl.utils import EarlyStopper
from experiments.rl.utils.buffer import ReplayBuffer

from .base import TransitionModel


class SFRTransitionModel(TransitionModel):
    def __init__(
        self,
        network: torch.nn.Module,
        state_dim: int,
        learning_rate: float = 1e-2,
        num_iterations: int = 1000,
        batch_size: int = 64,
        # num_workers: int = 1,
        num_inducing: int = 100,
        dual_batch_size: Optional[int] = None,
        prior_precision: float = 0.0001,  # weight decay
        sigma_noise: float = 1.0,
        jitter: float = 1e-4,
        wandb_loss_name: str = "Transition model loss",
        early_stopper: EarlyStopper = None,
        device: str = "cuda",
        prediction_type: str = "SVGPMeanOnly",  # "SVGPMeanOnly" or "SVGP" or "NN"
        logging_freq: int = 500,
        train_sigma_noise: bool = False,
        separate_sigma_noise: bool = False,
    ):
        network.to(device)

        self.network = network
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.num_inducing = num_inducing
        self.prior_precision = prior_precision
        self.sigma_noise = sigma_noise
        self.wandb_loss_name = wandb_loss_name
        self.early_stopper = early_stopper
        self.device = device
        self.prediction_type = prediction_type
        self.logging_freq = logging_freq
        self.train_sigma_noise = train_sigma_noise
        self.separate_sigma_noise = separate_sigma_noise

        if separate_sigma_noise:
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
            num_inducing=num_inducing,
            dual_batch_size=dual_batch_size,
            # dual_batch_size=None,
            jitter=jitter,
            device=device,
        )

        self.sfr.to(device)
        logger.info(f"Transition SFR on {self.sfr.device}")

    @torch.no_grad()
    def predict(self, state: State, action: Action) -> StatePrediction:
        state_action_input = torch.concat([state, action], -1)
        if "NN-only" in self.prediction_type:
            delta_state_mean = self.network.forward(state_action_input)
            delta_state_var = torch.zeros_like(delta_state_mean)
        elif "GP-mean" in self.prediction_type:
            delta_state_mean, delta_state_var = self.sfr.predict_f(
                state_action_input, device="gpu"
            )
        elif "NN-mean" in self.prediction_type:
            delta_state_mean, delta_state_var = self.sfr.predict_f(
                state_action_input, device="gpu"
            )
        # elif "GPMeanOnly" in self.prediction_type:
        #     delta_state_mean = self.sfr.predict_mean(state_action_input)
        #     delta_state_var = torch.zeros_like(delta_state_mean)
        else:
            raise NotImplementedError(
                "prediction_type should be one of SVGP, SVGPMeanOnly or NN"
            )
        return StatePrediction(
            state_mean=state + delta_state_mean,
            state_var=delta_state_var,
            noise_var=self.sfr.likelihood.sigma_noise**2,
        )

    def train(self, replay_buffer: ReplayBuffer):
        if self.early_stopper is not None:
            self.early_stopper.reset()
        # self.network.apply(weights_init_normal)
        self.sfr.train()
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
        self.sfr.set_data((state_action_inputs_all, state_diff_all))
        self.sfr.to_device(self.device)

    def update(self, data_new):
        return self.sfr.update(x=data_new[0], y=data_new[1])
