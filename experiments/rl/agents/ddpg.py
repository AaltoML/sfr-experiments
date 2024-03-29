#!/usr/bin/env python3
import logging
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import experiments.rl.agents.utils as util
import torch
import torch.nn as nn
import wandb
from experiments.rl.custom_types import Action, EvalMode, State, T0
from experiments.rl.utils.buffer import ReplayBuffer

from .agent import Agent


class Actor(nn.Module):
    def __init__(self, state_dim: int, mlp_dims: List[int], action_dim: int):
        super().__init__()
        self._actor = util.mlp(state_dim, mlp_dims, action_dim)
        self.apply(util.orthogonal_init)

    def forward(self, state: State, std):
        mu = self._actor(state)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return util.TruncatedNormal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int, mlp_dims: List[int], action_dim: int):
        super().__init__()
        self._critic1 = util.mlp(state_dim + action_dim, mlp_dims, 1)
        self._critic2 = util.mlp(state_dim + action_dim, mlp_dims, 1)
        self.apply(util.orthogonal_init)

    def forward(self, state: State, action: Action):
        state_action_input = torch.cat([state, action], dim=-1)
        return self._critic1(state_action_input), self._critic2(state_action_input)


class DDPGAgent(Agent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        mlp_dims: List[int] = [512, 512],
        learning_rate: float = 3e-4,
        max_ddpg_iterations: int = 100,  # for training DDPG
        # std_schedule: str = "linear(1.0, 0.1, 50)",
        std: float = 0.1,  # TODO make this schedule
        std_clip: float = 0.3,
        nstep: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
    ):
        self.max_ddpg_iterations = max_ddpg_iterations
        self.std = std
        self.std_clip = std_clip
        self.nstep = nstep
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(state_dim, mlp_dims, action_dim).to(device)
        self.critic = Critic(
            state_dim=state_dim, mlp_dims=mlp_dims, action_dim=action_dim
        ).to(device)
        self.critic_target = Critic(
            state_dim=state_dim, mlp_dims=mlp_dims, action_dim=action_dim
        ).to(device)
        print("Critic on device: {}".format(device))

        # Init optimizer
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self._update_q_fn = update_q(
            optim=self.optim_critic,
            actor=self.actor,
            critic=self.critic,
            critic_target=self.critic_target,
            std_clip=std_clip,
        )
        self._update_actor_fn = update_actor(
            optim=self.optim_actor,
            actor=self.actor,
            critic=self.critic,
            std_clip=std_clip,
        )

    def train(self, replay_buffer: ReplayBuffer) -> dict:
        info = {"std": self.std}
        for i in range(self.max_ddpg_iterations):
            # std = linear_schedule(std_schedule, i)  # linearly udpate std
            # info = {"std": std}
            # std = 0.1
            # info = {"std": std}

            # std_schedule, (num_iterations - 10) * episode_idx + i

            samples = replay_buffer.sample()
            state = samples["state"]  # [B, state_dim]
            action = samples["action"]  # [B, action_dim]
            # reward = samples["reward"][..., None]  # needs to be [B, 1]
            reward = samples["reward"]  # needs to be [B, 1]
            # print("REWARD {}".format(reward.shape))
            next_state = samples["next_state"][None, ...]  # [1, B, state_dim]

            info.update(
                self._update_q_fn(
                    state=state,
                    action=action,
                    reward=reward,
                    discount=1,
                    next_state=next_state,
                    std=self.std,
                )
            )
            info.update(self._update_actor_fn(state=state, std=self.std))

            # Update target network
            util.soft_update_params(self.critic, self.critic_target, tau=self.tau)
            # if i % 10 == 0:
            #     print(
            #         "DDPG: Iteration {} | Q Loss {} | Actor Loss {}".format(
            #             i, info["q_loss"], info["actor_loss"]
            #         )
            #     )
            # if early_stop(loss_value):
            #     print("SAC loss early stop criteria met, exiting training loop")
            #     break
            if wandb.run is not None:
                wandb.log(info)

        return info

    @torch.no_grad()
    def select_action(self, state: State, eval_mode: EvalMode = False, t0: T0 = None):
        if isinstance(state, np.ndarray):
            # state = torch.from_numpy(state).to(device).float()
            # print("state {}".format(state.shape))
            state = torch.tensor(state, dtype=torch.float64, device=self.device)
        if state.ndim == 2:
            # TODO added this but not checked its right
            # TODO previously had torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            assert state.shape[0] == 1
            state = state[0, ...]
            # print("state {}".format(state.shape))
        dist = self.actor.forward(state, std=self.std)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action


def update_q(
    optim: torch.optim.Optimizer,
    actor: Actor,
    critic: Critic,
    critic_target: Critic,
    std_clip: float = 0.3,
):
    def update_q_fn(
        state: State,
        action: Action,
        reward: float,
        discount: float,
        next_state: State,
        std: float,
    ):
        with torch.no_grad():
            next_action = actor(next_state, std=std).sample(clip=std_clip)

            td_target = reward + discount * torch.min(
                *critic_target(next_state, next_action)
            )

        q1, q2 = critic(state, action)
        q_loss = torch.mean(util.mse(q1, td_target) + util.mse(q2, td_target))

        optim.zero_grad(set_to_none=True)
        q_loss.backward()
        optim.step()

        return {"q": q1.mean().item(), "q_loss": q_loss.item()}

    return update_q_fn


def update_actor(
    optim: torch.optim.Optimizer, actor: Actor, critic: Critic, std_clip: float = 0.3
):
    def update_actor_fn(state: State, std: float):
        action = actor(state, std=std).sample(clip=std_clip)
        Q = torch.min(*critic(state, action))
        actor_loss = -Q.mean()

        optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        optim.step()

        return {"actor_loss": actor_loss.item()}

    return update_actor_fn


# def init(
#     state_dim: int,
#     action_dim: int,
#     mlp_dims: List[int] = [512, 512],
#     learning_rate: float = 3e-4,
#     num_iterations: int = 100,
#     std_schedule: str = "linear(1.0, 0.1, 50)",
#     std_clip: float = 0.3,
#     nstep: int = 3,
#     gamma: float = 0.99,
#     tau: float = 0.005,
#     device: str = "cuda",
# ) -> Agent:
#     actor = Actor(state_dim, mlp_dims, action_dim).to(device)
#     critic = Critic(state_dim=state_dim, mlp_dims=mlp_dims, action_dim=action_dim).to(
#         device
#     )
#     critic_target = Critic(
#         state_dim=state_dim, mlp_dims=mlp_dims, action_dim=action_dim
#     ).to(device)

#     # Init optimizer
#     optim_actor = torch.optim.Adam(actor.parameters(), lr=learning_rate)
#     optim_critic = torch.optim.Adam(critic.parameters(), lr=learning_rate)

#     update_q_fn = update_q(
#         optim=optim_critic,
#         actor=actor,
#         critic=critic,
#         critic_target=critic_target,
#         std_clip=std_clip,
#     )
#     update_actor_fn = update_actor(
#         optim=optim_actor, actor=actor, critic=critic, std_clip=std_clip
#     )

#     # def train_fn(data_loader: DataLoader):
#     def train_fn(replay_iter) -> dict:
#         std = 0.1
#         info = {"std": std}
#         # data_iter = iter(data_loader)
#         # for batch in data_loader:
#         for i in range(num_iterations):
#             # i = epoch * num_iterations + batch_idx
#             # for i, batch in enumerate(data_loader):
#             # std = linear_schedule(std_schedule, i)  # linearly udpate std
#             # info = {"std": std}
#             std = 0.1
#             info = {"std": std}

#             # std_schedule, (num_iterations - 10) * episode_idx + i

#             batch = next(replay_iter)
#             state, action, reward, discount, next_state = utils.to_torch(
#                 batch, device, dtype=torch.float32
#             )
#             # swap the batch and horizon dimension -> [H, B, _shape]
#             action, reward, discount, next_state = (
#                 torch.swapaxes(action, 0, 1),
#                 torch.swapaxes(reward, 0, 1),
#                 torch.swapaxes(discount, 0, 1),
#                 torch.swapaxes(next_state, 0, 1),
#             )

#             # form n-step samples
#             _reward, _discount = 0, 1
#             for t in range(nstep):
#                 _reward += _discount * reward[t]
#                 _discount *= gamma

#             info.update(
#                 update_q_fn(state, action[0], _reward, _discount, next_state, std)
#             )
#             info.update(update_actor_fn(state, std))

#             # Update target network
#             util.soft_update_params(critic, critic_target, tau=tau)
#             # if i % 10 == 0:
#             #     print(
#             #         "DDPG: Iteration {} | Q Loss {} | Actor Loss {}".format(
#             #             i, info["q_loss"], info["actor_loss"]
#             #         )
#             #     )
#             # if early_stop(loss_value):
#             #     print("SAC loss early stop criteria met, exiting training loop")
#             #     break
#             wandb.log(info)

#         return info

#     @torch.no_grad()
#     def select_action_fn(state: State, eval_mode: EvalMode = False, t0: T0 = None):
#         state = torch.Tensor(state)
#         dist = actor.forward(state, std=0)
#         if eval_mode:
#             action = dist.mean
#         else:
#             action = dist.sample(clip=None)
#         return action

#     return Agent(select_action=select_action_fn, train=train_fn)
