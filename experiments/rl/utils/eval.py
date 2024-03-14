#!/usr/bin/env python3
import logging
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import torch
from dm_env import StepType

from ..agents import Agent
from .buffer import ReplayBuffer
from .video import VideoRecorder


def evaluate(
    env,
    agent,
    episode_idx: int = 0,
    num_episodes: int = 10,
    video: Optional[VideoRecorder] = None,
    reward_fn=None,
):
    """Evaluate a trained agent and optionally save a video."""
    episode_returns = []
    for i in range(num_episodes):
        if video:
            video.init(env, enabled=(i == 0))
        episode_returns.append(
            rollout(
                env=env, agent=agent, eval_mode=True, video=video, reward_fn=reward_fn
            )
        )
        logger.info(
            "Eval episode {}/{}, G={}".format(i + 1, num_episodes, episode_returns[-1])
        )
        if i == 0:
            if video:
                video.save(episode_idx)
    return np.nanmean(episode_returns), np.nanstd(episode_returns)


def rollout(env, agent: Agent, eval_mode: bool = True, video=None, reward_fn=None):
    time_step = env.reset()
    episode_reward = 0
    while not time_step.last():
        action = agent.select_action(
            time_step.observation,
            eval_mode=eval_mode,
            t0=time_step.step_type == StepType.FIRST,
        )
        action_np = action.cpu().numpy()

        time_step = env.step(action_np)

        if reward_fn is not None:
            next_state = torch.tensor(time_step.observation)[None, ...].to(
                action.device
            )
            episode_reward += reward_fn(
                state=next_state, action=action[None, ...]
            ).reward_mean
        else:
            episode_reward += time_step.reward

        if video:
            video.record(env)

    if isinstance(episode_reward, torch.Tensor):
        episode_reward = episode_reward.cpu().numpy()[0]
    return episode_reward
