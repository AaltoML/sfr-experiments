#!/usr/bin/env python3
import hydra
import omegaconf
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="main")
def train(cfg: DictConfig):
    import logging
    import os
    import random
    import time
    from pathlib import Path

    import experiments
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import wandb
    from dm_env import StepType
    from experiments.rl.utils import ReplayBuffer, set_seed_everywhere

    # This is needed to render videos on GPU
    # os.environ["MUJOCO_GL"] = "egl"
    # os.environ["MUJOCO_GL"] = "osmesa"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.set_default_dtype(torch.float64)

    ##### Make experiment reproducible #####
    try:
        set_seed_everywhere(cfg.random_seed)
    except:
        random_seed = random.randint(0, 10000)
        set_seed_everywhere(random_seed)

    ##### Put tensors on GPU if requested and available #####
    if "cuda" in cfg.device:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {cfg.device}")

    ##### Ensure that all operations are deterministic on GPU #####
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    cfg.episode_length = cfg.episode_length // cfg.env.action_repeat
    num_train_steps = cfg.num_train_episodes * cfg.episode_length

    ##### Create train and evaluation environments #####
    env = hydra.utils.instantiate(cfg.env)
    ts = env.reset()
    eval_env = hydra.utils.instantiate(cfg.env, seed=cfg.env.seed + 42)

    ##### Hard coded in config #####
    # TODO don't hard code this in config
    # cfg.state_dim = tuple(int(x) for x in env.observation_spec().shape)
    # cfg.state_dim = cfg.state_dim[0]
    # cfg.action_dim = tuple(int(x) for x in env.action_spec().shape)
    # cfg.action_dim = cfg.action_dim[0]

    # ###### Set up workspace ######
    if cfg.wandb.use_wandb:  # Initialise WandB
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            dir=os.path.join(get_original_cwd(), "output"),
            monitor_gym=cfg.wandb.monitor_gym,
        )

    ###### Create video recorders ######
    eval_video_recorder = None
    if cfg.save_eval_video:
        eval_video_recorder = experiments.rl.utils.VideoRecorder(
            Path().cwd(), wandb=cfg.wandb.use_wandb
        )

    ##### Create replay buffer #####
    num_workers = 1
    replay_memory = ReplayBuffer(
        capacity=num_train_steps, batch_size=cfg.batch_size, device=cfg.device
    )

    ##### Create our custom reward function (sparse) #####
    reward_fn = hydra.utils.instantiate(cfg.reward_model).predict

    ##### Create agent #####
    agent = hydra.utils.instantiate(cfg.agent)

    start_time = time.time()
    last_time = start_time
    global_step = 0
    for episode_idx in range(cfg.num_train_episodes):
        logger.info(f"Episode {episode_idx} | Collecting data")

        ##### Collect trajectory #####
        time_step = env.reset()

        t, episode_reward = 0, 0
        while not time_step.last():
            ##### Select action #####
            if episode_idx <= cfg.init_random_episodes:
                action = np.random.uniform(-1, 1, env.action_spec().shape).astype(
                    dtype=np.float64
                )
            else:
                action_select_time = time.time()
                action = agent.select_action(
                    time_step.observation,
                    eval_mode=False,
                    t0=time_step.step_type == StepType.FIRST,
                )

                action_select_end_time = time.time()
                if t % 100 == 0:
                    logger.info(
                        "timestep={} took {}s to select action".format(
                            t, action_select_end_time - action_select_time
                        )
                    )
                action = action.cpu().numpy()

            state = torch.Tensor(time_step["observation"]).to(cfg.device)

            ##### Apply action in the environment #####
            time_step = env.step(action)

            ##### Get the reward from our custom reward function #####
            action_input = torch.Tensor(time_step["action"]).to(cfg.device)
            next_state = torch.Tensor(time_step["observation"]).to(cfg.device)
            reward = reward_fn(
                state=next_state[None, ...], action=action_input[None, ...]
            ).reward_mean

            ##### Add transition to replay buffer #####
            replay_memory.push(
                state=state.clone().to(cfg.device),
                action=action_input.clone().to(cfg.device),
                next_state=next_state.clone().to(cfg.device),
                reward=reward.clone().to(cfg.device),
            )

            global_step += 1
            t += 1
            # episode_reward += time_step["reward"] # TODO put back to env reward
            episode_reward += reward

        logger.info("Finished collecting {} time steps".format(t))

        ##### Log training metrics #####
        env_step = global_step * cfg.env.action_repeat
        elapsed_time = time.time() - last_time
        total_time = time.time() - start_time
        last_time = time.time()
        train_metrics = {
            "episode": episode_idx,
            "step": global_step,
            "env_step": env_step,
            "episode_time": elapsed_time,
            "total_time": total_time,
            "episode_return": episode_reward,
        }
        logger.info(
            "TRAINING | Episode: {} | Reward: {}".format(episode_idx, episode_reward)
        )
        if cfg.wandb.use_wandb:
            wandb.log({"train/": train_metrics})

        if episode_idx >= cfg.init_random_episodes:
            ##### Train agent #####
            logger.info("Training agent")
            agent.train(replay_memory)

            ##### Log rewards/videos in eval env #####
            if episode_idx % cfg.eval_episode_freq == 0:
                logger.info("Starting eval episodes")
                eval_episode_reward, eval_episode_std = experiments.rl.utils.evaluate(
                    eval_env,
                    agent,
                    episode_idx=episode_idx,
                    num_episodes=cfg.num_eval_episodes,
                    video=eval_video_recorder,
                    reward_fn=reward_fn,
                )
                eval_metrics = {
                    "episode": episode_idx,
                    "env_step": env_step,
                    "total_time": total_time,
                    "episode_return": eval_episode_reward,
                    "episode_return_std": eval_episode_std,
                }
                logger.info(
                    f"EVAL | Episode: {episode_idx} | Reward: {eval_episode_reward}"
                )
                if cfg.wandb.use_wandb:
                    wandb.log({"eval/": eval_metrics})


if __name__ == "__main__":
    train()  # pyright: ignore
