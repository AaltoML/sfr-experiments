#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib
import wandb
from scipy.stats import sem


LABELS = {
    "sfr-sample": "\our (Ours)",
    "mlp": "\sc mlp",
    "ddpg": "\sc ddpg",
    "laplace-sample": "\sc laplace-glm",
    "ensemble-sample": "\sc ensemble",
}
COLORS = {
    "sfr-sample": "c",
    "mlp": "m",
    "ddpg": "y",
    "laplace-sample": "r",
    "ensemble-sample": "g",
}
LINESTYLES = {
    "sfr-sample": "-",
    "mlp": "-",
    "ddpg": "-",
    "laplace-sample": "-",
    "ensemble-sample": "-",
}


def plot_train_curves(
    save_dir: str = "./", filename: str = "rl", seed: int = 42, window_width: int = 8
):
    # Fix random seed for reproducibility
    np.random.seed(seed)

    api = wandb.Api()

    fig, ax = plt.subplots()

    df = pd.read_csv("rl_data.csv")
    print(df)

    df["episode"] = df.apply(lambda row: int(row["env_step"] / 1000), axis=1)

    df_with_stats = (
        df.groupby(["agent", "episode"])
        .agg(
            mean=("episode_return", "mean"),
            sem=("episode_return", "sem"),
            count=("episode_return", "count"),
        )
        .reset_index()
    )
    print(df_with_stats)

    for agent_name in df.agent.unique():
        agent_df = df_with_stats[df_with_stats["agent"] == agent_name]
        agent_df = agent_df[~pd.isnull(agent_df["sem"])]
        ax.plot(
            agent_df["episode"].to_numpy(),
            agent_df["mean"].to_numpy(),
            label=LABELS[agent_name],
            color=COLORS[agent_name],
            linestyle=LINESTYLES[agent_name],
        )
        ax.fill_between(
            agent_df["episode"].values,
            agent_df["mean"].to_numpy() - agent_df["sem"].to_numpy(),
            agent_df["mean"].to_numpy() + agent_df["sem"].to_numpy(),
            alpha=0.1,
            color=COLORS[agent_name],
        )
        if agent_name in [
            "ddpg",
            "mlp",
            "ensemble-sample",
            # "laplace-sample",
            # "sfr-sample",
        ]:
            ax.plot(
                agent_df["episode"].values,
                np.ones_like(agent_df["episode"]) * np.max(agent_df["mean"].to_numpy()),
                color=COLORS[agent_name],
                linestyle="--",
            )

    # breakpoint()
    ax.set_xlim(0, 50)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    plt.legend()

    plt.savefig(os.path.join(save_dir, filename + ".pdf"), transparent=True)
    tikzplotlib.save(
        os.path.join(save_dir, filename + ".tex"),
        axis_width="\\figurewidth",
        axis_height="\\figureheight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", help="directory to save figures", default="./")
    parser.add_argument(
        "--seed", type=int, help="fix random seed for reproducibility", default=42
    )
    args = parser.parse_args()

    plot_train_curves(save_dir=args.save_dir, seed=args.seed)
