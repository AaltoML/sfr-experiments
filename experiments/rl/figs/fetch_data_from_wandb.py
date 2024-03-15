#!/usr/bin/env python3
import pandas as pd
import wandb


WANDB_ENTITY = "aalto-ml"
WANDB_PROJECT = "sfr-rl"

WANDB_RUNS = [
    # SFR
    "jf9q1rlx",  # 100
    "6lzuv74a",  # 69
    "zlndtzac",  # 50
    "yvkwf9y5",  # 666
    "qdrtvc1m",  # 54
    # "b7si8ixz",  # 100
    # "qv7sbg15",  # 69
    # "ied71nvd",  # 50
    # "e0f4vcg2",  # 666
    # "1rr0v1qg",  # 54
    # Laplace
    "q7lmtzqg",  # 100
    "femssfbq",  # 666
    "0z26ul11",  # 50
    "2uq7ewt9",  # 69
    "5osqnp6m",  # 54
    # Ensemble
    "jxen42b6",  # 100
    "m11lksm5",  # 666
    "feddigfn",  # 50
    "rp90l4mf",  # 69
    "c9r35pag",  # 54
    # MLP
    "7zz4c5qp",  # 666
    "7ad8dh4d",  # 100
    "ikj6p7dv",  # 69
    "0zl0f79o",  # 50
    "rf93ioyo",  # 54
    # DDPG
    "tz3l2g9x",  # 100
    "xq5dbhvc",  # 666
    "g2h4i98c",  # 50
    "oqr3fm3m",  # 69
    "ww46ejcr",  # 54
    # "b7yjffut",  # 100
    # "zspxa3sd",  # 666
    # "a947buks",  # 50
    # "ppkd3wrz",  # 69
    # "9q7j6cm9",  # 54
]


TITLES = ["env_step", "episode_return"]
KEYS = ["eval/.env_step", "eval/.episode_return"]

COLUMN_TITLES = {a: b for a, b in zip(KEYS, TITLES)}


def fetch_data(save_path: str = "./rl_data.csv"):
    api = wandb.Api(timeout=19)
    data = []
    for run_id in WANDB_RUNS:
        wandb_run = api.run(WANDB_ENTITY + "/" + WANDB_PROJECT + "/" + run_id)
        print(f"Fetching run with ID: {wandb_run}")
        history = wandb_run.history(keys=KEYS)
        history = history.rename(columns=COLUMN_TITLES)

        env_id = wandb_run.config["env"]["env_name"]
        dmc_task = wandb_run.config["env"]["task_name"]
        env_name = env_id + "-" + dmc_task
        history["env"] = env_name

        history["seed"] = wandb_run.config["random_seed"]
        history["agent"] = wandb_run.config["alg_name"]

        data.append(history)
    df = pd.concat(data)
    df.to_csv(save_path)
    return df


if __name__ == "__main__":
    fetch_data(save_path="./rl_data.csv")
