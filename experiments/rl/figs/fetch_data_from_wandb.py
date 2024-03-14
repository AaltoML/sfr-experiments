#!/usr/bin/env python3
import pandas as pd
import wandb


WANDB_ENTITY = "aalto-ml"
# WANDB_PROJECT = "sfr-experiments"
WANDB_PROJECT = "nn2svgp"


WANDB_RUNS = [
    # SFR
    "1981eud7",  # 100
    "1y1vf8xk",  # 69
    "6cud81zp",  # 50
    "mspoc1tp",  # 666
    "m20zxspk",  # 54
    # Laplace
    "ovi8ooil",  # 100
    "ttgleyig",  # 666
    "r6j12038",  # 50
    "zc55qobv",  # 69
    "p92x5hkv",  # 54
    # Ensemble
    "zzcv9ew2",  # 100
    "bg3cgoze",  # 666
    "4jtgz6l5",  # 50
    "afpm5hc2",  # 69
    "jlumgtyu",  # 54
    # MLP
    "1ags2von",  # 666
    "3mnfz5s0",  # 100
    "nlcicvp0",  # 69
    "zigq11ow",  # 50
    "3cxumxzq",  # 54
    # DDPG
    "3vrkzcgo",  # 100
    "24trnix9",  # 666
    "2ej235vk",  # 50
    "rbq90bf5",  # 69
    "1ytpofrx",  # 54
]


TITLES = ["env_step", "episode_return"]
KEYS = ["train/.env_step", "train/.episode_return"]

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
    fetch_data()
