# Reproducing RL figure
The RL figure can be created with:
``` shell
python create_rl_figure.py
```
This loads the runs from [rl_data.csv](rl_data.csv).

Alternatively, reproduce the runs with:
``` sh
python train.py -m +experiment=sfr-sample,laplace-sample,ensemble-sample,ddpg,mlp ++random_seed=100,69,50,666,54
```
and then populate the [rl_data.csv](rl_data.csv) by fetching the runs from W&B:
``` sh
python fetch_data_from_wandb.py
```
