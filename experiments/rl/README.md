# Reinforcement learning experiments
Instructions for reproducing the RL results in the paper.

## Reproducing figure
See [figs/](./figs) for details on reproducing the RL figure in the ICLR paper.

## Running experiments
Run a single experiment (e.g SFR) for a single random seed with:
``` sh
python train.py +experiment=sfr-sample ++random_seed=100
```
You can display the base config using:
``` shell
python train.py --cfg=job
```
and an experiment's config with:
``` shell
python train.py +experiment=sfr-sample --cfg=job
```
Run all of the RL experiments for 5 random seeds with (you'll need a cluster for this):
``` sh
python train.py -m +experiment=sfr-sample,laplace-sample,ensemble-sample,ddpg,mlp ++random_seed=100,69,50,666,54
```
