# Supervised learning experiments
Instructions for reproducing the supervised learning (UCI/Fashion-MNIST/CIFAR-10) results in the paper.
[The notebooks](./media/) contain all the information for reproducing the experiments and generating the tables/figures in the ICRL paper.

# Running experiments

## Image classification experiments
[This notebook](./media/image-experiments-create-figs+tabs.ipynb) shows how to create the tables and figures for the image classification experiments in the ICRL paper. 
The SFR experiments were submitted with slurm using:
```sh
python train_and_inference.py -m ++wandb.use_wandb=True +experiment=cifar10_cnn,fmnist_cnn ++run_laplace_flag=False ++run_sfr_flag=True ++num_inducings='[3200]','[2048]','[1024]','[512]','[256]','[128]' ++random_seed=68,117,36,187,17 hydra/launcher=lumi_20hrs
```
and the Laplace baselines were submitted with:
```sh
python train_and_inference.py -m ++wandb.use_wandb=True +experiment=cifar10_cnn,fmnist_cnn ++run_laplace_flag=True ++run_sfr_flag=False ++random_seed=68,117,36,187,17
```
Alternatively, a single job can be run with:
```sh
python train_and_inference.py ++wandb.use_wandb=True +experiment=cifar10_cnn ++run_laplace_flag=False ++run_sfr_flag=True ++num_inducings='[2048]' ++random_seed=68
```
This avoids using hydra's sumbitit launcher plugin to submit jobs using slurm.

We report results (test. Accuracy, test. NLPD, test. ECE, AUROC) on Fashion-MNIST/CIFAR-10 for:
- Baselines
    - NN MAP
    - Laplace BNN/GLM with diag/kron Hessian structures both with/without prior precision $\delta$ tuning
    - GP predictive (Immer et al 2021) for M=3200 with/without prior precision $\delta$ tuning
- Ablation
    - SFR for M=128/256/512/1024/2048/3200 with/without prior precision $\delta$ tuning
    - GP subset for M=128/256/512/1024/2048/3200 with/without prior precision $\delta$ tuning

## UCI experiments
[This notebook](./media/uci-experiments-create-figs+tabs.ipynb) shows how to create the tables and figures for the UCI experiments in the ICRL paper. 
We logged runs using W&B which were run with:
```sh
python train_and_inference.py -m ++wandb.use_wandb=True +experiment=australian_uci,breast_cancer_uci,digits_uci,glass_uci,ionosphere_uci,satellite_uci,vehicle_uci,waveform_uci ++random_seed=68,117,36,187,17 hydra/launcher=lumi_5hrs
```
This submits 5x seeds for each of the 8 UCI data sets. Each job logs test Accuracy, test NLPD, and test ECE for:
- Baselines
    - NN MAP
    - Laplace BNN/GLM with kron/diag Hessian structures both with/without prior precision $\delta$ tuning
    - GP predictive (Immer et al 2021) for M=3200 with/without prior precision $\delta$ tuning
- Ablation
    - SFR for M=1%/2%/5%/10%/15%/20%/40%/60%/80%/100% of N with/without prior precision $\delta$ tuning
    - GP subset for M=1%/2%/5%/10%/15%/20%/40%/60%/80%/100% of N with/without prior precision $\delta$ tuning

## Running single experiments
Run the Fashion-MNIST image classification experiment with:
``` sh
python train_and_inference.py +experiment=fmnist_cnn
```
Or submit both of the image classification experiments to slurm with:
``` sh
python train_and_inference.py -m +experiment=cifar10_cnn,fmnist_cnn
```
You can display the base config using:
``` shell
python train_and_inference.py --cfg=job
```
All experiments are detailed in [./configs/experiments](./configs/experiments).
