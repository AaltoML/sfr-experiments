# Function-space Parameterization of Neural Networks for Sequential Learning
Code accompanying ICLR 2024 submission *Function-space Parameterization of Neural Networks for Sequential Learning*.
This repository contains code for reproducing the experiments in the ICLR 2024 paper.
Please see [this repo](https://github.com/AaltoML/sfr/tree/main) for a clean and minimal implementation of Sparse Function-space Representation of Neural Networks (SFR).
We recommend using the [clean and minimal repo](https://github.com/AaltoML/sfr/tree/main).

<table>
    <tr>
        <td>
            <a href="https://openreview.net/forum?id=2dhxxIKhqz&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)">
              <strong >Function-space Parameterization of Neural Networks for Sequential Learning</strong><br>
            </a>
            Aidan Scannell*, Riccardo Mereu*, Paul Chang, Ella Tamir, Joni Pajarinen, Arno Solin<br>
            <strong>International Conference on Learning Representations (ICLR 2024)</strong><br>
            <!-- <a href="https://arxiv.org/abs/2309.02195"><img alt="Paper" src="https://img.shields.io/badge/-Paper-gray"></a> -->
            <!-- <a href="https://github.com/aidanscannell/sfr"><img alt="Code" src="https://img.shields.io/badge/-Code-gray" ></a> -->
        </td>
    </tr>
    <tr>
        <td>
            <a href="https://arxiv.org/abs/2309.02195">
              <strong>Sparse Function-space Representation of Neural Networks</strong><br>
            </a>
            Aidan Scannell*, Riccardo Mereu*, Paul Chang, Ella Tamir, Joni Pajarinen, Arno Solin<br>
            <strong>ICML 2023 Workshop on Duality Principles for Modern Machine Learning</strong><br>
            <!-- <a href="https://arxiv.org/abs/2309.02195"><img alt="Paper" src="https://img.shields.io/badge/-Paper-gray"></a> -->
            <!-- <a href="https://github.com/aidanscannell/sfr"><img alt="Code" src="https://img.shields.io/badge/-Code-gray" ></a> -->
        </td>
    </tr>
</table>


## Install

### Install using virtual environment
Make a virtual environment:
``` sh
python -m venv .venv
```
Activate it with:
``` sh
source .venv/bin/activate
```
Install the dependencies with:
``` sh
python -m pip install --upgrade pip
pip install laplace-torch==0.1a2
pip install -e ".[experiments]"
```
We install `laplace-torch` separately due to version conflicts with `backpacpk-for-pytorch`.
Note that `laplace-torch` is only used for running the baselines.

### Install using pip
Alternatively, manually install the dependencies with:
``` sh
pip install laplace-torch==0.1a2
pip install -r requirements.txt
```

## Reproducing experiments
See [experiments](./experiments/) for details on how to reproduce the results in the paper.
This includes code for generating the tables and figures.

## Useage
See the [notebooks/README.md](./notebooks) for how to use our code for both regression and classification.

### Example
Here's a short example:
```python
import src
import torch

torch.set_default_dtype(torch.float64)

def func(x, noise=True):
    return torch.sin(x * 5) / x + torch.cos(x * 10)

# Toy data set
X_train = torch.rand((100, 1)) * 2
Y_train = func(X_train, noise=True)
data = (X_train, Y_train)

# Training config
width = 64
num_epochs = 1000
batch_size = 16
learning_rate = 1e-3
delta = 0.00005  # prior precision
data_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*data), batch_size=batch_size
)

# Create a neural network
network = torch.nn.Sequential(
    torch.nn.Linear(1, width),
    torch.nn.Tanh(),
    torch.nn.Linear(width, width),
    torch.nn.Tanh(),
    torch.nn.Linear(width, 1),
)

# Instantiate SFR (handles NN training/prediction as they're coupled via the prior/likelihood)
sfr = src.sfr.SFR(
    network=network,
    prior=src.priors.Gaussian(params=network.parameters, delta=delta),
    likelihood=src.likelihoods.Gaussian(sigma_noise=2),
    output_dim=1,
    num_inducing=32,
    dual_batch_size=None, # this reduces the memory required for computing dual parameters
    jitter=1e-4,
)

sfr.train()
optimizer = torch.optim.Adam([{"params": sfr.parameters()}], lr=learning_rate)
for epoch_idx in range(num_epochs):
    for batch_idx, batch in enumerate(data_loader):
        x, y = batch
        loss = sfr.loss(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

sfr.set_data(data) # This builds the dual parameters

# Make predictions in function space
X_test = torch.linspace(-0.7, 3.5, 300, dtype=torch.float64).reshape(-1, 1)
f_mean, f_var = sfr.predict_f(X_test)

# Make predictions in output space
y_mean, y_var = sfr.predict(X_test)
```


## Citation
Please consider citing our ICLR 2024 paper.
```bibtex
@inproceedings{scannellFunction2024,
  title           = {Function-space Prameterization of Neural Networks for Sequential Learning},
  booktitle       = {Proceedings of The Twelth International Conference on Learning Representations (ICLR 2024)},
  author          = {Aidan Scannell and Riccardo Mereu and Paul Chang and Ella Tami and Joni Pajarinen and Arno Solin},
  year            = {2024},
  month           = {5},
}
```
