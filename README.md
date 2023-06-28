# [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287)

This repository is forked from [the official repository](https://github.com/MilesCranmer/symbolic_deep_learning) of [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287). 

Miles Cranmer, Alvaro Sanchez-Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, Shirley Ho

Check out the [Blog](https://astroautomata.com/paper/symbolic-neural-nets/), [Paper](https://arxiv.org/abs/2006.11287), [Video](https://youtu.be/2vwwu59RPL8), and [Interactive Demo](https://colab.research.google.com/github/MilesCranmer/symbolic_deep_learning/blob/master/GN_Demo_Colab.ipynb).

[![](images/discovering_symbolic_eqn_gn.png)](https://astroautomata.com/paper/symbolic-neural-nets/)


## Requirements

For model:

- pytorch
- [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric)
- numpy

Symbolic regression:
- [PySR](https://github.com/MilesCranmer/PySR), a new open-source Eureqa alternative

For simulations:

- [jax](https://github.com/google/jax) (simple N-body simulations)
- [quijote](https://github.com/franciscovillaescusa/Quijote-simulations) (Dark matter data; optional)
- tqdm
- matplotlib

## Training

To train an example model from the paper, try out the [demo](https://colab.research.google.com/github/MilesCranmer/symbolic_deep_learning/blob/master/GN_Demo_Colab.ipynb).

Full model definitions are given in `models.py`. Data is generated from `simulate.py`.

Respect to the original `models.py`, now there are implemented 2 new GNN, called PM (Plus-Minus) and CUST (Custom). For all GNN Mean Squared Error (MSE), Mean Absolute Error (MAE) and Huber Loss (HUBER) are implemented.

Instead `simulation.py` is not modified except for the memory allocation.

## Code summary

The project is divided as following:

- Required packages are installed;
- Data are generated with a simulation;
- An explanation of models used is provided;
- Dataset is prepared in Dataloader;
- Training is done;
- The training is analyzed;
- Symbolic regression: PySR is used to obtain analytical expression of the force;
- Conclusions and possible improvements.


The model used are some from the original project (Bottleneck, L1, KL) and 2 new GNN.
Simulations used are r1, r2 spring and charge.
Available losses are MSE, MAE and Huber loss.

You can customize your training with these possible choises.

