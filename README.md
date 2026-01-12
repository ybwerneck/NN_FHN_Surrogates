# Neural Network Surrogates for FitzHugh–Nagumo Dynamics

This repository contains the code developed to accompany the paper *Neural Network Surrogates for Action Potentials*. The project implements neural network surrogate models for the **FitzHugh–Nagumo (FHN)** model—a simplified representation of excitable systems commonly used in computational neuroscience and electrophysiology. The surrogate models accelerate simulation and inference relative to classical numerical solvers.

The code supports training, evaluation, analysis, and compilation of surrogate networks. It integrates physics‑informed training paradigms as appropriate, data generation pipelines, model architecture exploration, and high‑performance inference exports.

## Key Features

- Training of Neural Surrogates  
  Neural network models approximating the FHN dynamics under multiple problem configurations.

- Data Generation  
  Scripts to systematically generate training datasets from reference ODE solutions.

- Benchmarking & Analysis  
  Tools to assess surrogate performance against numerical solvers and across model variants.

- High‑Performance Inference  
  Support for optimizing trained PyTorch models using TensorRT via `torch2trt` to enable accelerated inference on CUDA‑enabled hardware.

## Requirements

Core Dependencies:

- PyTorch
- PINNtorch — authoral package, for physics-informed loss formulations in surrogate training. Available at: https://github.com/ybwerneck/Pinn-Torch.
- torch2trt — for converting PyTorch models to TensorRT‑accelerated modules.
- Standard scientific packages: numpy, scipy, matplotlib.


## Repository Structure

NN_FHN_Surrogates/
├── data_generator/           # Dataset creation utilities
├── Arch_generator/           # Model architecture search and generator scripts
├── analysis/                 # Analysis routines and performance metrics
├── Problem_A/                # Problem specification and training scripts 
├── Problem_B/                # Problem specification and training scripts 
├── Problem_C/                # Problem specification and training scripts 
├── Problem_D/                # Problem specification and training scripts (B with ITNN)
├── Results/                  # Generated results and figures
└── README.md                 # Project overview and instructions

## License

Include your chosen license in a LICENSE file.


