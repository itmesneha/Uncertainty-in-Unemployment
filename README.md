# Uncertainty in Unemployment: Bayesian Survival Analysis

This repository provides a full pipeline for analyzing unemployment duration data using Bayesian survival analysis. The workflow includes data preprocessing, joint distribution estimation via Iterative Proportional Fitting (IPF), Bayesian model fitting (with both PyMC and Bayesian neural networks), and comprehensive model evaluation and visualization.

# Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Bayesian Modeling](#bayesian-modeling)
- [Neural Network Survival Model](#neural-network-survival-model)
- [Model Evaluation & Visualization](#model-evaluation--visualization)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## Overview

This project aims to model the duration of unemployment using advanced Bayesian methods, quantifying uncertainty in predictions and allowing for flexible, interpretable modeling of time-to-event data. The approach combines:

- Data harmonization and marginal fitting using IPF
- Bayesian parametric survival modeling (PyMC)
- Variational Bayesian neural network survival modeling (PyTorch)
- Posterior predictive checks and uncertainty quantification

## Features

- Automated data fetching and preprocessing from Singapore government datasets
- Iterative Proportional Fitting (IPF) to estimate joint distributions from marginals
- Interval-censored survival modeling with PyMC, supporting right-censoring
- Bayesian neural network for survival analysis with variational inference (PyTorch)
- Posterior predictive checks, WAIC, LOO, and visual diagnostics
- Flexible, modular codebase for extension and experimentation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/uncertainty-in-unemployment.git
   cd uncertainty-in-unemployment
2. Install dependencies:
   ```bash
    pip install -r requirements.txt

## Data Preparation

The preprocessing pipeline fetches, harmonizes, and merges unemployment data by age, sex, qualification, and duration. It uses IPF to estimate joint distributions and produces a synthetic dataset suitable for survival analysis.

- Run preprocessing:
  ```bash
   python preprocessing.py

  ## Data Preparation

The preprocessing pipeline fetches, harmonizes, and merges unemployment data by age, sex, qualification, and duration. It uses IPF to estimate joint distributions and produces a synthetic dataset suitable for survival analysis. 

This will:
- Download and merge datasets
- Estimate joint distributions using IPF
- Compute hybrid duration probabilities
- Output a processed CSV at `datasets/unemployment_survival_data.csv`
- Generate summary plots in the `plots/` directory

## Bayesian Modeling

### PyMC Regression Model

**Script:** `bayesian_model.py`

**Features:**
- Models log-unemployment duration as a function of year, sex, age group, and qualification
- Handles interval-censored data using `pm.Censored`
- Uses a Student-t likelihood for robustness
- Outputs posterior samples and diagnostics
- Run the model:
  ```bash
    python bayesian_model.py
  ```
   - Posterior samples are saved to `models/bayesian_unemployment_trace.nc`
   - Posterior predictive samples are saved to `models/lognormal_samples.npy`
   - Diagnostic plots are saved in `plots/`
- Posterior predictive checks and model fit:
  ```bash
    python bayesian_testing.py
  ```
  - Compares observed and predicted durations
  - Computes WAIC and LOO for model comparison
- Inference for new cases
  ```bash
  python bayesian_model_inference.py
  ```
  - Predicts unemployment duration for new demographic profiles
 
## Neural Network Survival Model

### Variational Bayesian Neural Network (PyTorch)

- **Data loader:** `data.py`
- **Model:** `model.py`
- **Training:** `run_train.py`, `train.py`
- **Loss:** `loss.py` (ELBO for log-normal survival)
- **Evaluation:** `eval_utils.py`, `mc_sampling.py`
- **Train the model:** `python run_train.py`
  - Trains a variational BNN for survival analysis
  - Supports interval and right-censoring
- **Monte Carlo survival curve estimation:** `python mc_sampling.py`
  - Generates survival curves with credible intervals
  


