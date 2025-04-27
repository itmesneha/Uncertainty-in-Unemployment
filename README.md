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
  


## Model Evaluation & Visualization

- **Posterior predictive checks:**  
  Visualize observed vs. predicted duration distributions, including credible intervals.

- **Summary plots:**  
  Boxplots and heatmaps for estimated unemployed by qualification, age, and sex.  
  Time trends for mean estimated unemployed.

- **Parameter posteriors:**  
  Visualize uncertainty in model parameters.

All plots are saved in the `plots/` directory.

## Project Structure
```
.
├── bayesian_model.py         # PyMC Bayesian survival model
├── bayesian_testing.py       # Posterior predictive checks, WAIC, LOO
├── bayesian_model_inference.py # Posterior predictive inference for new cases
├── data.py                   # PyTorch Dataset and DataLoader utilities
├── model.py                  # Bayesian neural network model (PyTorch)
├── train.py                  # Training loop for neural network
├── run_train.py              # Training entry point
├── loss.py                   # ELBO loss for survival analysis
├── eval_utils.py             # Model loading and evaluation utilities
├── mc_sampling.py            # Monte Carlo survival curve estimation
├── preprocessing.py          # Data fetching, IPF, and preprocessing
├── utils/
│   └── bayesian_check_model_fit.py # Posterior predictive check plotting
├── datasets/                 # Processed and raw data
├── models/                   # Saved model traces and parameters
├── plots/                    # All generated plots
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## References

1. Ministry of Manpower (2024), *Unemployed Residents Aged 15 Years and Over by Age and Duration of Unemployment*, data.gov, [https://data.gov.sg/datasets/d_db95e15ceffaa368a043310479dc7d57/view](https://data.gov.sg/datasets/d_db95e15ceffaa368a043310479dc7d57/view)

2. Ministry of Manpower (2024), *Unemployed Residents Aged 15 Years and Over by Highest Qualification Attained and Duration of Unemployment*, data.gov, [https://data.gov.sg/datasets/d_a0ca632fd1d6ff841f0e47298a9ab589/view](https://data.gov.sg/datasets/d_a0ca632fd1d6ff841f0e47298a9ab589/view)

3. Ministry of Manpower (2024), *Median Duration of Unemployment*, data.gov, [https://data.gov.sg/datasets/d_c01a3210fb10f1a52676f97498d4ec2c/view](https://data.gov.sg/datasets/d_c01a3210fb10f1a52676f97498d4ec2c/view)


## License

This project is licensed under the MIT License.
