# Example implementation to generate survival curves using the given dataset and trained model

import numpy as np
import torch
import matplotlib.pyplot as plt
from data import UnemploymentSurvivalDataset
from model import BayesianRiskNetwork
from eval_utils import load_model_from_checkpoint

def monte_carlo_survival(model, x_cat, x_cont, time_points, n_samples=100, n_realizations=5, device="cpu"):
    all_survival_curves = []

    model.eval()
    x_cat = x_cat.unsqueeze(0).to(device)
    x_cont = x_cont.unsqueeze(0).to(device)

    for _ in range(n_realizations):
        survival_probs = []
        for t in time_points:
            count_survive = 0
            for _ in range(n_samples):
                mu, sigma = model(x_cat, x_cont)
                y_sample = torch.normal(mu, sigma)
                T_sample = torch.exp(y_sample)
                count_survive += (T_sample > t).float().item()
            survival_probs.append(count_survive / n_samples)
        all_survival_curves.append(survival_probs)

    all_survival_curves = np.array(all_survival_curves)
    mean_curve = all_survival_curves.mean(axis=0)
    lower = np.percentile(all_survival_curves, 2.5, axis=0)
    upper = np.percentile(all_survival_curves, 97.5, axis=0)

    return mean_curve, lower, upper, all_survival_curves

# Load dataset and model
dataset = UnemploymentSurvivalDataset("datasets/unemployment_survival_data.csv")
category_sizes = dataset.get_category_sizes()
n_cont_features = dataset.get_continuous_feature_count()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = load_model_from_checkpoint(
    model_class=BayesianRiskNetwork,
    category_sizes=category_sizes,
    n_cont_features=n_cont_features,
    checkpoint_path="best_model.pt"
).to(device)

# Choose two samples: one assumed "Employed" and one "Unemployed" for demonstration
unemployed_idx = 100  # Example index

x_cat_unemp, x_cont_unemp, _, _ = dataset[unemployed_idx]

time_points = np.linspace(1, 60, 60)  # 1 to 60 weeks, every week


# Generate survival curves
mean_unemp, lower_unemp, upper_unemp, all_unemp_curves = monte_carlo_survival(model, x_cat_unemp, x_cont_unemp, time_points, device=device)

# Plotting
plt.figure(figsize=(8, 5))

for curve in all_unemp_curves:
    plt.plot(time_points, curve, color='orange', alpha=0.1)

plt.plot(time_points, mean_unemp, color='orange', label='Unemployed', linewidth=2)

plt.xlabel("Weeks")
plt.ylabel("S(t)")
plt.title("Survival Curves with Monte Carlo Integration")
plt.legend()
plt.show()
