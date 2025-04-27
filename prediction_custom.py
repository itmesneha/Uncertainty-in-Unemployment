import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from data import UnemploymentSurvivalDataset
from model import BayesianRiskNetwork
from eval_utils import load_model_from_checkpoint, predict_survival_probability

########################
# Load Dataset and Model
########################
dataset = UnemploymentSurvivalDataset("datasets/unemployment_survival_data.csv")
category_sizes = dataset.get_category_sizes()
n_cont_features = dataset.get_continuous_feature_count()

model = load_model_from_checkpoint(
    model_class=BayesianRiskNetwork,
    category_sizes=category_sizes,
    n_cont_features=n_cont_features,
    checkpoint_path="models/best_model.pt"
)

######################################
# Load Custom Input File (CHANGE HERE)
######################################
input_file = "datasets/qual_effect.json"  # Change this to the desired JSON file
with open(input_file, "r") as f:
    custom_inputs = json.load(f)

time_points = np.linspace(1, 60, 60)  # Weeks 1 to 60

plt.figure(figsize=(10, 6))

######################################
# Process Each Custom Input and Plot
######################################
for idx, entry in enumerate(custom_inputs):
    # Encode categorical features
    year_encoded = dataset.year_encoder[entry["year"]]
    qual_encoded = dataset.qual_encoder[entry["qualification"]]
    age_encoded = dataset.age_encoder[entry["age"]]
    sex_encoded = dataset.sex_encoder[entry["sex"]]

    x_cat = torch.tensor([year_encoded, qual_encoded, age_encoded, sex_encoded], dtype=torch.long)

    # Normalize continuous feature
    x_cont_value = (entry["estimated_unemployed"] - dataset.cont_mean[0]) / dataset.cont_std[0]
    x_cont = torch.tensor([x_cont_value], dtype=torch.float32)

    # Predict survival probabilities
    mean_probs, lower, upper = predict_survival_probability(
        model=model,
        x_cat=x_cat,
        x_cont=x_cont,
        time_points=time_points,
        n_samples=500
    )

    # Prepare label based on the feature being varied
    label_parts = []
    for key in ["year", "qualification", "age", "sex", "estimated_unemployed"]:
        label_parts.append(f"{key.capitalize()}: {entry[key]}")
    label = ", ".join(label_parts)

    plt.plot(time_points, mean_probs, label=label)
    plt.fill_between(time_points, lower, upper, alpha=0.15)

    # Print survival probability at t_query
    def print_survival_at_t(time_points, mean_probs, lower, upper, t_query, idx):
        closest_idx = (np.abs(time_points - t_query)).argmin()
        selected_time = time_points[closest_idx]
        mean_prob_at_t = mean_probs[closest_idx]
        lower_at_t = lower[closest_idx]
        upper_at_t = upper[closest_idx]
        print(f"\n🟢 Input {idx+1} Survival at t = {selected_time} weeks:")
        print(f"Mean Survival Probability   : {mean_prob_at_t:.4f}")
        print(f"95% Credible Interval       : [{lower_at_t:.4f}, {upper_at_t:.4f}]")

    print_survival_at_t(time_points, mean_probs, lower, upper, entry["t_query"], idx)

######################################
# Final Plot Adjustments
######################################
plt.xlabel('Time (weeks)')
plt.ylabel('Survival Probability $S(t \mid x)$')
plt.title(f'Survival Curves Comparison ({input_file.split(".")[0].replace("_", " ").capitalize()})')
plt.legend(fontsize='small', loc='upper right')
plt.tight_layout()
output_path = f"assets/survival_curves_{input_file.split('.')[0]}.png"
plt.savefig(output_path)
print(f"\n✅ Survival curves saved as '{output_path}'")
