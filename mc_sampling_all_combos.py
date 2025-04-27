import numpy as np
import torch
import matplotlib.pyplot as plt
from eval_utils import predict_survival_probability  # Your function
from data import UnemploymentSurvivalDataset
from model import BayesianRiskNetwork
from eval_utils import load_model_from_checkpoint

# --- Load dataset and model ---
dataset = UnemploymentSurvivalDataset("datasets/unemployment_survival_data.csv")
category_sizes = dataset.get_category_sizes()
n_cont_features = dataset.get_continuous_feature_count()

model = load_model_from_checkpoint(
    model_class=BayesianRiskNetwork,
    category_sizes=category_sizes,
    n_cont_features=n_cont_features,
    checkpoint_path="models/best_model_1.pt"
)

model.eval()

# --- Define time points ---
time_points = np.linspace(1, 60, 60)

# --- Groups to compare ---
age_groups = ['15-24', '25-29', '30-39', '40-49', '50 and over']
qualification = 'degree'  # You can change this to loop over qualifications as well
sex = 'female'  # Fixed for this example, can loop over as needed
year = 2023   # Adjust accordingly

# --- Create encoders from the dataset ---
year_encoder = dataset.year_encoder
qual_encoder = dataset.qual_encoder
age_encoder = dataset.age_encoder
sex_encoder = dataset.sex_encoder
print(qual_encoder)
plt.figure(figsize=(10, 6))

for age in age_groups:
    # Encode the group
    x_cat = torch.tensor([
        year_encoder[year],
        qual_encoder[qualification],
        age_encoder[age],
        sex_encoder[sex]
    ], dtype=torch.long).unsqueeze(0)

    # For continuous features (e.g., estimated_unemployed) assume mean
    x_cont_value = np.array([dataset.df['estimated_unemployed'].mean()])
    x_cont = torch.tensor((x_cont_value - dataset.cont_mean) / dataset.cont_std, dtype=torch.float32).unsqueeze(0)

    # Predict survival probabilities
    mean_probs, lower, upper = predict_survival_probability(
        model=model,
        x_cat=x_cat.squeeze(0),
        x_cont=x_cont.squeeze(0),
        time_points=time_points,
        n_samples=500
    )

    # Plot survival curve
    plt.plot(time_points, mean_probs, label=f'Age: {age}', linewidth=2)

plt.xlabel('Weeks')
plt.ylabel('Survival Probability S(t)')
plt.title('Survival Curves by Age Group (Qualification: Secondary, Sex: Female)')
plt.legend()
plt.savefig("assets/mc_sampling_all_combos_2.png")
