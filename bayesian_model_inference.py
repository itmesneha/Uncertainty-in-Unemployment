import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load trace
trace = az.from_netcdf('models/bayesian_unemployment_trace.nc')
posterior_samples = az.extract(trace)

# Load year mean for normalization
year_mean = np.load('models/year_mean.npy')

# Load training data to get age and qualification categories
df = pd.read_csv('datasets/unemployment_survival_data.csv')
ages = df['age'].unique()
qualifications = df['highest_qualification'].unique()

# --- User input ---
new_year = 2025
new_sex_code = 1  # female
new_age_group = "30-39"
new_qualification = qualifications[0]  # or set as needed

# Normalize year
new_year_norm = new_year - year_mean

# Find correct indices
new_age_idx = np.where(ages == new_age_group)[0][0]
new_qualification_idx = np.where(qualifications == new_qualification)[0][0]

# Extract posterior parameter samples
alpha_samples = posterior_samples['alpha'].values
beta_year_samples = posterior_samples['beta_year'].values
beta_sex_samples = posterior_samples['beta_sex'].values
sigma_samples = posterior_samples['sigma'].values
beta_age_samples = posterior_samples['beta_age'].values.T[:, new_age_idx]
beta_qualification_samples = posterior_samples['beta_qualification'].values.T[:, new_qualification_idx]

# Now predict
lin_pred_samples = (
    alpha_samples
    + beta_year_samples * new_year_norm
    + beta_sex_samples * new_sex_code
    + beta_age_samples
    + beta_qualification_samples
)
mu_samples = np.exp(lin_pred_samples)
lognormal_samples = np.random.lognormal(mean=np.log(mu_samples), sigma=sigma_samples, size=len(mu_samples))

# Plot
plt.figure(figsize=(10, 6))
plt.hist(lognormal_samples, bins=50, density=True, alpha=0.6, color='g')
plt.xlabel('Unemployment Duration (weeks)')
plt.ylabel('Density')
plt.title('Posterior Predictive Distribution')
plt.savefig('plots/posterior_predictive_dist_inference.png')

# Print results
print(f"Prediction for year={new_year}, sex={'female' if new_sex_code==1 else 'male'}, age group={new_age_group}, qualification={new_qualification}:")
print(f"Median Prediction: {np.median(lognormal_samples):.2f} weeks")
print(f"95% Credible Interval: [{np.percentile(lognormal_samples,2.5):.2f}, {np.percentile(lognormal_samples,97.5):.2f}] weeks")
