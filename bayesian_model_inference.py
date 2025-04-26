import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# Load trace
trace = az.from_netcdf('models/bayesian_unemployment_trace.nc')
posterior_samples = az.extract(trace)

# Define correct age groups
age_groups = ['15-24', '25-29', '30-39', '40-49', '50 and over']

# New input
new_year = 2025
new_sex_code = 1  # female
new_age_group = "30-39"

# Find correct index
new_age_idx = np.where(np.array(age_groups) == new_age_group)[0][0]

# Extract posterior parameter samples
alpha_samples = posterior_samples['alpha'].values
beta_year_samples = posterior_samples['beta_year'].values
beta_sex_samples = posterior_samples['beta_sex'].values
sigma_samples = posterior_samples['sigma'].values

# Extract beta_age correctly - the shape is (5, 8000), so we need to index differently
beta_age_all_samples = posterior_samples['beta_age'].values
beta_age_samples = beta_age_all_samples[new_age_idx, :]  # Get the specific age group samples

# Now predict
lin_pred_samples = alpha_samples + beta_year_samples * new_year + beta_sex_samples * new_sex_code + beta_age_samples
mu_samples = np.exp(lin_pred_samples)
lognormal_samples = np.random.lognormal(mean=np.log(mu_samples), sigma=sigma_samples, size=len(mu_samples))

# Plot
plt.figure(figsize=(10, 6))
plt.hist(lognormal_samples, bins=50, density=True, alpha=0.6, color='g')
plt.xlabel('Unemployment Duration (weeks)')
plt.ylabel('Density')
plt.title('Posterior Predictive Distribution')
plt.savefig('plots/posterior_predictive_dist.png')

# Print results
print(f"Prediction for year={new_year}, sex={'female' if new_sex_code==1 else 'male'}, age group={new_age_group}:")
print(f"Median Prediction: {np.median(lognormal_samples):.2f} weeks")
print(f"95% Credible Interval: [{np.percentile(lognormal_samples,2.5):.2f}, {np.percentile(lognormal_samples,97.5):.2f}] weeks")

# # Real observed durations (approximation)
# observed_durations = (lower + upper) / 2
# observed_durations[observed_durations <= 0] = 1  # Just like you did earlier

# # Posterior predictive samples
# predicted_samples = lognormal_samples

# # Call the function
# check_model_fit(observed_durations, predicted_samples, plot_path='plots/ppc_plot.png')