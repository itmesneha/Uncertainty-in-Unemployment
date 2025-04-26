import pymc as pm
import numpy as np
import pandas as pd
import logging
import arviz as az
import matplotlib.pyplot as plt
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Starting optimized Bayesian model script...")

# Create plots and models directory if they don't exist
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load data
logging.info("Loading dataset from CSV...")
df = pd.read_csv('datasets/unemployment_survival_data.csv')
logging.info(f"Original dataset shape: {df.shape}")

# OPTIMIZATION: Subsample the data to speed up training
# Taking a random 25% of the data
subsample_fraction = 0.25
df = df.sample(frac=subsample_fraction, random_state=42)
logging.info(f"Subsampled dataset shape: {df.shape}")

# Keep only necessary columns
df = df[['year', 'highest_qualification', 'age', 'sex', 'duration', 'estimated_unemployed']]

# Define duration map
duration_map = {
    'under 5': (0, 4),
    '5 to 9': (5, 9),
    '10 to 14': (10, 14),
    '15 to 19': (15, 19),
    '20 to 24': (20, 24),
    '25 to 29': (25, 29),
    '30 to 39': (30, 39),
    '40 to 51': (40, 51),
    '52 and over': (52, 104)
}

df['lower_bound'] = df['duration'].map(lambda x: duration_map[x][0])
df['upper_bound'] = df['duration'].map(lambda x: duration_map[x][1])

# Encode categorical variables
sex_map = {'male': 0, 'female': 1}
df['sex_code'] = df['sex'].map(sex_map)

qualifications = df['highest_qualification'].unique()
qualification_idx = pd.Categorical(df['highest_qualification'], categories=qualifications).codes

ages = df['age'].unique()
age_idx = pd.Categorical(df['age'], categories=ages).codes

# Prepare numpy arrays
year = df['year'].astype(float).to_numpy()
year_mean = np.mean(year)
year = year - year_mean
np.save('models/year_mean.npy', year_mean)
sex = df['sex_code'].astype(float).to_numpy()
lower = df['lower_bound'].to_numpy()
upper = df['upper_bound'].to_numpy()
estimated_unemployed = df['estimated_unemployed'].astype(float).to_numpy()

# Raw lower and upper bounds
lower = np.clip(lower, a_min=1e-3, a_max=None)
upper = np.clip(upper, a_min=1e-3, a_max=None)

# Midpoint before log
midpoint = (lower + upper) / 2
midpoint = np.clip(midpoint, a_min=1e-3, a_max=None)

# Log transform
observed_duration = np.log(midpoint)

# For Censored bounds, also log lower and upper
lower = np.log(lower)
upper = np.log(upper)
# observed_duration = np.clip(observed_duration, a_min=1e-3, a_max=None)  # Clip BEFORE log

logging.info("Preprocessing complete.")

# Build Model
with pm.Model() as unemployment_model:
    # Priors
    alpha = pm.Normal('alpha', mu=3, sigma=2)
    beta_year = pm.Normal('beta_year', mu=0, sigma=1)
    beta_sex = pm.Normal('beta_sex', mu=0, sigma=1)
    sigma_age = pm.HalfNormal('sigma_age', sigma=1)
    sigma_qualification = pm.HalfNormal('sigma_qualification', sigma=1)

    beta_age = pm.Normal('beta_age', mu=0, sigma=sigma_age, shape=len(ages))
    beta_qualification = pm.Normal('beta_qualification', mu=0, sigma=sigma_qualification, shape=len(qualifications))

    # Linear predictor
    lin_pred = (alpha
                + beta_year * year
                + beta_sex * sex
                + beta_age[age_idx]
                + beta_qualification[qualification_idx])

    # mu = pm.math.exp(lin_pred)

    # Likelihood noise (sigma for LogNormal)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Censored observed data
    duration_obs = pm.Censored(
        "duration_obs",
        pm.StudentT.dist(nu=3, mu=lin_pred, sigma=sigma),
        lower=lower,
        upper=upper,
        observed=observed_duration
    )

    # OPTIMIZATION
    # Fewer samples
    n_samples = 500  # Reduced from 2000
    n_tune = 300     # Reduced from 1000
    
    #  NUTS sampler with optimized parameters
    trace = pm.sample(
        n_samples, 
        tune=n_tune, 
        target_accept=0.95,   
        max_treedepth=15,    
        cores=8,             
        chains = 4,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )

    # Save trace
    az.to_netcdf(trace, 'models/bayesian_unemployment_trace.nc')
    logging.info('Saved posterior trace to models/bayesian_unemployment_trace.nc')

    # Plot posterior parameter distributions
    az.plot_posterior(trace, var_names=["alpha", "beta_year", "beta_sex", "sigma"], hdi_prob=0.95)
    plt.savefig('plots/posterior_parameter_plots.png')
    logging.info('Saved posterior parameter plots.')

    # OPTIMIZATION: Generate fewer posterior predictive samples
    # Take only a subset of posterior samples to generate predictions
    posterior_samples = az.extract(trace, num_samples=100)  # Reduced number of samples

    alpha_samples = posterior_samples['alpha'].values
    beta_year_samples = posterior_samples['beta_year'].values
    beta_sex_samples = posterior_samples['beta_sex'].values
    sigma_samples = posterior_samples['sigma'].values
    beta_age_samples = posterior_samples['beta_age'].values.T
    beta_qualification_samples = posterior_samples['beta_qualification'].values.T

    n_samples = alpha_samples.shape[0]
    n_data = len(year)

    lin_pred_samples = (
        alpha_samples[:, None]
        + beta_year_samples[:, None] * year
        + beta_sex_samples[:, None] * sex
        + beta_age_samples[:, age_idx]
        + beta_qualification_samples[np.arange(n_samples)[:, None], qualification_idx[None, :]]
    )

    mu_samples = np.exp(lin_pred_samples)

    rng = np.random.default_rng(seed=42)
    lognormal_samples = rng.lognormal(mean=np.log(mu_samples), sigma=sigma_samples[:, None])

    # Save posterior predictive samples
    np.save('models/lognormal_samples.npy', lognormal_samples)
    logging.info('Saved posterior predictive samples to models/lognormal_samples.npy')

logging.info("Optimized Bayesian model training complete.")