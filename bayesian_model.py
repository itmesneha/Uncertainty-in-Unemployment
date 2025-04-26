import pymc as pm
import numpy as np
import pandas as pd
import logging
import requests
import arviz as az
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Starting bayesian model script...")

def get_data(dataset_id, sample=None):
    logging.info(f"Fetching data for dataset: {dataset_id}")
    url = "https://data.gov.sg/api/action/datastore_search?resource_id=" + dataset_id
    params = {'offset': 0}
    dfs = []

    while True:
        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response['result']['records'])
        dfs.append(df)
        if response['result']['_links']['next'] is None or (sample is not None and len(dfs) * 100 >= sample):
            break
        params['offset'] += 100

    full_df = pd.concat(dfs, ignore_index=True)
    if sample is not None:
        full_df = full_df.head(sample)
    full_df = full_df.drop(['_id'], axis=1)
    logging.info(f"Finished loading dataset: {dataset_id} with shape {full_df.shape}")
    return full_df

# Prepare data
dataset_id = "d_db95e15ceffaa368a043310479dc7d57"
df_sex_age_unemployed_duration = get_data(dataset_id, 2000)
logging.info('Dataset loading complete.')

df_sex_age_unemployed_duration[['unemployed']] = df_sex_age_unemployed_duration[['unemployed']].apply(pd.to_numeric, errors='coerce')

# Convert duration categories to lower and upper bounds
duration_map = {
    'under 5': (0, 4),
    '5 to 9': (5, 9),
    '10 to 14': (10, 14),
    '15 to 19': (15, 19),
    '20 to 24': (20,24),
    '25 to 29': (25,29),
    '30 to 39': (30,39),
    '40 to 51': (40, 51),
    '52 and over': (52, 104)  # Set reasonable upper limit
}

df_sex_age_unemployed_duration['lower_bound'] = df_sex_age_unemployed_duration['duration'].map(lambda x: duration_map[x][0])
df_sex_age_unemployed_duration['upper_bound'] = df_sex_age_unemployed_duration['duration'].map(lambda x: duration_map[x][1])

# Encode sex as 0/1
sex_map = {'male': 0, 'female': 1}
df_sex_age_unemployed_duration['sex_code'] = df_sex_age_unemployed_duration['sex'].map(sex_map)

year = df_sex_age_unemployed_duration['year'].astype(float).to_numpy()
sex = df_sex_age_unemployed_duration['sex_code'].astype(float).to_numpy()
age_groups = df_sex_age_unemployed_duration['age'].unique()
age_idx = pd.Categorical(df_sex_age_unemployed_duration['age'], categories=age_groups).codes

with pm.Model() as unemployment_model:
    # Priors for coefficients
    alpha = pm.Normal('alpha', mu=20, sigma=10)
    beta_year = pm.Normal('beta_year', mu=0, sigma=1)
    beta_sex = pm.Normal('beta_sex', mu=0, sigma=2)
    sigma_age = pm.HalfNormal('sigma_age', sigma=5)
    beta_age = pm.Normal('beta_age', mu=0, sigma=sigma_age, shape=len(age_groups))

    lin_pred = alpha + beta_year * year + beta_sex * sex + beta_age[age_idx]
    mu = pm.math.exp(lin_pred)

    sigma = pm.HalfNormal('sigma', sigma=1)

    lower = df_sex_age_unemployed_duration['lower_bound'].to_numpy()
    upper = df_sex_age_unemployed_duration['upper_bound'].to_numpy()

    observed_duration = (lower + upper) / 2
    observed_duration[observed_duration <= 0] = 1 

    # Correct usage: assign directly, do not call as a function, do not use name=
    duration_obs = pm.Censored(
        "duration_obs",  # <<< Name (string)
        pm.LogNormal.dist(mu=pm.math.log(mu), sigma=sigma),
        lower=lower,
        upper=upper,
        observed=observed_duration
    )
    trace = pm.sample(2000, tune=1000, target_accept=0.95, max_treedepth = 15)

    print(az.summary(trace, round_to=2))

    az.plot_posterior(trace, var_names=["alpha", "beta_year", "beta_sex", "sigma_age", "sigma"], hdi_prob=0.95)
    # plt.show()
    plt.savefig('plots/posterior_parameter_plots.png')

    # Save trace to a NetCDF file
    az.to_netcdf(trace, 'models/bayesian_unemployment_trace.nc')


    # # inference

    # logging.info('Inference beginning...')
    # # Example input
    # new_year = 2025
    # new_sex_code = 1  # female
    # new_age_group = "30-39"

    # # Find the index of the age group
    # age_groups = df_sex_age_unemployed_duration['age'].unique()
    # new_age_idx = np.where(age_groups == new_age_group)[0][0]  # <-- use a new variable 'new_age_idx'

    # posterior_samples = az.extract(trace)

    # alpha_samples = posterior_samples['alpha'].values           # shape (8000,)
    # beta_year_samples = posterior_samples['beta_year'].values   # shape (8000,)
    # beta_sex_samples = posterior_samples['beta_sex'].values     # shape (8000,)
    # beta_age_samples = posterior_samples['beta_age'][:, new_age_idx].values  # shape (8000,)
    # sigma_samples = posterior_samples['sigma'].values           # shape (8000,)

    # # Now compute the linear predictor
    # lin_pred_samples = alpha_samples + beta_year_samples * new_year + beta_sex_samples * new_sex_code + beta_age_samples

    # # Get the mean (mu) for the LogNormal distribution
    # mu_samples = np.exp(lin_pred_samples)

    # # Sample from the predictive distribution
    # lognormal_samples = np.random.lognormal(mean=np.log(mu_samples), sigma=sigma_samples, size=len(mu_samples))

    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.hist(lognormal_samples, bins=50, density=True, alpha=0.6, color='g', label='Posterior predictive (with noise)')
    # plt.xlabel('Unemployment Duration (weeks)')
    # plt.ylabel('Density')
    # plt.title('Posterior Predictive Distribution with LogNormal Noise')
    # plt.legend()
    # plt.savefig('plots/posterior_predictive_dist.png')

    # # Calculate 95% CI
    # lower_ci = np.percentile(lognormal_samples, 2.5)
    # upper_ci = np.percentile(lognormal_samples, 97.5)
    # median_prediction = np.median(lognormal_samples)

    # print(f"Predicted unemployment duration (weeks):")
    # print(f"Median: {median_prediction:.2f} weeks")
    # print(f"95% Credible Interval: [{lower_ci:.2f}, {upper_ci:.2f}] weeks")
