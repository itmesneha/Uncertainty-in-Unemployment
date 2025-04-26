import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import sys
import requests
import arviz as az

# Import check_model_fit function
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from bayesian_check_model_fit import check_model_fit

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Starting model fit checking script...")

# Load training dataset
dataset_id = "d_db95e15ceffaa368a043310479dc7d57"

def get_data(dataset_id, sample=None):
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
    return full_df

df = get_data(dataset_id, sample=2000)
logging.info('Loaded original training dataset.')

# Normalize year using saved year_mean
year = df['year'].astype(float).to_numpy()
year_mean = np.load('models/year_mean.npy')  # Load year_mean saved during training
year = year - year_mean

# Process the dataset to create observed durations
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

lower = df['lower_bound'].to_numpy()
upper = df['upper_bound'].to_numpy()

# Clip to avoid log(0)
lower = np.clip(lower, a_min=1e-3, a_max=None)
upper = np.clip(upper, a_min=1e-3, a_max=None)

# Compute midpoint
midpoint = (lower + upper) / 2
midpoint = np.clip(midpoint, a_min=1e-3, a_max=None)

# Log transform the midpoint for consistency with training
observed_duration_log = np.log(midpoint)

# Load saved posterior predictive samples (already in log space!)
posterior_predictive_samples = np.load('models/lognormal_samples.npy')  

# No exp!
predicted_samples = np.median(posterior_predictive_samples, axis=0)

# Process observed durations
lower = np.clip(lower, a_min=1e-3, a_max=None)
upper = np.clip(upper, a_min=1e-3, a_max=None)
midpoint = (lower + upper) / 2
midpoint = np.clip(midpoint, a_min=1e-3, a_max=None)
observed_durations = np.log(midpoint)  # in log-weeks

# Now compare observed_durations and predicted_samples directly
all_pred_samples = posterior_predictive_samples.flatten()
observed_durations_weeks = np.exp(observed_durations)
check_model_fit(observed_durations_weeks, all_pred_samples, plot_path='plots/ppc_plot.png')

logging.info('Model fit checking complete. Posterior Predictive Check plot saved to plots/ppc_plot.png')

# Load your trace
trace = az.from_netcdf('models/bayesian_unemployment_trace.nc')

# Compute WAIC
waic = az.waic(trace)
print("WAIC:")
print(waic)

# Compute LOO
loo = az.loo(trace)
print("\nLOO:")
print(loo)