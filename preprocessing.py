import requests
import pandas as pd
import numpy as np
import ipfn
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Starting preprocessing...")


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


#age with unemployment duration
dataset_id_1 = "d_db95e15ceffaa368a043310479dc7d57"
df_age_duration = get_data(dataset_id_1, 2000)
logging.info('Age duration dataset loading complete.')

#highest education with unemployment duration
dataset_id_2 = "d_a0ca632fd1d6ff841f0e47298a9ab589"
df_qual_duration = get_data(dataset_id_2, 2000)
logging.info('Qualification duration dataset loading complete.')

#median duration of unemployment
dataset_id_3 = "d_c01a3210fb10f1a52676f97498d4ec2c"
median_duration = get_data(dataset_id_3, 2000)
logging.info('Median duration dataset loading complete.')

logging.info('Sample of df_age_duration:')
logging.info(df_age_duration.head())

logging.info('Sample of df_qual_duration:')
logging.info(df_qual_duration.head())

logging.info('Sample of median_duration:')
logging.info(median_duration.head())

logging.info('Unique durations found in df_age_duration:')
logging.info(df_age_duration["duration"].unique())

df_age_duration[['unemployed']] = df_age_duration[['unemployed']].apply(pd.to_numeric, errors='coerce')
df_qual_duration[['unemployed']] = df_qual_duration[['unemployed']].apply(pd.to_numeric, errors='coerce')

# getting duration midpoint
def create_duration_midpoints(df):
    logging.info("Creating duration midpoints...")
    midpoints = {}
    for category in df["duration"].unique():
        if category == "under 5":
            midpoint = 2.5
        elif category == "52 and over":
            midpoint = 52 + (104 - 52) / 2
        else:
            lower, upper = map(int, category.split(" to "))
            midpoint = (lower + upper) / 2
        midpoints[category] = midpoint
    logging.info(f"Duration midpoints created: {midpoints}")
    return midpoints

# Marginal 1: age × sex
logging.info("Computing age × sex marginal...")
age_sex_year_marginal = (
    df_age_duration.groupby(['year', 'age', 'sex'])['unemployed'].sum().reset_index()
)
logging.info(f"Computed age × sex marginal with shape: {age_sex_year_marginal.shape}")

# Marginal 2: qualification × sex
logging.info("Computing qualification × sex marginal...")
qual_sex_year_marginal = (
    df_qual_duration.groupby(['year', 'highest_qualification', 'sex'])['unemployed'].sum().reset_index()
)
logging.info(f"Computed qualification × sex marginal with shape: {qual_sex_year_marginal.shape}")

joint_tables = []

for year in df_age_duration['year'].unique():
    logging.info(f"Running IPF for year {year}...")

    age_marginal = age_sex_year_marginal[age_sex_year_marginal['year'] == year]
    qual_marginal = qual_sex_year_marginal[qual_sex_year_marginal['year'] == year]

    age_pivot = age_marginal.pivot(index='age', columns='sex', values='unemployed').fillna(0)
    age_marg = age_pivot.values
    age_labels = age_pivot.index.tolist()

    qual_pivot = qual_marginal.pivot(index='highest_qualification', columns='sex', values='unemployed').fillna(0)
    qual_marg = qual_pivot.values
    qualification_labels = qual_pivot.index.tolist()

    sex_labels = qual_pivot.columns.tolist()

    initial_guess = np.ones((qual_marg.shape[0], age_marg.shape[0], qual_marg.shape[1]))

    aggregates = [qual_marg, age_marg]
    dimensions = [[0, 2], [1, 2]]

    ipf_fitter = ipfn.ipfn.ipfn(initial_guess, aggregates, dimensions)
    fitted_joint = ipf_fitter.iteration()
    logging.info(f"IPF for year {year} complete with joint shape {fitted_joint.shape}")

    joint_tables.append((year, fitted_joint))

# Duration probabilities from age data
logging.info("Computing duration probabilities from age data...")
duration_counts_age = df_age_duration.groupby(['year', 'age', 'sex', 'duration'])['unemployed'].sum().reset_index()
total_age = duration_counts_age.groupby(['year', 'age', 'sex'])['unemployed'].sum().reset_index().rename(columns={'unemployed': 'total_unemployed'})
duration_probs_age = duration_counts_age.merge(total_age, on=['year', 'age', 'sex'])
duration_probs_age['probability_age'] = duration_probs_age['unemployed'] / duration_probs_age['total_unemployed']
duration_probs_age = duration_probs_age.rename(columns={'unemployed': 'unemployed_age'})
duration_probs_age = duration_probs_age[['year', 'age', 'sex', 'duration', 'unemployed_age', 'probability_age']]

# Duration probabilities from qualification data
logging.info("Computing duration probabilities from qualification data...")
duration_counts_qual = df_qual_duration.groupby(['year', 'highest_qualification', 'sex', 'duration'])['unemployed'].sum().reset_index()
total_qual = duration_counts_qual.groupby(['year', 'highest_qualification', 'sex'])['unemployed'].sum().reset_index().rename(columns={'unemployed': 'total_unemployed'})
duration_probs_qual = duration_counts_qual.merge(total_qual, on=['year', 'highest_qualification', 'sex'])
duration_probs_qual['probability_qual'] = duration_probs_qual['unemployed'] / duration_probs_qual['total_unemployed']
duration_probs_qual = duration_probs_qual.rename(columns={'unemployed': 'unemployed_qual'})
duration_probs_qual = duration_probs_qual[['year', 'highest_qualification', 'sex', 'duration', 'unemployed_qual', 'probability_qual']]

logging.info("Merging age-based and qualification-based probabilities...")
hybrid_probs = duration_probs_age.merge(
    duration_probs_qual,
    left_on=['year', 'sex', 'duration'],
    right_on=['year', 'sex', 'duration'],
    how='outer'
)

hybrid_probs['probability_age'] = hybrid_probs['probability_age'].fillna(0)
hybrid_probs['probability_qual'] = hybrid_probs['probability_qual'].fillna(0)

w = 0.5

hybrid_probs['hybrid_probability'] = (
    w * hybrid_probs['probability_age'] + (1 - w) * hybrid_probs['probability_qual']
)
logging.info("Hybrid probability computed")

joint_results = []

for i, (year, fitted_joint) in enumerate(joint_tables):
    logging.info(f"Processing joint counts for year {year}...")
    for q_idx, qual in enumerate(qualification_labels):
        for a_idx, age in enumerate(age_labels):
            for s_idx, sex in enumerate(sex_labels):
                joint_results.append({
                    'year': year,
                    'highest_qualification': qual,
                    'age': age,
                    'sex': sex,
                    'joint_probability_qual_age_sex': fitted_joint[q_idx, a_idx, s_idx]
                })

joint_df = pd.DataFrame(joint_results)
logging.info(f"Constructed joint DataFrame with shape {joint_df.shape}")

merged_df = joint_df.merge(hybrid_probs, on=['year', 'age', 'sex', 'highest_qualification'], how='left')

missing_probs = merged_df['hybrid_probability'].isna().sum()
if missing_probs > 0:
    logging.warning(f"{missing_probs} rows have missing probability values")

prob_sums = merged_df.groupby(['year', 'age', 'sex', 'highest_qualification'])['hybrid_probability'].sum()
if not np.allclose(prob_sums, 1.0, rtol=0.01):
    logging.warning("Probabilities don't sum to 1 for some groups")

merged_df['estimated_unemployed'] = merged_df['joint_probability_qual_age_sex'] * merged_df['hybrid_probability']

# Convert to integer
merged_df['estimated_unemployed'] = merged_df['estimated_unemployed'].round().astype(int)

logging.info("Final merged DataFrame preview:")
logging.info(merged_df.tail(100))

logging.info(f"Final shape: {merged_df.shape}")
logging.info(f"Final columns: {merged_df.columns.tolist()}")

logging.info('Preprocessing complete.')
merged_df.to_csv("datasets/unemployment_survival_data.csv", index=False)
logging.info('Data saved to datasets/unemployment_survival_data.csv')

# Select only numeric columns for correlation
numeric_cols = merged_df.select_dtypes(include=[np.number])
corr = numeric_cols.corr()

# plt.figure(figsize=(8,6))
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix of Numeric Variables (after IPF)")
# plt.xticks(rotation=0)
# plt.show()

# Example: mean estimated_unemployed by age and sex
pivot = merged_df.pivot_table(index='age', columns='sex', values='estimated_unemployed', aggfunc='mean')
ax = pivot.plot(kind='bar', figsize=(10,6))
plt.title("Mean Estimated Unemployed by Age and Sex")
plt.ylabel("Mean Estimated Unemployed")
plt.xticks(rotation=0)
# plt.show()
plt.savefig('plots/MeanEstimatedUnemployedByAgeAndSex.png')