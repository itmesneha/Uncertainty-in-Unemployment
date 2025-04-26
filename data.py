import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class UnemploymentSurvivalDataset(Dataset):
    def __init__(self, dataframe_path, censoring_rate=0.3, seed=42, normalize=True):
        """
        Args:
            dataframe_path (str): Path to the CSV file.
            censoring_rate (float): Probability of censoring for '52 and over'.
            seed (int): Random seed for reproducibility.
        """
        self.df = pd.read_csv(dataframe_path)

        # Map duration buckets to numeric midpoints
        self.duration_map = {
            'under 5': 2.5,
            '5 to 9': 7,
            '10 to 14': 12,
            '15 to 19': 17,
            '20 to 24': 22,
            '25 to 29': 27,
            '30 to 39': 35,
            '40 to 51': 45.5,
            '52 and over': 55  # Treat as maximum observed duration
        }
        self.df['duration_numeric'] = self.df['duration'].map(self.duration_map)

        # Encode categorical variables
        self.year_encoder = {v: i for i, v in enumerate(sorted(self.df['year'].unique()))}
        self.qual_encoder = {v: i for i, v in enumerate(sorted(self.df['highest_qualification'].unique()))}
        self.age_encoder = {v: i for i, v in enumerate(sorted(self.df['age'].unique()))}
        self.sex_encoder = {'male': 0, 'female': 1}

        # Apply censoring only for '52 and over'
        np.random.seed(seed)
        self.df['is_censored'] = 0  # Default: not censored
        mask_52_plus = self.df['duration'] == '52 and over'
        self.df.loc[mask_52_plus, 'is_censored'] = np.random.binomial(1, censoring_rate, size=mask_52_plus.sum())

        # Define event flag: 1 = observed, 0 = censored
        self.df['event'] = 1 - self.df['is_censored']

        # Prepare categorical and continuous features
        self.df['year_cat'] = self.df['year'].map(self.year_encoder)
        self.df['qual_cat'] = self.df['highest_qualification'].map(self.qual_encoder)
        self.df['age_cat'] = self.df['age'].map(self.age_encoder)
        self.df['sex_cat'] = self.df['sex'].map(self.sex_encoder)

        # Continuous features (can add more continuous features here if needed)
        self.continuous_features = ['estimated_unemployed']
        self.normalize = normalize
        if self.normalize:
            self.cont_mean = self.df[self.continuous_features].mean().values
            self.cont_std = self.df[self.continuous_features].std().values


    def __len__(self):
        return len(self.df)
    

    def get_category_sizes(self):
        return [
            len(self.year_encoder),
            len(self.qual_encoder),
            len(self.age_encoder),
            len(self.sex_encoder)
        ]

    def get_continuous_feature_count(self):
        return len(self.continuous_features)
    

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Categorical features as separate columns for embedding
        x_cat = torch.tensor([
            row['year_cat'],
            row['qual_cat'],
            row['age_cat'],
            row['sex_cat']
        ], dtype=torch.long)

        # Continuous features as float tensor
        x_cont = torch.tensor([row[feature] for feature in self.continuous_features], dtype=torch.float32)

        if self.normalize:
            x_cont = (x_cont - self.cont_mean) / self.cont_std
            x_cont = x_cont.to(torch.float32)

        duration = torch.tensor(row['duration_numeric'], dtype=torch.float32).unsqueeze(0)
        event = torch.tensor(row['event'], dtype=torch.float32).unsqueeze(0)

        return x_cat, x_cont, duration, event
