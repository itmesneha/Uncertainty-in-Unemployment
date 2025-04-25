import torch
from torch.utils.data import Dataset

class UnemploymentSurvivalDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.copy()

        self.duration_map = {
            'under 5': 2.5,
            '5 to 9': 7,
            '10 to 14': 12,
            '15 to 19': 17,
            '20 to 24': 22,
            '25 to 29': 27,
            '30 to 39': 35,
            '40 to 51': 45.5,
            '52 and over': 55
        }
        self.df['duration_numeric'] = self.df['duration'].map(self.duration_map)

        # Encode categorical variables
        self.year_encoder = {v: i for i, v in enumerate(sorted(self.df['year'].unique()))}
        self.qual_encoder = {v: i for i, v in enumerate(sorted(self.df['highest_qualification'].unique()))}
        self.age_encoder = {v: i for i, v in enumerate(sorted(self.df['age'].unique()))}
        self.sex_encoder = {'male': 0, 'female': 1}

        self.df['event'] = 1  # Assume observed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        year = self.year_encoder[row['year']]
        qual = self.qual_encoder[row['highest_qualification']]
        age = self.age_encoder[row['age']]
        sex = self.sex_encoder[row['sex']]

        x = torch.tensor([year, qual, age, sex], dtype=torch.float32)
        duration = torch.tensor(row['duration_numeric'], dtype=torch.float32).unsqueeze(0)
        event = torch.tensor(row['event'], dtype=torch.float32).unsqueeze(0)
        weight = torch.tensor(row['estimated_unemployed'], dtype=torch.float32).unsqueeze(0)  # ✅ add weight here

        return x, duration, event, weight
