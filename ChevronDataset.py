import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch


raw_data = pd.read_csv("/Users/ericching/Documents/GitHub/Rice-Datathon-2024-Submission/training (1).csv")

good_columns = ['proppant_intensity','total_proppant','frac_fluid_intensity','true_vertical_depth','bin_lateral_length','gross_perforated_length','total_fluid','OilPeakRate']

full_data = raw_data.dropna(subset=good_columns)

bad_columns = []

for i in full_data.columns:
    if i not in good_columns:
        bad_columns.append(i)

full_data = full_data.drop(axis=1,columns=bad_columns)

full_data = full_data.truncate(after=100)

class ChevronDataset(Dataset):
    def __init__(self, full_data):
        self.datums = full_data
    def __len__(self):
        return len(self.datums)

    def __getitem__(self, idx):
        label = self.datums.iloc[idx, 7]
        features = self.datums.iloc[idx, :7]
        return torch.tensor(features).float(), torch.tensor(label).float()

dataset = ChevronDataset(full_data=full_data)
