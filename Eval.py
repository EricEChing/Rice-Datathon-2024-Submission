import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from ChevronDataset import ChevronDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from ChevronModel import ChevronModel


raw_data = pd.read_csv("scoring.csv")

good_columns = ['proppant_intensity','total_proppant','frac_fluid_intensity','true_vertical_depth','bin_lateral_length','gross_perforated_length','total_fluid']

full_data = raw_data.dropna(subset=good_columns)

bad_columns = []

for i in full_data.columns:
    if i not in good_columns:
        bad_columns.append(i)

full_data = full_data.drop(axis=1,columns=bad_columns)

full_data['EVAL'] = pd.Series(range(len(full_data)))

model = ChevronModel()

model.load_state_dict(torch.load("/Users/ericching/Documents/GitHub/Rice-Datathon-2024-Submission/ChevronModel.pth"))

model.eval()

full_set = ChevronDataset(full_data=full_data)
loader = DataLoader(full_set,batch_size=1,shuffle=False)

resulting_data = {1: "OilPeakRate"}

for index, (features,_) in enumerate(loader):
    with torch.inference_mode():
        output = model(features)
        resulting_data[index+2] = float(abs(output))

resulting_data = pd.Series(data=resulting_data)
print(resulting_data)
resulting_data.to_excel('results.xlsx')