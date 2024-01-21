import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from ChevronDataset import ChevronDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from ChevronModel import ChevronModel
import torch.nn.functional as F

BATCH_SIZE = 200
NUM_EPOCHS = 2500

transform = transforms.ToTensor()


raw_data = pd.read_csv("/Users/ericching/Documents/GitHub/Rice-Datathon-2024-Submission/training (1).csv")

good_columns = ['proppant_intensity','total_proppant','frac_fluid_intensity','true_vertical_depth','bin_lateral_length','gross_perforated_length','total_fluid','OilPeakRate']

full_data = raw_data.dropna(subset=good_columns)

bad_columns = []

for i in full_data.columns:
    if i not in good_columns:
        bad_columns.append(i)

full_data = full_data.drop(axis=1,columns=bad_columns)

# print(full_data.columns)

# full_data = full_data.truncate(after=100)

# print(len(full_data))
train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size

full_set = ChevronDataset(full_data=full_data)

train_set, test_set = random_split(full_set, [train_size, test_size])

train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)


model = ChevronModel()

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    for index, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = model(inputs)

        #loss = criterion(F.normalize(outputs.flatten(), dim=0), F.normalize(labels, dim=0))
        loss = criterion(outputs.flatten(), labels)

        loss.backward()
        optimizer.step()
        if index==10:
            break
    if epoch % 100 == 0:    
        print('Epoch: {} Average Loss: {:.4f}'.format(epoch, loss / BATCH_SIZE))

with torch.no_grad():
    for index, (inputs, labels) in enumerate(test_loader):

        outputs = model(inputs)

        loss = criterion(outputs.flatten(), labels)
        print(loss / BATCH_SIZE)
        break
