import numpy as np 
import pandas as pd

raw_data = pd.read_csv("/Users/ericching/Documents/GitHub/Rice-Datathon-2024-Submission/training (1).csv")

#print(raw_data.size)
# x = raw_data.iloc[:,0:30]
# y = raw_data.iloc[:,30]

print(raw_data.corr())

