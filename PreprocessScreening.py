import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


raw_data = pd.read_csv("/Users/ericching/Documents/GitHub/Rice-Datathon-2024-Submission/training (1).csv")

correlation = raw_data.corr()

correlation = correlation.iloc[25].sort_values(key=abs)

correlation = correlation.loc['proppant_intensity':'OilPeakRate']
print(correlation)
print()
for i in correlation.index:
    print(i + ": " + str(len(raw_data.dropna(subset=[i]))))

full_data = raw_data.dropna(subset=['proppant_intensity','total_proppant','frac_fluid_intensity','true_vertical_depth','bin_lateral_length','gross_perforated_length','total_fluid'])
print("Final, full dataset size: " + str(len(full_data)))
'''
plt.figure()
sns.heatmap(correlation)

plt.show()
'''
