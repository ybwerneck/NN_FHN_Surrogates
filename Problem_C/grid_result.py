import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Directory containing subfolders with .h5 files
import statsmodels.api as sm
from statsmodels.formula.api import ols

import os
import h5py
import numpy as np
import pandas as pd
import pickle
import torch
# Directory containing subfolders with .h5 files
# Call the plot_results function with the path to your HDF5 file
import sys

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import time as TIME
class SinusoidalActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
base_dir=sys.argv[1] 

def combine_strings(strings):
    return ', '.join(item.split('.')[-1].upper() for item in strings)
def process_string(s):
    if( s=="<class '__main__.SinusoidalActivation'>"):
        return "SIN"
    return s.split('.')[-1].upper()
data=[]
learning_curves={}
times={}
randoms={}# Iterate through each subfolder
plt.figure(figsize=(12, 6),dpi=100)
S=[]
anova_data = []
# Create subplots: two axes, one for each scatter plot
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))

# First subplot for Mean Error and Max Error scatter plots
for subdir in os.listdir(base_dir):
    print(f" INTO {subdir}")
    if os.path.isdir(os.path.join(base_dir, subdir)) and subdir.startswith('simulation'):
        model_path=os.path.join(base_dir, subdir,"model")
        h5_file_path = os.path.join(base_dir, subdir, 'val_err.h5')
        pickle_file_path = os.path.join(base_dir, subdir, 'my_dict.pkl')
        time_csv_path=os.path.join(base_dir, subdir, 'it_time.csv')

        average_time = 0  # Placeholder value for average time

        ## Error stats
        if os.path.isfile(h5_file_path):
            with h5py.File(h5_file_path, 'r') as f:
                print(model_path)
                learning_curve = np.array(f['error_stats'])
                try:
                    final_value1 = learning_curve.T[0][-1]
                    final_value2 = learning_curve.T[1][-1]
                    
                    learning_curves[subdir] = learning_curve
                except:
                    f = 0
                layers = []

            ## Model info
            with open(pickle_file_path, 'rb') as pf:
                model_info = pickle.load(pf)
                layers = model_info.get('model_params', 'N/A')["hidden_layers"]
                print(layers)
                neuron=np.sum(x[1] for x in model_info.get('model_params', 'N/A')["hidden_layers"])
                layers = combine_strings([f"({process_string(str(x[0]))}-{x[1]})" for x in layers])
                lnid = layers

                try:
                    ra = randoms[layers]
                except:
                    ra = randoms[layers] = np.random.rand((1))

                try:
                    pinn = model_info.get('model_params', 'N/A')['pinn']
                    iccs = model_info.get('model_params', 'N/A')['iccs']
                    bs = model_info.get('model_params', 'N/A')['bs']
                except:
                    print("-")
                    bs = 64

                neuron = np.sum(x[1] for x in model_info.get('model_params', 'N/A')["hidden_layers"])
        
                if iccs:
                    c = 0  # Use valid color names
                else:
                   c = 1
                 
                c1=["red","blue"]
                c2=["lightcoral","lightblue"]
               
                
                # Scatter plot on the first subplot (Mean Error and Max Error)
                ax1.scatter(neuron + iccs * 2, final_value2, color=c2[c], label='Max Error')
                ax1.scatter(neuron + iccs * 2, final_value1, color=c1[c], label='Mean Error')
                
                anova_data.append([iccs, neuron,final_value1, final_value2])
                data.append([subdir, final_value1, final_value2,average_time,neuron, layers,bs,True,pinn])            


# Create a DataFrame and save to CSV
pd.options.display.float_format = '{:.2e}'.format
df = pd.DataFrame(data, columns=['Folder', 'Mean err', 'Max err',"It time","neuron","layes","bs","iccs","pinn"])
df = df.sort_values(by=[ 'neuron',"Mean err"])
df.to_csv(base_dir+'/final_values.csv', index=False)
# Formatting the first subplot

ax1.set_title('Mean Validaton Error')
ax1.set_xlabel('Neuron count')
ax1.set_ylabel('Error')
ax1.set_ylim(1e-3, 1e-3*10000)
ax1.set_xlim(0, 110)


# Second subplot for Model Architecture Type and Number of Layers
scatter0 = ax1.scatter(100, 100, color="#ffffff", label="Bottle neck")
scatter1 = ax1.scatter(100, 100, color="blue", label="Bottle neck")
scatter2 = ax1.scatter(100, 100, color="red", label="Rectangle")

scatter7 = ax1.scatter(100, 100, color="#bfbfbf", label="Three layer Arch")
scatter4 = ax1.scatter(100, 100, color="#262626", label="Three layer Arch")

# First legend in ax2
first_legend = ax1.legend([scatter0, scatter1, scatter2], ["Extra data constraint", "No", "Yes"], loc="upper right")
ax1.add_artist(first_legend)

# Second legend in ax2
second_legend = ax1.legend([scatter7, scatter4], ["Max Err", "Mean Err"], loc="upper left")
ax1.add_artist(second_legend)


ax1.set_yscale("log")
# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(base_dir+"/b_s_s_t.pdf")
df_anova = pd.DataFrame(anova_data, columns=['iccs', 'neuron', 'mean_err','max_err'])

# Fit the ANOVA model: mean_err as the dependent variable, iccs and neuron as independent variables
model = ols('max_err ~ C(iccs)*neuron + C(iccs) + neuron', data=df_anova).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Output the ANOVA results
print("max err")
print(anova_table)

model = ols('mean_err ~ C(iccs)*neuron + C(iccs) + neuron', data=df_anova).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Output the ANOVA results
print("mean err")
print(anova_table)

smallest_model_err_001 = df[df['Mean err'] < 0.01].sort_values(by='neuron').iloc[0]

# Find the smallest model neuron count with Mean err < 0.001
smallest_model_err_0001 = df[df['Mean err'] < 0.1].sort_values(by='neuron').iloc[0]

# Find the best model with neuron count < 64
best_model_lt_64 = df[df['neuron'] <= 32].sort_values(by='Mean err').iloc[0]

# Find the best model with neuron count < 128
best_model_lt_128 = df[df['neuron'] <= 128].sort_values(by='Mean err').iloc[0]

# Display the results
print("Smallest model with Mean err < 0.01:", smallest_model_err_001)
print("Smallest model with Mean err < 0.1:", smallest_model_err_0001)
print("Best model with neuron count < 32:", best_model_lt_64)
print("Best model:", df.sort_values(by='Mean err').iloc[0])