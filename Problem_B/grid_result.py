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

base_dir=sys.argv[1] 
base_dir_2=sys.argv[2] 
def combine_strings(strings):
    return ', '.join(item.split('.')[-1].upper() for item in strings)
def process_string(s):
    #print(s)
    return s.split('.')[-1].upper()
data=[]
learning_curves={}
times={}
randoms={}# Iterate through each subfolder
plt.figure(figsize=(12, 6),dpi=100)
S=[]

for subdir in os.listdir(base_dir):

    
    
    
    print(f" INTO {subdir}")
    if os.path.isdir(os.path.join(base_dir, subdir)) and subdir.startswith('simulation'):
        model_path=os.path.join(base_dir, subdir,"model")
        h5_file_path = os.path.join(base_dir, subdir, 'val_err.h5')
        pickle_file_path = os.path.join(base_dir, subdir, 'my_dict.pkl')
        time_csv_path=os.path.join(base_dir, subdir, 'it_time.csv')
        # Read the CSV file
        df_time = pd.read_csv(time_csv_path)

        # Assuming the column with iteration times is named 'it_times'
        # Modify 'it_times' to match the actual column name in your CSV file
        average_time = df_time['ITTIME'].mean()


        ##Error stats
        if os.path.isfile(h5_file_path):
            with h5py.File(h5_file_path, 'r') as f:
                print(model_path)
                # Adjust the following line to match the structure of your .h5 file
                learning_curve = np.array(f['error_stats'])
                try:
                    final_value1 = learning_curve.T[0][-1]
                    final_value2 = learning_curve.T[1][-1]
                    
                    learning_curves[subdir]=learning_curve
                except:
                    f=0
                #print(learning_curve)
            layers=[]

    
        ##Model info
            with open(pickle_file_path, 'rb') as pf:
                 model_info = pickle.load(pf)
                 print(model_info)
                 layers = model_info.get('model_params', 'N/A')["hidden_layers"]
                 print(layers)
                # print(layers[0][0])
                 layers=combine_strings( [ "("+str(process_string(str(x[0])))+"-"+str(x[1])+")"  for x in layers]  )
                 lnid=layers
                 if False and layers!="(SILU'>-32), (SILU'>-16)":
                     continue
                 try:
                    ra=randoms[layers]
                 except:         
                    ra=randoms[layers]=np.random.rand((1))

                
                 try:
                    pinn=model_info.get('model_params', 'N/A')['pinn']
                    iccs=model_info.get('model_params', 'N/A')['iccs']
                    bs=model_info.get('model_params', 'N/A')['bs']                    
                 except:
                     print("-")
                     bs=64
                
                 lyr=[x[1] for x in model_info.get('model_params', 'N/A')["hidden_layers"]]
                 neuron=np.sum(x[1] for x in model_info.get('model_params', 'N/A')["hidden_layers"])
                 mean_err=final_value1
                 bs=bs
                                  
# Predefined palette of blue shades
                 blue_palette = ["#add8e6",  # Light Blue
                                "#87CEEB",  # Sky Blue
                                "#4682B4",  # Steel Blue
                                "#4169E1",  # Royal Blue
                                "#0000FF",  # Blue
                                "#0000CD",  # Medium Blue
                                "#00008B",  # Dark Blue
                                "#000080"]  # Navy

                # Map neuron values to the palette
                 if True:
                     c="blue"
                 else:
                     c="red"
                                
                 if(pinn==True and iccs==True):
                    S.append(mean_err)
                 plt.scatter(neuron +iccs*2,mean_err,color=c)

                 plt.ylim(1e-4,1e1)
                 #plt.xlim(20,200)

                 
            times[subdir]=neuron
            print("aaa",len(lyr))
            if(len(lyr)<0):
                continue
            data.append([subdir, mean_err, final_value2,average_time,neuron, layers,bs,False,pinn])
            
for subdir in os.listdir(base_dir_2):
    base_dir=base_dir_2
    
    
    
    print(f" INTO {subdir}")
    if os.path.isdir(os.path.join(base_dir, subdir)) and subdir.startswith('simulation'):
        model_path=os.path.join(base_dir, subdir,"model")
        h5_file_path = os.path.join(base_dir, subdir, 'val_err.h5')
        pickle_file_path = os.path.join(base_dir, subdir, 'my_dict.pkl')
        time_csv_path=os.path.join(base_dir, subdir, 'it_time.csv')
        # Read the CSV file
        df_time = pd.read_csv(time_csv_path)

        # Assuming the column with iteration times is named 'it_times'
        # Modify 'it_times' to match the actual column name in your CSV file
        average_time = df_time['ITTIME'].mean()


        ##Error stats
        if os.path.isfile(h5_file_path):
            with h5py.File(h5_file_path, 'r') as f:
                print(model_path)
                # Adjust the following line to match the structure of your .h5 file
                learning_curve = np.array(f['error_stats'])
                try:
                    final_value1 = learning_curve.T[0][-1]
                    final_value2 = learning_curve.T[1][-1]
                    
                    learning_curves[subdir]=learning_curve
                except:
                    f=0
                #print(learning_curve)
            layers=[]

    
        ##Model info
            with open(pickle_file_path, 'rb') as pf:
                 model_info = pickle.load(pf)
                 print(model_info)
                 layers = model_info.get('model_params', 'N/A')["hidden_layers"]
                 print(layers)
                # print(layers[0][0])
                 layers=combine_strings( [ "("+str(process_string(str(x[0])))+"-"+str(x[1])+")"  for x in layers]  )
                 lnid=layers
                 if False and layers!="(SILU'>-32), (SILU'>-16)":
                     continue
                 try:
                    ra=randoms[layers]
                 except:         
                    ra=randoms[layers]=np.random.rand((1))

                
                 try:
                    pinn=model_info.get('model_params', 'N/A')['pinn']
                    iccs=model_info.get('model_params', 'N/A')['iccs']
                    bs=model_info.get('model_params', 'N/A')['bs']                    
                 except:
                     print("-")
                     bs=64
                
                 lyr=[x[1] for x in model_info.get('model_params', 'N/A')["hidden_layers"]]
                 neuron=np.sum(x[1] for x in model_info.get('model_params', 'N/A')["hidden_layers"])
                 mean_err=final_value1
                 bs=bs
                                  
# Predefined palette of blue shades
                 blue_palette = ["#add8e6",  # Light Blue
                                "#87CEEB",  # Sky Blue
                                "#4682B4",  # Steel Blue
                                "#4169E1",  # Royal Blue
                                "#0000FF",  # Blue
                                "#0000CD",  # Medium Blue
                                "#00008B",  # Dark Blue
                                "#000080"]  # Navy

                # Map neuron values to the palette
                 if False:
                     c="blue"
                 else:
                     c="red"
                                
                 if(pinn==True and iccs==True):
                    S.append(mean_err)
                 plt.scatter(neuron,mean_err,color=c)

                 plt.ylim(1e-4,1e0)
                 #plt.xlim(20,200)

                 
            times[subdir]=neuron
            print("aaa",len(lyr))
            if(len(lyr)<0):
                continue
            data.append([subdir, mean_err, final_value2,average_time,neuron, layers,bs,True,pinn])            
            
 
# Create a DataFrame and save to CSV
pd.options.display.float_format = '{:.2e}'.format
df = pd.DataFrame(data, columns=['Folder', 'Mean err', 'Max err',"It time","neuron","layes","bs","iccs","pinn"])
df = df.sort_values(by=[ 'neuron',"Mean err"])
df.to_csv(base_dir+'/final_values.csv', index=False)
mean_err_stats_df = df.groupby('iccs')['Mean err'].agg(['mean', 'std', 'min','max']).reset_index()
print(mean_err_stats_df)
plt.xlabel('Training batch size')
plt.ylabel('Mean Loss')
plt.xscale('log')

# First set of scatter points (for the first legend)
scatter0 = plt.scatter(100, 100, color="red", label="")
scatter1 = plt.scatter(100, 100, color="blue", label="Bottle neck")

scatter5 = plt.scatter(100, 100, color="#ffff80", label="Diammond")
# First legend (upper right) - this will only include "Increasing", "Decreasing", and "Same"
first_legend = plt.legend([scatter0,scatter1], ["Small Time Domain","Large Time Domain"], loc="upper right")
plt.gca().add_artist(first_legend)

# Show the plot
plt.show()
#plt.legend()

print(df)

df = df.select_dtypes(include=[np.number,bool])  # Select only numeric columns

#df['pinn']=df['pinn']*2 + df['iccs']

plt.title('Model Accuracy for Models Handling Time Domains of Different Sizes')
plt.xlabel('Model size  (N of neurons) ')

plt.yscale('log')  # Optional: Use log scale for better visualization of the error
plt.savefig(base_dir+"/b_s_s_t.pdf")
print(mean_err_stats_df)

model = ols('Q("Mean err") ~ neuron', data=df).fit()

# Print the summary of the ANCOVA results
print(model.summary())
anova_table = sm.stats.anova_lm(model, typ=1)
print(anova_table)
print(model.params)



df_neuron_above_60 = df[df['neuron'] > 120]
df_neuron_60_or_less = df[df['neuron'] <= 120]

df=df_neuron_60_or_less
print(df_neuron_60_or_less)
# Perform ANCOVA for neurons > 60
model_above_60 = ols('Q("Mean err") ~ C(iccs)*neuron + C(iccs)', data=df_neuron_above_60).fit()

# Perform ANCOVA for neurons <= 60
model_60_or_less = ols('Q("Mean err") ~ C(iccs)*neuron + C(iccs)', data=df_neuron_60_or_less).fit()

# Print summaries for both models
print("ANCOVA for neurons > 120")
print(sm.stats.anova_lm(model_above_60, typ=1))

print("\nANCOVA for neurons <= 120")
print(sm.stats.anova_lm(model_60_or_less, typ=1))


import numpy as np
import pandas as pd

# Sample DataFrame creation (replace this with your actual DataFrame)
# df = pd.read_csv('your_data.csv')

# Initialize variables to store group means, variances, and counts
group_means = []
group_variances = []
group_counts = []

# Loop through each group defined by 'iccs'
for group, data in df.groupby('iccs'):
    mean_err = data['Mean err'].mean()  # Mean of Mean err for the group
    variance_err = data['Mean err'].var(ddof=1)  # Variance of Mean err for the group
    count = data.shape[0]  # Number of observations in the group
    
    # Store mean, variance, and count
    group_means.append(mean_err)
    group_variances.append(variance_err)
    group_counts.append(count)



# Calculate the overall mean of Y
overall_mean = df['Mean err'].mean()

# Calculate intra-group variance (within-group variance)
intra_group_variance = sum(variance * (count - 1) for variance, count in zip(group_variances, group_counts)) / (df.shape[0] - len(group_counts))

# Calculate extra-group variance (between-group variance)
extra_group_variance = sum(count * (mean - overall_mean) ** 2 for mean, count in zip(group_means, group_counts)) / (len(group_means) - 1)

# Print the results
print(f'Intra-group variance (within-group variance): {intra_group_variance}')
print(f'Extra-group variance (between-group variance): {extra_group_variance}')
