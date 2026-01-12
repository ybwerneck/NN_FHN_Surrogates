import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Directory containing subfolders with .h5 files


import os
import h5py
import numpy as np
import pandas as pd
import pickle
import torch
# Directory containing subfolders with .h5 files
# Call the plot_results function with the path to your HDF5 file
import sys
from scipy import stats


from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import time as TIME

base_dir=sys.argv[1] 

def combine_strings(strings):
    return ', '.join(item.split('.')[-1].upper() for item in strings)
def process_string(s):
    #print(s)
    return s.split('.')[-1].upper()
data=[]
learning_curves={}
times={}
randoms={}# Iterate through each subfolder
plt.figure(figsize=(10, 6))
for subdir in os.listdir(base_dir):

    
    
    S=[]
    print(f" INTO {subdir}")
    if os.path.isdir(os.path.join(base_dir, subdir)) and subdir.startswith('simulation'):
        model_path=os.path.join(base_dir, subdir,"model")
        h5_file_path = os.path.join(base_dir, subdir, 'val_err.h5')
        pickle_file_path = os.path.join(base_dir, subdir, 'my_dict.pkl')

        # Assuming the column with iteration times is named 'it_times'
        # Modify 'it_times' to match the actual column name in your CSV file
        average_time =0


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

                 try:
                    ra=randoms[layers]
                 except:         
                    ra=randoms[layers]=np.random.rand((1))*10 -5

                
                 try:
                    pinn=model_info.get('model_params', 'N/A')['pinn']
                    iccs=model_info.get('model_params', 'N/A')['iccs']
                    bs=model_info.get('model_params', 'N/A')['bs']                    
                 except:
                     print("-")
                     bs=64
                     iccs=False
                 neuron=ra+np.sum(x[1] for x in model_info.get('model_params', 'N/A')["hidden_layers"])
                 mean_err=final_value1

                 if pinn:
                    c = 'red' if not iccs else 'lightcoral'  # Use valid color names
                 else:
                    c = 'blue' if not iccs else 'lightblue'
                 
                 S.append(final_value1)
                 plt.scatter(neuron,S[-1],color=c)

                 plt.ylim(1e-6,1e1)

                 
            times[subdir]=neuron
            data.append([ subdir,layers,final_value1, final_value2,pinn])
# Create a DataFrame and save to CSV
pd.options.display.float_format = '{:.2e}'.format
df = pd.DataFrame(data, columns=["f","layes", 'Mean err', 'Max err',"pinn"])
df = df.sort_values(by=['layes', 'Mean err'])
df.to_csv(base_dir+'/final_values.csv', index=False)

# Assuming df is your DataFrame with columns X, B, and Y
print(df[df['pinn'] == True]['Mean err'])
group1 = df[df['pinn'] == False]['Mean err']
group2 = df[df['pinn'] == True]['Mean err']

# Perform a t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_stat}, P-value: {p_value}")


plt.scatter(100,100,color="red",label="PINN")
plt.scatter(100,100,color="blue",label="DDNN")
#plt.scatter(100,100,color="lightcoral",label="PINN a")
#plt.scatter(100,100,color="lightblue",label="DDNN a" )
plt.xlabel('Model size  (N of neurons) ')
plt.ylabel('Mean Loss ')
plt.yscale('log')
plt.xscale('log')

plt.title('DDNNs vs PINNs loss over scarce training dataset')
plt.legend(loc="best")
#plt.legend()
plt.savefig(base_dir+"/results.pdf")

# Plotting the learning curves with log scale for y-axis


