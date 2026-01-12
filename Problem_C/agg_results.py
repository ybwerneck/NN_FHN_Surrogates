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

base_dir=sys.argv[1] 

def combine_strings(strings):
    return ', '.join(item.split('.')[-1].upper() for item in strings)
def process_string(s):
    #print(s)
    return s.split('.')[-1].upper()
# Initialize a list to store final values and folder names
data = []
learning_curves={}

# Iterate through each subfolder
for subdir in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, subdir)) and subdir.startswith('simulation'):
        h5_file_path = os.path.join(base_dir, subdir, 'val_err.h5')
        pickle_file_path = os.path.join(base_dir, subdir, 'my_dict.pkl')

        if os.path.isfile(h5_file_path):
            with h5py.File(h5_file_path, 'r') as f:
                print(subdir)
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
            with open(pickle_file_path, 'rb') as pf:
                 model_info = pickle.load(pf)
                 print(model_info)
                 layers = model_info.get('model_params', 'N/A')["hidden_layers"]
                 print(layers)
                # print(layers[0][0])
                 layers=combine_strings( [ "("+str(process_string(str(x[0])))+"-"+str(x[1])+")"  for x in layers]  )
                 print(layers)
            
            data.append([subdir, final_value1, final_value2, layers])
# Create a DataFrame and save to CSV
df = pd.DataFrame(data, columns=['Folder', 'Final Value 1', 'Final Value 2',"info"])
df = df.sort_values(by='Final Value 2')
df.to_csv(base_dir+'/final_values.csv', index=False)

# Plotting the learning curves with log scale for y-axis
plt.figure(figsize=(10, 6))
for sim, curve in learning_curves.items():
   
   # print(curve)
    plt.plot(curve.T[0], label=f'{sim} (train)')
    plt.plot(curve.T[1], "--", label=f'{sim} (val)')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Learning Curves Comparison (Log Scale)')
#plt.legend()
plt.savefig(base_dir+"/batch_results.png")
plt.show()
