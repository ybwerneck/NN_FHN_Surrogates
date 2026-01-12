import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import numpy as np
import operator
import csv
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
import pandas as pd
def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header=next(csvreader)  # Skip the header row if it exists
        next
        for row in csvreader:
            data.append([float(val) for val in row])
    return data,header

def plot_losses(data,header):
        iterations = [row[0] for row in data]
        num_losses = len(data[0]) - 1
        plt.figure(figsize=(10, 6))
        plt.yscale('log')
        plt.ylim(1e-10, 1e2)
        
        for i in range(1, num_losses + 1):
            loss_values = [row[i] for row in data]
            plt.plot(iterations, loss_values, label=header[i])
        plt.title('Losses over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir+"/losses.png")

def plot_results(file_path):
    with h5py.File(file_path + "/val.h5", 'r') as hf:
        A=len(hf["pred"])
        target = np.array(hf['target'])[:A]
        pred = np.array(hf['pred'])[:A]
        input_data = np.array(hf['input'])[:A]
    # Calculate the error
    error = np.sum(np.abs(target - pred),axis=1)

  #  X = top_10_percent_input[:, 0].reshape(-1, 1)  # input.T[0]
  #  y = top_10_percent_input[:, 1]  # input.T[1]
  #  model = LinearRegression()
  #  model.fit(X, y)
  # Step 1: Extract initial conditions (first two columns)

# Assuming input_data has the shape [n, 3], where the first column is time, and the second and third are ICs (initial conditions).
# The 'error' array contains error values corresponding to time steps and initial conditions.

# Step 1: Extract initial conditions (columns 1 and 2)
    initial_conditions = input_data[:, 1:3]
    print(initial_conditions)
    # Step 2: Create a unique set of initial conditions
    unique_conditions, inverse_indices = np.unique(initial_conditions, axis=0, return_inverse=True)
    print(len(unique_conditions))
    # Step 3: Aggregate the errors by averaging them for each unique initial condition
    mean_errors = np.zeros(len(unique_conditions))
    for i in range(len(unique_conditions)):
        mean_errors[i] = error[inverse_indices == i].mean()

    # Step 4: Now, plot the unique initial conditions against the mean errors
    fig, axes = plt.subplots(1, 2 ,figsize=(18, 12), dpi=100)

    axes[0].scatter(target.T[0][input_data.T[0]>20][0:10000], target.T[1][input_data.T[0]>20][0:10000],c="#d3d3d3")
    err=np.abs((target.T[1]-pred.T[1])>0.1)
    #plt.scatter(target.T[0][err], target.T[1][err],c=np.abs((target.T[1]-pred.T[1]))[err],cmap="viridis")

    axes[0].scatter(unique_conditions[:, 0][mean_errors>0.05], unique_conditions[:, 1][mean_errors>0.05], c=mean_errors[mean_errors>0.05], cmap='viridis', label='Mean Error')
    
    #plt.colorbar(label='Mean Error')
    plt.xlabel('$U_0$')
    plt.ylabel('$W_0$')

    
    
    
    
         
    df = pd.DataFrame({
        'u': input_data.T[1],  # Second row is first solution parameter
        'w': input_data.T[2],  # Third row is second solution parameter
        'up': pred.T[0],  # Second row is first solution parameter
        'wp': pred.T[1],
        'ut': target.T[0],
        'wt': target.T[1],
        'error': error  # Error data
    })
    plt.figure(figsize=(12, 8))
    
    # Plot all points with their error
 
    y = input_data.T[2] 
    
    K=input_data.T[2]*0.1 

    # Assume input_data, pred, target, and error are already defined

    # Set figure size
    fig, axs = plt.subplots(1,3, figsize=(10,6), subplot_kw={'projection': '3d'},dpi=100)

    # Define the different Y slices (change values according to your data)
    y_slices = [0.10]  # Example Y values to slice, modify these as needed

    # Loop over the y_slices and generate the plots
  
    for i, y_value in enumerate(y_slices):
        k = ((y > y_value - 0.01) & (y < y_value + 0.01))
        #k[100:]=False# Narrow window for each Y slice
        #k[:100]=True# Narrow window for each Y slice

        # Select the data based on the current Y slice
        X = input_data.T[0][k]
        U0 = input_data.T[1][k]  
        U = pred.T[0][k]
        W_t = target.T[0][k]  
        err = error[k]

        # Define grids for the interpolation
        grid_x, grid_y = np.mgrid[min(X):max(X):100j, min(U0):max(U0):100j]
        grid_pred = griddata((X, U0), U, (grid_x, grid_y), method='linear')
        grid_true = griddata((X, U0), W_t, (grid_x, grid_y), method='linear')
        grid_err = griddata((X, U0), np.abs(W_t - U), (grid_x, grid_y), method='linear')
       
        # Plot the predictions, true values, and error for the current Y slice
        for j, (grid_z, zlabel, vmin, vmax) in enumerate([(grid_pred, 'Pred U', -0.1, 1),
                                                        (grid_true, 'True U', -0.1, 1),
                                                        (grid_err, 'Error', 0, 0.15)]):
 
            ax = axs[j]
            surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap= 'viridis' if zlabel !='Error' else 'coolwarm', vmin=vmin, vmax=vmax)
            ax.set_zlim(vmin, vmax)
            #ax.set_xlim(0,20)
            ax.tick_params(axis='both', labelsize=22, label1On=False)
            if(zlabel=='Error'):
                ax.set_zlim(0,1)
            
            ax.set_ylim(np.min(U0), np.max(U0))
            ax.set_xlabel('T', fontsize=22)
            ax.set_ylabel('$U_0$', fontsize=22)
            if(j==0):
                ax.set_zlabel('$U$', fontsize=22)

            
    # Adjust layout
    plt.tight_layout()
 #   fig.text(-0.0, 1.1 - ( 1* 0.3) + 0.02,  f"¨$V_0$ = {y_slices[0]}", va='center', ha='center', rotation=90, fontsize=16)
#    fig.text(-0.0, 1.1 - (2 * 0.3),  f"¨$V_0$ = {y_slices[1]}", va='center', ha='center', rotation=90, fontsize=16)
#    fig.text(-0.0, 1.1 - (3 * 0.3) -0.05,  f"¨$V_0$ = {y_slices[2]}", va='center', ha='center', rotation=90, fontsize=16)

    # Save and show the figure
    plt.savefig(dir+"/3x3gridplot.pdf")

        
# Call the plot_results function with the path to your HDF5 file
import sys
file_path=sys.argv[1]

dir=file_path
plot_results(file_path)


#plot_results(file_path)
    
   