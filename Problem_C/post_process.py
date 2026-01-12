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

    axes[0].scatter(target.T[0][input_data.T[0]>10][0:10000], target.T[1][input_data.T[0]>10][0:10000],c="#d3d3d3")
    err=np.abs((target.T[1]-pred.T[1])>0.1)
    #plt.scatter(target.T[0][err], target.T[1][err],c=np.abs((target.T[1]-pred.T[1]))[err],cmap="viridis")

    axes[0].scatter(unique_conditions[:, 0][mean_errors>0.05], unique_conditions[:, 1][mean_errors>0.05], c=mean_errors[mean_errors>0.05], cmap='viridis', label='Mean Error')
    
    #plt.colorbar(label='Mean Error')
    plt.xlabel('$U_0$')
    plt.ylabel('$W_0$')

    
    
    with h5py.File(file_path + "/val.h5", 'r') as hf:
        A=100000
        target = np.array(hf['target'])[:A]
        pred = np.array(hf['pred'])[:A]
        input_data = np.array(hf['input'])[:A]
    
         
    df = pd.DataFrame({
        'u': input_data.T[1],  # Second row is first solution parameter
        'w': input_data.T[2],  # Third row is second solution parameter
        'up': pred.T[0],  # Second row is first solution parameter
        'wp': pred.T[1],
        'ut': target.T[0],
        'wt': target.T[1],
    })



    # Save the filtered DataFrame
    df.to_csv('filtered_df.csv', index=False)

    
    
    def ode(t, x):
            nP = len(x)
            u_i=x[0]
            v_i=x[1]
            k=0
            dxdt = [10*((1)*(u_i*(u_i-0.4)*(1-u_i))-v_i + k*0.04 + 0.08  ),
                    
                    ((u_i*0.04-0.16*v_i))
                    
                    ]
            return dxdt


    df['xp'] = df.apply(lambda row: (ode(0, [row['u'], row['w'],row["up"],row["wp"]]))[0], axis=1)
    df['yp'] = df.apply(lambda row: (ode(0, [row['u'], row['w'],row["up"],row["wp"]]))[1], axis=1)
    df['derivative_magnitudep'] =df.apply(lambda row: np.linalg.norm(ode(0, [row['u'], row['w'],row["up"],row["wp"]])), axis=1)
    
    
    df['derivative_magnitude'] =df.apply(lambda row: np.linalg.norm(ode(0, [row['u'], row['w'],row["ut"],row["wt"]])), axis=1)
    df['x'] = df.apply(lambda row: (ode(0, [row['u'], row['w'],row["ut"],row["wt"]]))[0], axis=1)
    df['y'] = df.apply(lambda row: (ode(0, [row['u'], row['w'],row["ut"],row["wt"]]))[1], axis=1)
    # Create subplots: 3 rows and 3 columns


    norm_derivative = TwoSlopeNorm(vmin=0, vcenter=df['derivative_magnitude'].mean(), vmax=df['derivative_magnitude'].max())
    # 3rd row: True, Predicted, Error for derivative magnitude
    axes[1].scatter(df['u'], df['w'], c=np.abs(df['derivative_magnitude']), cmap='seismic', norm=norm_derivative, marker='o')
    axes[1].quiver(df['u'], df['w'], df["x"],df["y"])

    axes[1].set_title('True Derivative Magnitude')
    axes[1].set_xlabel('u')
    axes[1].set_ylabel('w')


    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("9_subplots.png")
    
    

    
        
# Call the plot_results function with the path to your HDF5 file
import sys
file_path=sys.argv[1]

dir=file_path
plot_results(file_path)

file_path = "batch_results/simulation1"

dir=file_path
#plot_results(file_path)