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

# Find the 90% of the max error value
    error_threshold =0.00

    # Filter the DataFrame where error > 90% of the max error
    df = df[df['error'] > error_threshold]

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
    
    

    
    plt.figure()
    c=501
    first=True
    for n in range(0,A//c):

        C = "red" if (input_data[n*c][1:] in top_10_percent_input.T[1:].T) else "#d3d3d3"  # Light grey color
        u = target[c * n:c * (n + 1)].T[0]
        uc = pred[c * n:c * (n + 1)].T[0]
        # Plot with conditional transparency and linestyle for light grey lines
        if C == "#d3d3d3":
            plt.plot(np.linspace(0,20,c),u, c=C, linestyle=':', alpha=0.8)  # Faded and dotted for grey

    for n in range(0,A//c):

        C = "red" if (input_data[n*c][1:] in top_10_percent_input.T[1:].T) else "#d3d3d3"  # Light grey color
        u = target[c * n:c * (n + 1)].T[0]
        uc = pred[c * n:c * (n + 1)].T[0]
        # Plot with conditional transparency and linestyle for light grey lines
        if C != "#d3d3d3":
            plt.plot(np.linspace(0,20,c),u, c=C, linestyle=':', alpha=0.8)  # Faded and dotted for grey
            if(first):
                plt.plot(np.linspace(0,20,c),u,"--", c=C)
                plt.plot(np.linspace(0,20,c),uc, c=C,label="High error solutions")
                
                # Normal line for red
                first=False
            else:
                print("A",input_data[n*c][1:]) 
                plt.plot(np.linspace(0,20,c),u,"--", c=C)
                plt.plot(np.linspace(0,20,c),uc, c=C)
        

# Normal line for red

    plt.legend(loc="best")
    plt.xlabel("$t$")
    plt.ylabel("$U$")
    plt.savefig(dir+"/trouble_solution.pdf", bbox_inches='tight')


    # Get the line's slope and intercept
 #   slope = model.coef_[0]
 #   intercept = model.intercept_

    # Create line for plotting
 #   x_line = np.linspace(np.min(X), np.max(X), 100)
 #   y_line = slope * x_line + intercept

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot all points with their error
 
    y = input_data.T[2] 
    K=input_data.T[3] 

    # Assume input_data, pred, target, and error are already defined

    # Set figure size
    fig, axs = plt.subplots(3, 3, figsize=(20, 15), subplot_kw={'projection': '3d'})

    # Define the different Y slices (change values according to your data)
    y_slices = [0.0, y[error==np.max(error)][0], 0.10]  # Example Y values to slice, modify these as needed

    # Loop over the y_slices and generate the plots
  
    for i, y_value in enumerate(y_slices):
        k = (y > y_value - 0.01) & (y < y_value + 0.01)  &(np.abs(K-K[error==np.max(error)][0])<0.1)# Narrow window for each Y slice

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
                                                        (grid_err, 'Error', 0, 1)]):
 
            ax = axs[i, j]
            surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_zlim(vmin, vmax)
            ax.set_xlim(0,20)
            ax.set_ylim(np.min(U0), np.max(U0))
            ax.set_xlabel('T')
            ax.set_ylabel('$U_0$')
            ax.set_zlabel(zlabel)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Adjust layout
    plt.tight_layout()
    fig.text(-0.0, 1.1 - ( 1* 0.3) + 0.02,  f"¨$V_0$ = {y_slices[0]}", va='center', ha='center', rotation=90, fontsize=16)
    fig.text(-0.0, 1.1 - (2 * 0.3),  f"¨$V_0$ = {y_slices[1]}", va='center', ha='center', rotation=90, fontsize=16)
    fig.text(-0.0, 1.1 - (3 * 0.3) -0.05,  f"¨$V_0$ = {y_slices[2]}", va='center', ha='center', rotation=90, fontsize=16)

    # Save and show the figure
    plt.savefig(dir+"/3x3gridplot.pdf", bbox_inches='tight')
    plt.show()

        
# Call the plot_results function with the path to your HDF5 file
import sys
file_path=sys.argv[1]

dir=file_path
plot_results(file_path)

file_path = "batch_results/simulation1"

dir=file_path
#plot_results(file_path)