import h5py
import matplotlib.pyplot as plt
import numpy as np
import operator
import csv
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression

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
        target = hf['target'][:]
        pred = hf['pred'][:]
        input_data = hf['input'][:]

    # Calculate the error
    error = np.sum(np.abs(target - pred),axis=1)
    threshold = np.percentile(error, 90)
    
    # Get indices of the top 10% error points
    top_10_percent_indices = np.where(error >= threshold)[0]

    # Extract the corresponding inputs and errors
    top_10_percent_input = input_data[top_10_percent_indices]
    top_10_percent_error = error[top_10_percent_indices]

    # Fit a linear regression model
    X = top_10_percent_input[:, 0].reshape(-1, 1)  # input.T[0]
    y = top_10_percent_input[:, 1]  # input.T[1]
    model = LinearRegression()
    model.fit(X, y)

    # Get the line's slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    # Create line for plotting
    x_line = np.linspace(np.min(X), np.max(X), 100)
    y_line = slope * x_line + intercept

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot all points with their error
    plt.scatter(input_data.T[0], input_data.T[1], c=error, cmap='viridis', label='All Points')
    plt.colorbar(label='Error Magnitude')
    plt.savefig(f"{file_path}/error_map.png")
    
    plt.figure(figsize=(12, 8))
    
    # Plot all points with their error
    plt.scatter(input_data.T[0], input_data.T[1], c=error, cmap='viridis', label='All Points')
    plt.colorbar(label='Error Magnitude')
    
    # Highlight top 10% error points
    plt.scatter(top_10_percent_input[:, 0], top_10_percent_input[:, 1], color='red', label='Top 10% Error')
    
    # Plot the best fit line
    plt.plot(x_line, y_line, color='blue', label=f'Best Fit Line: y = {slope:.2f}x + {intercept:.2f}')

    # Labeling
    plt.xlabel('Input Dimension 1')
    plt.ylabel('Input Dimension 2')
    plt.title('Top 10% Error Points and Best Fit Line')
    plt.legend()
    
    # Save the plot
    plt.savefig(f"{file_path}/top_10_percent_fit.png")
    
# Call the plot_results function with the path to your HDF5 file
import sys

file_path = "batch_results/simulation0"

dir=file_path
plot_results(file_path)

file_path = "batch_results/simulation1"

dir=file_path
plot_results(file_path)