import h5py
import matplotlib.pyplot as plt
import numpy as np
import operator
import csv

def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header row
        for row in csvreader:
            data.append([float(val) for val in row])
    return data, header

def plot_losses(data, header, output_dir):
    iterations = [row[0] for row in data]
    num_losses = len(data[0]) - 1
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.ylim(1e-10, 1e2)
    
    for i in range(1, num_losses + 1):
        loss_values = [row[i] for row in data]
        plt.plot(iterations, loss_values, label=header[i])
    
    plt.title('Losses over Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/losses_ieee_style.png", dpi=300)

def plot_results(file_path, file_path2):
    with h5py.File(f"{file_path}/val_ts.h5", 'r') as hf:
        target_ts = hf['target'][:]
        pred_ts = hf['pred'][:]
    with h5py.File(f"{file_path}/val_ts_err.h5", 'r') as hf:
        err_ts = hf['error_stats'][:]
    with h5py.File(f"{file_path2}/val_ts.h5", 'r') as hf:
        target_ts2 = hf['target'][:]
        pred_ts2 = hf['pred'][:]
    with h5py.File(f"{file_path2}/val_ts_err.h5", 'r') as hf:
        err_ts2 = hf['error_stats'][:]

    window_size = 50
    absolute_error = np.abs(target_ts - pred_ts)
    max_error_index = np.argmax(absolute_error.T[0])
    start_index = max(0, max_error_index - window_size)
    end_index = start_index + window_size * 2

    # Validation error plot
    plt.figure(figsize=(10, 6))
    plt.plot(err_ts, label='Validation Error Time Series')
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.yscale('log')
    plt.ylim(1e-4, 1e1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_path}/val_errorts_ieee_style.png", dpi=300)

    # Side-by-side validation plots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 6),sharey=True)
    colors = ["red", "blue"]
    
    for i in range(target_ts.shape[1]):
        axs[0].plot(target_ts[:window_size, i], "--", color=colors[i], label=f"Target {i+1}")
        axs[0].plot(pred_ts[:window_size, i], color=colors[i], label=f"Prediction {i+1}")
    
    for i in range(target_ts2.shape[1]):
        axs[1].plot(target_ts2[:window_size, i], "--", color=colors[i], label=f"Target2 {i+1}")
        axs[1].plot(pred_ts2[:window_size, i], color=colors[i], label=f"Prediction2 {i+1}")
    
    axs[0].set_xlabel('Iterator Step', fontsize=12)
    axs[0].set_ylabel('Solution $U,W$', fontsize=12)
    #axs[0].legend(loc='upper right')
    axs[1].set_xlabel('Iterator Step', fontsize=12)
    #axs[1].legend(loc='upper right')
    #axs[0].grid(True)
    #axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"val_error_ts_ieee_style.pdf", dpi=300)

    # Loss plot
    filename = f"{file_path}/losses.csv"
    data, header = read_csv(filename)
    plot_losses(data, header, file_path)

# Example usage
if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    file_path2 = sys.argv[2]
    plot_results(file_path, file_path2)