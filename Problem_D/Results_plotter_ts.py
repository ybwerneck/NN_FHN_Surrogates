import h5py
import matplotlib.pyplot as plt
import numpy as np
import operator
import csv

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
        
        for i in range(1, num_losses +1):
            loss_values = [row[i] for row in data]
            plt.plot(iterations, loss_values, label=header[i])
        plt.title('Losses over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir+"/losses.png")

def plot_results(file_path):

    with h5py.File(file_path + "val_ts.h5", 'r') as hf:
        target_ts = hf['target'][:]
        pred_ts = hf['pred'][:]
            
     
    with h5py.File(file_path + "val_ts_err.h5", 'r') as hf:
        err_ts = hf['error_stats'][:]

    window_size = 50
    absolute_error = np.abs(target_ts - pred_ts)
    max_error_index = np.argmax(absolute_error.T[0])
    print(max_error_index)

    start_index = max(0, max_error_index - window_size)
    end_index = start_index + window_size * 2
    print(start_index,end_index)

    dir=file_path


    plt.figure()
    print("Val Error Ts")
    plt.plot(err_ts)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.ylim(1e-4, 1e1)
    plt.savefig(dir+'/val_errorts.png')
    plt.clf()
    print("Last Validation Set")
    cls=["Red","Blue"]
    for i in range(len(target_ts.T)):
        plt.plot(target_ts[:window_size,i],"--",c=cls[i])
        plt.plot(pred_ts[:window_size,i],c=cls[i])
    
    plt.plot()
    plt.xlabel('Iterator Step (1 t.u.)')
    plt.ylabel('Solution $W,S$')

    plt.savefig(dir+'/val_error_ts.pdf')
    plt.clf()
        


    plt.clf()
    plt.scatter(target_ts.T[0][start_index:end_index], pred_ts.T[0][start_index:end_index])
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.legend(['Target', 'Prediction'])
    plt.savefig(dir+'/target_vs_prediction_ts_ms.png')
    plt.clf()


    # Example usage
    filename = dir+'/losses.csv'
    data,h = read_csv(filename)
    plot_losses(data,h)
    
    
# Call the plot_results function with the path to your HDF5 file
import sys

#file_path = "batch_results/simulation0"
file_path=sys.argv[1]

dir=file_path
plot_results(file_path)

