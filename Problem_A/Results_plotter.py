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
        plt.figure(figsize=(10, 6),dpi=100)
        plt.yscale('log')
        plt.ylim(1e-7, 1e3)
        
        for i in range(1, num_losses + 1):
            loss_values = [row[i] for row in data]
            plt.plot(iterations, loss_values, label=header[i])
        plt.title('Losses over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(dir+"/losses.pdf")
        plt.clf()

def plot_results(file_path):
    with h5py.File(file_path + "/val.h5", 'r') as hf:
        target = hf['target'][:]
        pred = hf['pred'][:]
            
    with h5py.File(file_path + "/val_err.h5", 'r') as hf:
        err = hf['error_stats'][:]

    window_size = 5000
    absolute_error = np.abs(target - pred)
    max_error_index = np.argmax(absolute_error.T[0])
    print(max_error_index)
    print(err)

    start_index = max(0, max_error_index - window_size)
    end_index = start_index + window_size * 2

    # Plot error
    print("Val Error")
    plt.figure(dpi=100,figsize=(10,6))
    plt.title("Validation loss over iteratons PINN")
    plt.plot(np.linspace(0,2e6,len(err)),err,label=["Mean Error","Max Error"])
    plt.legend(loc="best")
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.yscale('log')
    plt.ylim(1e-4, 1e1)
    plt.savefig(dir+'/val_error.pdf')
    plt.clf()
        
    # Plot target and prediction for the first window size
    print("Last Validation Set")
    cls=["Blue","Blue"]
    i=0
    plt.figure(dpi=100,figsize=(10,6))
    plt.plot(target.T[0],"--",c=cls[i],label="Target")
    plt.plot(target.T[1],"--",c=cls[i])

    plt.plot(pred.T[0],c=cls[i],label="Prediction")
    plt.plot(pred.T[1],c=cls[i])

    plt.title("$DDNN$ validation set")
    plt.xlabel('t')
    plt.ylabel('U,W')
    plt.legend(loc="best")
    plt.savefig(dir+'/last_validation_set.pdf')
    plt.clf()

    # Plot absolute error in the region around the maximum error
    print(np.mean(absolute_error))
    plt.plot(absolute_error[start_index:end_index])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(['Errw', 'Errv'])
    plt.savefig(dir+'/absolute_error.png')
    plt.clf()
  
    # Plot target and prediction in the region around the maximum error
    plt.plot(target[start_index:end_index, 0], label='Target')
    plt.plot(pred[start_index:end_index, 0], label='Prediction')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Losses over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend() 
    plt.savefig(dir+'/target_prediction_max_error.png')
    plt.clf()
    
    # Scatter plot of target vs prediction
    print(np.shape(target),np.shape(pred))
    plt.scatter(target[:10000], pred[:10000])
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.legend(['Target', 'Prediction'])
    plt.savefig(dir+'/target_vs_prediction.png')
    plt.clf()


    # Example usage
    filename = dir+'/losses.csv'
    data,h = read_csv(filename)
    plot_losses(data,h)
# Call the plot_results function with the path to your HDF5 file
import sys

file_path = "batch_results/simulation0"
file_path=sys.argv[1]

dir=file_path
plot_results(file_path)

