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
        
        for i in range(0, num_losses ):
            loss_values = [row[i+1] for row in data]
            plt.plot(iterations, loss_values, label=header[i+1])
        plt.title('Losses over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(dir+"/losses.png")

def plot_results(file_path,make=False):
    with h5py.File(file_path + "/val.h5", 'r') as hf:
        target = hf['target'][:]
        pred = hf['pred'][:]
        input_data= hf['input'][:]

            
    with h5py.File(file_path + "/val_err.h5", 'r') as hf:
        err = hf['error_stats'][:]

    window_size = 500
    absolute_error = np.abs(target - pred)
    max_error_index = np.argmax(absolute_error.T[0])

    start_index = max(0, max_error_index - window_size)
    end_index = start_index + window_size * 2

    # Plot error
    print("Val Error")
    plt.plot(err)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.ylim(1e-4, 1e1)
    plt.savefig(dir+'/val_error.png')
    plt.clf()
        
    # Plot target and prediction for the first window size
    print("Last Validation Set")
    cls=["Red","Blue"]
    for i in range(len(target.T)):
        plt.plot(target[:window_size,i],"--",c=cls[i])
        plt.plot(pred[:window_size,i],c=cls[i])
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(['Target', 'Prediction'])
    plt.savefig(dir+'/last_validation_set.png')
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
    plt.legend()
    plt.savefig(dir+'/target_prediction_max_error.png')
    plt.clf()
    
   # plt.scatter(input_data.T[0][0:], input_data.T[1][0:], c=absolute_error[0:,0], cmap='viridis', label='All Points')
   # plt.colorbar(label='Error Magnitude')
   # plt.savefig(f"{file_path}/error_map.png")
   # plt.clf()
    # Scatter plot of target vs prediction
    print(np.shape(target),np.shape(pred))
    plt.scatter(target[:10000], pred[:10000])
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.legend(['Target', 'Prediction'])
    plt.savefig(dir+'/target_vs_prediction.png')
    plt.clf()

    if(make):
       
        error = np.mean(np.abs(target - pred),axis=1)
     
        print(u,v)
        plt.scatter(u,v,c=error)

        plt.tight_layout()
        plt.savefig("3d.pdf")
    # Example usage
    filename = dir+'/losses.csv'
    data,h = read_csv(filename)
    plot_losses(data,h)
# Call the plot_results function with the path to your HDF5 file
import sys

file_path = "batch_results/simulation0"
file_path=sys.argv[1]

try:
    make=sys.argv[2]=="f"
except:
    make=False
dir=file_path
plot_results(file_path,make)

