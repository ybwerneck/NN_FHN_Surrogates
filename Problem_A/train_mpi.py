import sys
import os

# Set up the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../'))


from train_protocol import train
from PinnTorch.dependencies import *


import itertools
rank=0
import pickle

log_file = f"logs/mpi_log{rank}.txt"

import os

dir="batch_results/"
try:
    os.mkdir(dir)
except:
    print("Result dir already exists, using some saved results")

import itertools
import torch.nn as nn
import sys
import os
import pickle
import pandas as pd
import torch
import ast
from mpi4py import MPI

# Initialize the MPI communicator
comm = MPI.COMM_WORLD

# Get the rank of the current process
rank = comm.Get_rank()
csize = comm.Get_size()

# Set up the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../'))

from train_protocol import train

# Path to the model zoo CSV file
MODEL_ZOO_PATH = "model_zoo.csv"

# Rank and logging setup

log_file = f"logs/mpi_log{rank}.txt"

# Directory for batch results
dir = "Problem_A_results/"
try:
    os.mkdir(dir)
except FileExistsError:
    print("Result directory already exists, using saved results")

# Read model zoo from the CSV file
def load_model_zoo(filepath):
    """
    Load the model zoo from a CSV file.

    Args:
        filepath (str): Path to the model zoo CSV file.

    Returns:
        List of model configurations.
    """
    df = pd.read_csv(filepath)
    models = []

    for _, row in df.iterrows():
     for npt in [10,100,1000]:
        print(ast.literal_eval(row["layers"]))
        model_config = {
            "model_params": {
                "hidden_layers": ast.literal_eval(row["layers"]),  # Deserialize layers
                "pinn": True,  # Default to False if not present
                "bs": 1024,
                "npt": npt# Default batch size
            }
            
        }
        models.append(model_config)
        model_config = {
            "model_params": {
                "hidden_layers": ast.literal_eval(row["layers"]),  # Deserialize layers
                "pinn": False,  # Default to False if not present
                "bs": 1024,
                "npt": npt# Default batch size
            }
            
        }
        models.append(model_config)

    return models


# Load simulations from the model zoo
simulations = load_model_zoo(MODEL_ZOO_PATH)

# Print summary
print(f"Generated {len(simulations)} models.")
# Example: Print the generated combinations for verification
for idx, comb in enumerate(simulations):
    print(f"Simulation {idx}: {comb}")


if(True):
        # Number of processes per node (ppn)


        # Total number of GPUs available
        total_gpus = torch.cuda.device_count()

        # Number of GPUs each process should use
        gpus_process = 0
        # Log the information
        with open(log_file, "w") as log:
            log.write(f"Total GPUs available: {total_gpus} \n") 
            
            log.write(f"Process {rank} is using GPU {gpus_process} \n")
            # Check if CUDA is available
            if torch.cuda.is_available():
                # Print all available GPU devices
                for i in range(torch.cuda.device_count()):
                    log.write(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                log.write("No GPU available. Only CPU is available.")
    
    
        for o,_ in enumerate(simulations):
      
            pdir=dir+f"simulation{o}"
            if(os.path.exists(pdir)):
                print("skipping ",pdir)
                continue
            
            if(o%csize==rank):
                try:
                    os.mkdir(pdir)
                except:
                     print("overwr")
                with open(pdir+'/my_dict.pkl', 'wb') as f:
                        pickle.dump(simulations[o], f)
                with open(log_file, "a") as log:
                    log.write(f"\n RANK{rank}, runnin simu {o} params:\n {simulations[o]  } \n in dir {pdir}")
                train(simulations[o]["model_params"],outputfolder=pdir,gpuid=gpus_process)