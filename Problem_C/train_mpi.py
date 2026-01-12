import sys
import os
import pandas as pd
import ast
# Set up the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../'))
import torch
print(torch.__file__)  # Path to PyTorch installation
print(torch.version.cuda)  # CUDA version used by PyTorch

from train_protocol import train
from PinnTorch.dependencies import *
import itertools
import torch.nn as nn

import itertools
import pickle
from mpi4py import MPI

# Initialize the MPI communicator
comm = MPI.COMM_WORLD

# Get the rank of the current process
rank = comm.Get_rank()
csize = comm.Get_size()
print(rank)
machine_id=int(sys.argv[1])
log_file = f"logs/mpi_log{rank}_{machine_id}.txt"






# Print the rank
print(f"Process rank: {rank}")


import os

dir="Problem_C_Results/"
pinn=False

try:
    os.mkdir(dir)
except:
    print("already ",dir)
    #clear_folder(dir)
    # Define the list
training_sets=["traininig_data/treino_s/"]

model_range=[0,100]
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

    for ts in training_sets:
     for _, row in df.iterrows():
        print(ast.literal_eval(row["layers"]))
        model_config = {
                    "model_params": {
                    "hidden_layers": ast.literal_eval(row["layers"]),  # Deserialize layers
                    "pinn": False,  # Default to False if not present
                    "bs": row.get("bs", 256),  # Default batch size
                    "iccs":False,
                    "training_set":ts
                }
        }
        models.append(model_config)



    return models


# Load simulations from the model zoo
simulations = load_model_zoo("model_zoo.csv")

# Print summary
if(rank==0):
    print(f"Generated {len(simulations)} models.")
                        
    print(simulations)
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
            print(o%csize,"bbbbb")
            

            if(o%csize==rank):
                    print("aaa",pdir)
                    with open(log_file, "a") as log:
                        log.write(f"trying {pdir}")
                        if(os.path.exists(pdir)):
                            log.write(f"skipping {pdir}")
                            continue
                    try:
                        os.mkdir(pdir)
                    except:
                        continue
                    with open(pdir+'/my_dict.pkl', 'wb') as f:
                            pickle.dump(simulations[o], f)
                    with open(log_file, "a") as log:
                        log.write(f"\n RANK{rank}, runnin simu {o} params:\n {simulations[o]  } \n in dir {pdir} with gpu {gpus_process}")
                    train(simulations[o]["model_params"],outputfolder=pdir,gpuid=gpus_process)