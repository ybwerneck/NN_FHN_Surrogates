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
    clear_folder(dir)
        
# Define the list
hl = [(nn.SiLU, 16),(nn.ELU, 8),(nn.SiLU, 32)]

# Define the input array with sizes
sizes = [2,3]

# Initialize the final list to store sets of combinations for each size
simulations = []
iccss=[True,False]
# Function to create a sorting key
def sort_key(array_of_tuples):
    # Sort each tuple by (num, cls.__name__)
    return [array_of_tuples[i][0].__name__ for i in range (len(array_of_tuples))]
# Generate combinations for each size and add to the final list
result=[]
for size in sizes:
    result=[]
    unique_combinations = set(itertools.combinations_with_replacement(hl, size))
    for combination in unique_combinations:
        permutations = itertools.product(combination, repeat=size)
        result.extend(permutations)
    result=set(result)
    sorted_combinations = sorted(result, key=sort_key)  # Ensure the combinations are ordered consistently
    for u in sorted_combinations:
        for iccs in iccss:
            simulations.append(
                {
                    "model_params": {
                        "hidden_layers": u,
                        "pinn":False,
                        "iccs":iccs
                    }
                }
        )

# Example: Print the generated combinations for verification
for idx, comb in enumerate(simulations):
    print(f"Simulation {idx}: {comb}")
    
if(True):
        # Number of processes per node (ppn)


        # Total number of GPUs available
        total_gpus = torch.cuda.device_count()

        # Number of GPUs each process should use
        gpus_process = 1
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

            if(True):
                try:
                    os.mkdir(pdir)
                except:
                     print("overwr")
                with open(pdir+'/my_dict.pkl', 'wb') as f:
                        pickle.dump(simulations[o], f)
                with open(log_file, "a") as log:
                    log.write(f"\n RANK{rank}, runnin simu {o} params:\n {simulations[o]  } \n in dir {pdir}")
                train(simulations[o]["model_params"],outputfolder=pdir,gpuid=gpus_process)