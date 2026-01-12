import sys
import os
import shutil


# Set up the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../'))


from train_protocol import train
from PinnTorch.dependencies import *


import itertools
import pickle
from mpi4py import MPI

# Initialize the MPI communicator
comm = MPI.COMM_WORLD

# Get the rank of the current process
rank = comm.Get_rank()
csize = comm.Get_size()
print(rank)
log_file = f"logs/mpi_log2{rank}.txt"



# Print the rank
print(f"Process rank: {rank}")

class SinusoidalActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
import os
import shutil

dir=sys.argv[1]
try:
    os.mkdir(dir)
    dst=dir
    shutil.copyfile("train_protocol.py", dst)
    shutil.copyfile("train_mpi.py", dst)
except:
    print("already ",dir)
    #clear_folder(dir)
        
# Define the listhl = [(nn.SiLU, 16),(nn.ELU, 8),(nn.SiLU, 32)]
hl = [(nn.SiLU,8),(nn.SiLU,16),(nn.SiLU,32),(nn.SiLU,64),(nn.SiLU,128)]

# Define the input array with sizes
sizes = [1,2,3,4,5]

# Initialize the final list to store sets of combinations for each size
simulations = []
piins=[False]
iccss=[True]
bsz=[128]
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
        for p in piins:
            for iccs in iccss:
                for bs in bsz:
                    simulations.append(
                        {
                            "model_params": {
                                "hidden_layers": u,
                                "pinn":p,
                                "iccs":iccs,
                                "bs":bs
                            }
                        }
        )


if(len(sys.argv)>=3):
    i,e=int(sys.argv[2]),int(sys.argv[3])
# Example: Print the generated combinations for verification
else:
    i,e=0,len(simulations)
if(rank==0):
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
            pdir=dir+f"simulation{o+100}"

            if(o<i or o>=e):
                with open(log_file, "a") as log:
                            log.write(f"skipping {pdir} out of b \n")
                continue
            

            if(o%csize==rank):
                    if(simulations[o]["model_params"]["pinn"]==True):
                        with open(log_file, "a") as log:
                            log.write(f"skipping {pdir} pinn \n")
                        continue
                    print("aaa",pdir)
                    with open(log_file, "a") as log:
                        log.write(f"trying {pdir}\n ")
                        if(os.path.exists(pdir)):
                            log.write(f"skipping {pdir} \n")
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
        for o,_ in enumerate(simulations):
            pdir=dir+f"simulation{o}"
            if(o<i or o>e):
                with open(log_file, "a") as log:
                            log.write(f"skipping {pdir} out of b \n")
            
            print(o%csize,"bbbbb")
            

            if(True):
                    if(simulations[o]["model_params"]["pinn"]==True):
                        with open(log_file, "a") as log:
                            log.write(f"skipping {pdir} pinn")
                        continue
                    print("aaa",pdir)
                    with open(log_file, "a") as log:
                        log.write(f"trying {pdir}")
                        if(os.path.exists(pdir)):
                            log.write(f"skipping {pdir}")
                            continue
                    try:
                        os.mkdir(pdir)
                    except:
                        print("overwr")
                    with open(pdir+'/my_dict.pkl', 'wb') as f:
                            pickle.dump(simulations[o], f)
                    with open(log_file, "a") as log:
                        log.write(f"\n RANK{rank}, runnin simu {o} params:\n {simulations[o]  } \n in dir {pdir} with gpu {gpus_process}")
                    train(simulations[o]["model_params"],outputfolder=pdir,gpuid=gpus_process)
            