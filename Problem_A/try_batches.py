import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import collections as coll
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import gc
import numpy as np
import torch as pt
import torch
import torch.nn as nn
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import tensorrt as trt
import time as TIME
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../data_generator'))
from FHNCUDAlib import FHNCUDA
import itertools
import pandas as pd
sys.path.append(os.path.join(current_dir, '../'))

from torch2trt import torch2trt
import torch

def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
    cached_memory = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"Allocated GPU Memory: {allocated_memory:.2f} GB")
    print(f"Cached GPU Memory: {cached_memory:.2f} GB")

print_gpu_memory()

dir=base_dir=sys.argv[1] 

pt.set_grad_enabled (False) 
 





# Use the appropriate GPU device
device = torch.device('cuda')

net=pt.load(dir+'/model').to(device)



def runModel(x, M=net, batch_size=10):

    my2dspace = x
    M.eval()
        
    num_samples = my2dspace.shape[0]
    gc.collect()
    torch.cuda.empty_cache()
    uu_list = []
   

    reftime =0
    
    gc.collect()
    torch.cuda.empty_cache()


    times=[]

    for i in range(0, num_samples, batch_size):

        
        
        
        #print_gpu_memory()

        batch_input =torch.tensor(my2dspace[i:i+batch_size], requires_grad=False).float().cuda() 
        
        
       # print_gpu_memory()

        
        
        start_time = TIME.time()
        #print(np.shape(batch_input))

        batch_output = M(batch_input)
        
        
 

        torch.cuda.synchronize()  # Wait for the events to be recorded!
        
        
        
        
        reftime = TIME.time() - start_time
        times.append(reftime)
        
        uu_list.append(batch_output.cpu().numpy())
        #del batch_input
        #del batch_output
        #print_gpu_memory()
       
 
    uu = np.concatenate(uu_list)
    
    return uu, np.sum(times),batch_size/np.mean(times), 0,np.mean(times)



runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

#engine = runtime.deserialize_cuda_engine(f.read())
#context = engine.create_execution_context()
log_file="logs/log_speed_comp.txt"



T=int(1e6)*4
ts=[int(1e6*2**(i)) for  i in range(1,3)] +[int(1e5 * 2**i) for i in range(1, 5)]+[int(1e4 * 2**i) for i in range(1, 5)]


# Use the appropriate GPU device
device = torch.device('cuda')

# Clear GPU memory
torch.cuda.empty_cache()

tcs,tns,tes=[],[],[]

with open(log_file, "w") as log:
   log.write("beggining speed comp")
   log.write(f"{ts}")
k=print 
def f(x):
    with open(log_file, "a") as log:
         log.write(x+"\n")



data=[]
if True:
    print=f
    for bt in ts:
            
            torch.cuda.empty_cache()
            print("\n \n \n \n \n")
            T
            
            print(f"Set of size {T} \n")
            sampleset=torch.rand((T,1))
            print("Cuda -")
            nrp=2


            x = torch.ones((bt,1)).cuda()
            model_trt = torch2trt(net, [x])
            
            
            print("tensort py -")                    
            net_time_tt=[]
            it_time_tt=[]
            pass_time=[]
            toput=[]
            for i in range(0):
                pr,net_time_a,_=runModel(sampleset,M=model_trt,batch_size=bt)
            for i in range(nrp):
                pr,net_time_a,tpt,it,passt=runModel(sampleset,M=model_trt,batch_size=bt)
                net_time_tt.append(net_time_a)
                it_time_tt.append(it)
                pass_time.append(passt)
                toput.append(tpt)


            
            print(f"time tensorrt {net_time_tt}")
            pd.options.display.float_format = '{:.2e}'.format
            data.append([T,bt,np.mean(net_time_tt),np.mean(tpt),np.mean(it_time_tt),np.mean(pass_time)])
            df = pd.DataFrame(data, columns=['Set_size','Batch_size',  'Tensorrt Time'," Thogurput ","Mean it time","Mean pass time"])
            df = df.sort_values(by='Batch_size')
            df.to_csv(base_dir+'/time_values_batch.csv', float_format='%.1e',  index=False)


            tes.append(np.mean(net_time_tt))
            print("\n \n \n \n \n")
            





    # Create a DataFrame and save to CSV
