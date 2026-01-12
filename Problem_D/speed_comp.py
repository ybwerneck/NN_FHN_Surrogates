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
 
batch_size_nn=200000




# Use the appropriate GPU device
device = torch.device('cuda')

net=pt.load(dir+'/model').to(device)




##PRED NETWORK RT

def runModel(x, M=net, batch_size=10,its=10):

    my2dspace = x
    M.eval()
        
    num_samples = my2dspace.shape[0]
    gc.collect()
    torch.cuda.empty_cache()
    uu_list = []
   


    times=[]
    it_times=[]
    gc.collect()
    torch.cuda.empty_cache()
    #print(f"batchs {num_samples//batch_size}")
    for k in range (its):
        it_time=0
        for i in range(0, num_samples, batch_size):

            
            #print(f"{i} de {num_samples}")
            
            #print_gpu_memory()
            if(num_samples-i < batch_size):
                 i=num_samples-batch_size

            batch_input =my2dspace[i:i+batch_size].clone().float().cuda() 
            
            
        # print_gpu_memory()

            
            
            start_time = TIME.time()
            #print(np.shape(batch_input))

            batch_output = M(batch_input)
            
            
    

            torch.cuda.synchronize()  # Wait for the events to be recorded!
            
            
            
            t=TIME.time() - start_time
            times.append(t)
     
            
            #print(f" {i} out of {num_samples}")
            uu_list.append(batch_output.cpu().numpy())
            #del batch_input
            #del batch_output
        #print_gpu_memory()
            it_time+=t
        it_times.append(it_time)
    uu = np.concatenate(uu_list)
    print(f" {len(times)} it in avg {np.mean(np.array(times))} {times}" )


    return uu, np.sum(times),batch_size/np.mean(times),np.mean(it_times),np.mean(times)




runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

#engine = runtime.deserialize_cuda_engine(f.read())
#context = engine.create_execution_context()
log_file="logs/log_speed_comp.txt"




ts=[int(1e6 * 2**i) for i in range(1, 7)]

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
    for T in ts:
            
            torch.cuda.empty_cache()
            print("\n \n \n \n \n")
            nrp=1

            print(f"Set of size {T} \n")
            sampleset=torch.rand((T,2))
            torch.cuda.empty_cache()

           

            its=10

            x = torch.ones((batch_size_nn,2)).cuda()
            model_trt = torch2trt(net, [x])
            
            
            print("tensort py -")                    
            net_time_tt=[]
            it_time_tt=[]
            pass_time=[]
            toput=[]
            for i in range(0):
                pr,net_time_a,_=runModel(sampleset,M=model_trt,batch_size=batch_size_nn,its=its)
            for i in range(nrp):
                pr,net_time_a,tpt,it,passt=runModel(sampleset,M=model_trt,batch_size=batch_size_nn,its=its)
                net_time_tt.append(net_time_a)
                it_time_tt.append(it)
                pass_time.append(passt)
                toput.append(tpt)


            cuda_time=net_time_tt
            bt=batch_size_nn
            print(f"time tensorrt {net_time_tt}")
            pd.options.display.float_format = '{:.3e}'.format
            tf=float(its)
            data.append([T,bt,f"{(T+bt-1)//bt} X {its}",np.mean(cuda_time),np.mean(cuda_time)/tf,np.mean(cuda_time)/((tf//0.1 + 1)*T)])
            df = pd.DataFrame(data, columns=['Set_size','Batch_size', 'ITS', 'Tensort Time'," Tensort time per 1 s ","Tensor time per it per cel"])
            df = df.sort_values(by='Batch_size')


            df.to_csv(base_dir+'/speed_values.csv', float_format='%.3e',  index=False)




    # Create a DataFrame and save to CSV
