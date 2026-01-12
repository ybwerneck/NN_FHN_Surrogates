import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../'))

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
from data_generator.FHNCUDAlib import FHNCUDA
import itertools
from torch2trt import torch2trt
import torch
import pandas as pd

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

batch_size=10000
batch_size_n=10000*26

# create example data
x = torch.ones((batch_size_n,4)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(net, [x])
##PRED NETWORK RT

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
    #print(f"batchs {num_samples//batch_size}")

    for i in range(0, num_samples, batch_size):

        
        
        
        #print_gpu_memory()

        batch_input =torch.tensor(my2dspace[i:i+batch_size], requires_grad=False).float().cuda() 
        
        
       # print_gpu_memory()

        
        
        start_time = TIME.time()
        #print(np.shape(batch_input))

        batch_output = M(batch_input)
        
        
 

        torch.cuda.synchronize()  # Wait for the events to be recorded!
        
        
        
        
        reftime =reftime+ TIME.time() - start_time
        
        
        uu_list.append(batch_output.cpu().numpy())
        #del batch_input
        #del batch_output
        #print_gpu_memory()
       
 
    uu = np.concatenate(uu_list)
    
    return uu, reftime



#!nvcc cuda.cu -o a.out -arch=sm_86 -O3 --use_fast_math --ptxas-options=-v -Xptxas -dlcm=cg -Xcompiler -ffast-math --maxrregcount=32

def runCuda(sample_set, batch_size=10240,dt=0.01,rate=100,tt=50):
    K=int(tt/(dt*rate))+1
   # print(f"aaaaaaaa{K}")
    N=int(K*len(sample_set))
    # Initialize lists to store results
    pt=np.zeros(3)
    x0=np.zeros((N,4))
    batch_size=len(sample_set)
    u_num=np.zeros(N)
    I=0
    # Batch processing loop
    for i in range(0, len(sample_set), batch_size):
        batch_samples = sample_set[i:i+batch_size]

        # Prepare batch for CUDA
        x0_batch = np.array(batch_samples)

        

        # Execute CUDA computation
        start_time = TIME.time()
        u, v, t, p = FHNCUDA.run(x0_batch, tt, dt, rate)
        cudatime = TIME.time() - start_time
        # Process results
         
        p = [i / 1000 for i in p[0]]
        t = np.array(t).flatten()

        # Store results
        pt+=p
    
        u_num[K*i :K*(i+batch_size)]=0
        # Generate parameter list for each time step
        param_list = []
 
        #for sample in batch_samples:
         #   u, v, k = sample
          #  print(I)

           # for T in t:
            #    x0[I]=[T, u, v, k]
             #   I+=1
 


    return pt,np.ones((N*len(t),4)), u_num, u_num


runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

#engine = runtime.deserialize_cuda_engine(f.read())
#context = engine.create_execution_context()
log_file="logs/log_speed_comp.txt"




ts=[int(10000*(2**(x//2))) for x in range(2,10)]

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
            T=bt
            torch.cuda.empty_cache()
            print(f"Set of size {T} \n")
            sampleset=torch.rand((T,1))
            print("Cuda -")
            nrp=1
            cuda_time=0
            for i in range(nrp):
                    cuda_time_a,x0,un,uref=runCuda(sampleset,batch_size=-1,dt=.01,rate=200)
                    #print(f"{cuda_time_a}")
                    cuda_time=cuda_time+cuda_time_a[1]/nrp
            print(f"time : {cuda_time} s\n")
            
            
            print("pytorch -")
            net_time=0
            for i in range(10):
                pr,net_time_a=runModel(x0,batch_size=batch_size_n)
                net_time=net_time+net_time_a/nrp
            
            for i in range(nrp):
                pr,net_time_a=runModel(x0,batch_size=batch_size_n)
                net_time=net_time+net_time_a/nrp
            
            print(f"time pytorch {net_time}")
            
            
   
            
            net_time_tt=0
            for i in range(10):
                pr,net_time_a=runModel(x0,M=model_trt,batch_size=batch_size_n)
            for i in range(nrp):
                pr,net_time_a=runModel(x0,M=model_trt,batch_size=batch_size_n)
                net_time_tt=net_time_tt+net_time_a/nrp
            
            print(f"time tensorrt {net_time_tt}")

            data.append([T,cuda_time,net_time,net_time_tt])
   
            tcs.append(cuda_time)    
   
            tns.append(net_time)
            tes.append(net_time_tt)

        
plt.plot(ts,tcs,label="Tempo Cuda")
plt.plot(ts,tns, label="Tempo tensorrt ")
plt.plot(ts,tes, label="Tempo pytorch")

plt.legend(loc="best")


plt.savefig(base_dir+"/results.png")

# Create a DataFrame and save to CSV
df = pd.DataFrame(data, columns=['Size', 'Cuda', 'Python',"tensorrt"])
df = df.sort_values(by='Size')
df.to_csv(base_dir+'/time_values.csv', index=False)
