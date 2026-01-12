
import subprocess
import torch
import numpy as np


def recursive_validator(val_obj,time_series_data,file="Results_plotter.py",dump=False):
    
    data_in_ts,target_ts=time_series_data
    model=val_obj.model
    print(np.shape(data_in_ts))
    print("bb",np.shape(val_obj.data_in))
    xi=val_obj.data_in.to(val_obj.device)
    data_out=torch.zeros_like(target_ts).reshape((len(xi),-1,2))
    print(np.shape(target_ts))
    k=1
    print(np.shape(target_ts))
    its=len(time_series_data[0])//len(xi)
    data_out[:,0,:]=torch.stack((xi.T[0],xi.T[1]),axis=1)
    
    for i in range(its-1):
        
        xi=model(xi)      

        if(i%1==0):
            data_out[:,k,:]=xi
            
            k+=1
        
        xi=torch.stack((xi.T[0],xi.T[1],val_obj.data_in.T[2].to(val_obj.device)),axis=1)
    
    val_obj.dump_f_def(dump=dump)
    val_obj.dump_f_def(target_ts,data_out.permute(0, 1, 2).reshape(-1, 2),data_in_ts,sufix="_ts",dump=dump)
    subprocess.Popen(f"python {file} {val_obj.folder}/", shell=True, stdout=subprocess.PIPE).stdout.read()


def gen_recursive_validator(time_series_data):
    return lambda val_obj,dump:recursive_validator(val_obj,time_series_data,file="Results_plotter.py",dump=dump)