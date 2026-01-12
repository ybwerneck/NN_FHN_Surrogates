from PinnTorch.dependencies import *
from PinnTorch.Validator import Validator

def ensure_at_least_one_column(x):
    # If x is 1-dimensional, reshape to (len(x), 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    # If x is already 2-dimensional, return it unchanged
    elif x.ndim == 2:
        return x
    else:
        raise ValueError("Input array must be 1D or 2D")
        

class Recursive_Validator(Validator):
    def __init__(self,data_in,time_series_data,name="val",device=torch.device("cuda"),dump_f=0,dump_func=0,script=0):
        self.time_series_data=time_series_data

        self.data_input=data_in.to("cpu")
        self.data_in,self.target=time_series_data
        self.device=device
        self.dump_f=dump_f
        self.dump_func=self.dump_f_def if  dump_func==0 else lambda dump:dump_func(self,dump=dump) 
        self.name=name
        self.count=0
        self.last_out=0
        self.model=0
        self.script=script
        
    
    def val(self, model,p=False):
        # Evaluate the model
        self.model=model
        batch_size = 100*2048  # Choose an appropriate batch size

        data_in_ts,target_ts=self.time_series_data

        val_obj=self
        model=val_obj.model
        xi=val_obj.data_input.to(val_obj.device)
        data_out=torch.zeros_like(target_ts).reshape((len(xi),-1,2))
        k=1
        its=len(target_ts.T[0])//len(xi)
        print("!!!",np.shape(target_ts),len(target_ts.T[0]))
        data_out[:,0,:]=torch.stack((xi.T[0],xi.T[1]),axis=1)
        for i in range(its-1):
            print(xi)
            xi=model(xi)      

            if(i%1==0):
                data_out[:,k,:]=xi
                
                k+=1
            
            #xi=torch.stack((xi.T[0],xi.T[1],val_obj.data_input.T[2].to(val_obj.device)),axis=1)
            xi=torch.stack((xi.T[0],xi.T[1]),axis=1)

        dump=(self.count%self.dump_f==0)

        self.last_out=data_out.permute(0, 1, 2).reshape(-1, 2)
        print("aa",data_out)
        print(np.shape(data_out))
        self.dump_func(dump=dump)
    
        self.count+=1
        if(dump):
            if(self.script!=0):
                self.script()
        return 0

