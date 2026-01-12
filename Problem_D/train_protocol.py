from PinnTorch.dependencies import *
from PinnTorch.Net import *
from PinnTorch.Trainer import *
from PinnTorch.Validator import *
from PinnTorch.Loss import *
from PinnTorch.Loss_PINN import *
from PinnTorch.Utils import *
from Recursive_Validator import Recursive_Validator
def ensure_at_least_one_column(x):
    # If x is 1-dimensional, reshape to (len(x), 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    # If x is already 2-dimensional, return it unchanged
    elif x.ndim == 2:
        return x
    else:
        raise ValueError("Input array must be 1D or 2D")
    

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val
# Check if GPU is availabls
def train(model_params,outputfolder,gpuid):
        device = torch.device(f"cuda:{gpuid}")##torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        tf=20
        ##Model Parameters
        k_range = (0,1)
        v_range = (0,1)
        u_range = (0,1)
        t_range=(0,tf)
        nP=2

        ##Model Arch
        input_shape = 2
        output_shape = nP
           
        hidden_layer = model_params["hidden_layers"]
        use_region=model_params["region"]
        dtype=torch.float32
        model = FullyConnectedNetworkMod(input_shape, output_shape, hidden_layer,dtype=dtype).to(device)
        trainer=Trainer(model,output_folder=outputfolder,print_steps=1000,val_steps=5000)

        print(model)

        ICs=[0.5,0]  

        data_int,data_outt=LoadDataSet("training_data/treino/",data_in=["U.npy","V.npy"],device=device,dtype=dtype)
        trainer.add_loss(LPthLoss(data_int,data_outt,1024,2,True,device,"Data Loss"))
        data_int,data_outt=LoadDataSet("training_data/treino_region/",data_in=["U.npy","V.npy"],device=device,dtype=dtype)

        if(use_region):
            trainer.add_loss(LPthLoss(data_int,data_outt,1024,2,True,device,"Data Loss region"))

        
        data_inv,data_ouv=LoadDataSet("training_data/validation/",data_in=["U.npy","V.npy"],device=device,dtype=dtype)
        time_series_data=LoadDataSet("training_data/validation_ts/",data_in=["T.npy","U.npy","V.npy"],device=device,dtype=dtype)


        plot_script=lambda:subprocess.Popen(f"python Results_plotter_ts.py {outputfolder}/", shell=True, stdout=subprocess.PIPE).stdout.read()


        print(np.shape(data_outt))

        ##Validator

        trainer.add_validator(Recursive_Validator(data_inv,time_series_data,"val_ts",device,dump_f=5,script= plot_script))
        trainer.add_validator(Validator(data_inv,data_ouv,"val",device,dump_f=5 ,dump_func=default_file_val_plot))


        trainer.train(300000)





        print(model)

