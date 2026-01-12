from PinnTorch.dependencies import *
from PinnTorch.Net import *
from PinnTorch.Trainer import *
from PinnTorch.Validator import *
from PinnTorch.Loss import *
from PinnTorch.Loss_PINN import *
from PinnTorch.Utils import *
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
        device = torch.device("cpu")#torch.device(f"cuda:{gpuid}")##torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        tf=20
        ##Model Parameters
        norm=False#model_params["norm"]
        k_range = (0,1)
        v_range = (0,.12)
        u_range = (0,0.9)
        t_range=(0,tf)
        
        if(norm):
              k_range = (0,1)
              v_range = (0,1)
              u_range = (0,1)
              t_range=(0,1)
        
        
        nP=2
  
        iccs=model_params["iccs"]
        pinn=model_params["pinn"]
        batch_size=model_params["bs"]

        if(norm):
            setl=setl+"_norm"

        ##Model Arch
        input_shape = 4 
        output_shape = nP
           
        hidden_layer = model_params["hidden_layers"]
        dtype=torch.float32
        model = FullyConnectedNetworkMod(input_shape, output_shape, hidden_layer,dtype=dtype).to(device)
        trainer=Trainer(model,output_folder=outputfolder,print_steps=10000,val_steps=10000)

        print(model)

        ICs=[0.5,0]



        def ode(t, x):
            nP = len(x)
            u_i=x[0]
            v_i=x[1]
            k=x[2]
            dxdt = [10*((1)*(u_i*(u_i-0.4)*(1-u_i))-v_i + k*0.04 + 0.08  ),
                    
                    ((u_i*0.04-0.16*v_i))
                    
                    ]
            return dxdt

            
        def FHN_LOSS(data_in, model):
                    
                    ts = data_in
                    u = model(ts)
                    u=torch.stack((u.T[0],u.T[1],data_in.T[3]),axis=1)
                    ode_target=ode(ts,u.T)
                
                    odeA=0

                    factors=[1/(50*0.91),1/(50*0.12)] if norm else [1,1]


                    for i in range(nP):
                        ui= u.T[i].view(-1,1)

                        tgt=ode_target[i].view(-1,1)*factors[i]


                        #Mode1
                        dT = grad(ui, ts)[0].T[0].view(-1,1)          
                        odeA += (dT - tgt)**2
                    

                    # MSE of ODE

                    return torch.mean(odeA)







        batch_gen=lambda size,de:default_batch_generator(size,[t_range,u_range,v_range,k_range],device=de)

        if(pinn):
            trainer.add_loss(LOSS_PINN(FHN_LOSS,batch_gen,batch_size=batch_size,device=device),weigth=1)

    

        data_int,data_outt=LoadDataSet(f"training_data/treino/",data_in=["T.npy","U.npy","V.npy","K.npy"],device=device,dtype=dtype)
        data_inv,data_ouv=LoadDataSet(f"training_data/validation/",data_in=["T.npy","U.npy","V.npy","K.npy"],device=device,dtype=dtype)
        trainer.add_loss(LPthLoss(data_int,data_outt,batch_size,2,True,device,"Data Loss"),weigth=10)

        if(iccs):
            data_inte,data_outte=LoadDataSet(f"training_data/treino_r/",data_in=["T.npy","U.npy","V.npy","K.npy"],device=device,dtype=dtype)

            trainer.add_loss(LPthLoss(data_inte,data_outte,batch_size,2,True,device,"Data Loss Extra"),weigth=10)


        def f(data_in, model):
    
                u = model(data_in) 
                u0=data_in.T[1].view(-1,1)
                v0=data_in.T[2].view(-1,1)
                
                u0_pred,u1_pred= u.T[0].view(-1,1) , u.T[1].view(-1,1)

                

                t0 = (u0-u0_pred)**2 + (u1_pred-v0)**2



                return torch.mean(t0)

        
        batch_gen=lambda size,de:default_batch_generator(size,[(0,0),u_range,v_range,k_range],device=de)
        if(pinn):
            trainer.add_loss(LOSS_PINN(f,batch_gen,batch_size=batch_size,device=device),weigth=1)


        ##Validator
        trainer.add_validator(Validator(data_inv,data_ouv,"val",device,dump_f=5 ,dump_func=default_file_val_plot))


        trainer.train(int(5e5)+500)





        print(model)

