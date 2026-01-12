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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        print(outputfolder)
        tf=25
        ##Model Parameters
        k_range = (0,1)
        v_range = (0,.15)
        u_range = (-.1,0.9)
        t_range=(0,tf)
        nP=2

        ##Model Arch
        input_shape = 1  
        output_shape = nP
           
        hidden_layer = model_params["hidden_layers"]
        pinn=model_params["pinn"]
        npt=model_params["npt"]
        dtype=torch.float32
        model = FullyConnectedNetworkMod(input_shape, output_shape, hidden_layer,dtype=dtype).to(device)
        trainer=Trainer(model,output_folder=outputfolder,print_steps=1000,val_steps=1000)

        print(model)

        ICs=[0.5,0]



        def ode2(t, x):
                dxdt = [0.04*x[i]*x[i]*(-1)**i for i in range(nP)]
                return dxdt


        def ode(t, x):
            nP = len(x)
            u_i=x[0]
            v_i=x[1]
            dxdt = [10*((1)*(u_i*(u_i-0.4)*(1-u_i))-v_i   ),
                    
                    ((u_i*0.04-0.16*v_i))
                    
                    ]
            return dxdt

       

        def FHN_LOSS(data_in, model):
            ts = data_in  # time collocation points
            u_pred = model(ts)  # shape: [N, 2] with columns u and v

            # Compute the time derivatives for each state variable separately
            du_dt = torch.autograd.grad(u_pred[:, 0], ts, grad_outputs=torch.ones_like(u_pred[:, 0]), create_graph=True)[0]
            dv_dt = torch.autograd.grad(u_pred[:, 1], ts, grad_outputs=torch.ones_like(u_pred[:, 1]), create_graph=True)[0]

            # Extract u and v from the model's output
            u_i = u_pred[:, 0]
            v_i = u_pred[:, 1]

            # Compute the ODE right-hand sides individually
            ode_u = 10 * (u_i * (u_i - 0.4) * (1 - u_i) - v_i)
            ode_v = 0.04 * u_i - 0.16 * v_i

            # Compute the mean squared errors of the residuals
            loss_u = torch.mean((du_dt - ode_u) ** 2)
            loss_v = torch.mean((dv_dt - ode_v) ** 2)
            ode_loss = loss_u + loss_v

            return ode_loss






        batch_gen=lambda xs,device:ensure_at_least_one_column(default_batch_generator(xs,[[0,tf]],device))
        if(pinn==True):
            trainer.add_loss(LOSS_PINN(FHN_LOSS,batch_gen,batch_size=64),weigth=2)



        

        ##BoundaryLossss

        def f(data_in, model):
    
                u = model(data_in) 

                u0,u1= u.T[0].view(-1,1) , u.T[1].view(-1,1)

                

                t0 = (u0-ICs[0])**2 + (u1-ICs[1])**2



                return torch.mean(t0)
        

        
        batch_gen=lambda size,de:ensure_at_least_one_column(torch.zeros(size,requires_grad=True).to(de).T)
        if(pinn==True):
            trainer.add_loss(LOSS_PINN(f,batch_gen,batch_size=2,device=device,name="BC"),weigth=1)

        #if(pinn==False):
        trainer.add_loss(FHN_LOSS_fromODE(ode,[0,tf],ICs,batch_size=8,num_points=npt,device=device,dtype=dtype,folder=outputfolder),weigth=100)

        ##Validator
        trainer.add_validator(FHN_VAL_fromODE(ode,[0,tf],ICs,1024*10,device=device,name="val",dtype=dtype,dump_factor=20))



        t0 = time.time()
        trainer.train(int(5e4) + 1000)
        elapsed = time.time() - t0

        with open(f"{outputfolder}/timing.txt", "a") as f:
            f.write(f"{elapsed:.4f}")

        print(model)

