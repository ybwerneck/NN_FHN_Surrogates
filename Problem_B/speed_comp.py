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
import os
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
import pandas as pd
import torch
from torch2trt import torch2trt
import time as TIME


def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
    cached_memory = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"Allocated GPU Memory: {allocated_memory:.2f} GB")
    print(f"Cached GPU Memory: {cached_memory:.2f} GB")


def runModel(x, M=0, batch_size=10):
    my2dspace = x
    M.eval()

    num_samples = my2dspace.shape[0]
    gc.collect()
    torch.cuda.empty_cache()
    uu_list = []

    times = []

    for i in range(0, num_samples, batch_size):
        batch_input = torch.tensor(my2dspace[i:i+batch_size], requires_grad=False).float().cuda()
        
        start_time = TIME.time()
        batch_output = M(batch_input)
        torch.cuda.synchronize()

        reftime = TIME.time() - start_time
        times.append(reftime)

        uu_list.append(batch_output.cpu().numpy())

    uu = np.concatenate(uu_list)
    return uu, np.sum(times), batch_size / np.mean(times), 0, np.mean(times)


if __name__ == "__main__":
    ts = [int(1e6 * 2**i) for i in range(6, 7)]
    base_dir = sys.argv[1]

    pt.set_grad_enabled(False)
    batch_size_nn = 80000

    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    total_subfolders = len(subfolders)

    # Use the appropriate GPU device
    device = torch.device('cuda')

    for index, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(base_dir, subfolder)

        log_file = os.path.join(subfolder_path, "log_speed_comp.txt")
        model_path = os.path.join(subfolder_path, 'model')

        if not os.path.exists(model_path):
            print(f"[{index+1}/{total_subfolders}] No model found in {subfolder_path}, skipping.")
            continue

        # Load model
        net = pt.load(model_path).to(device)

        # Clear GPU memory
        torch.cuda.empty_cache()

        tcs, tns, tes = [], [], []

        with open(log_file, "w") as log:
            log.write("Beginning speed comparison\n")
            log.write(f"{ts}\n")

        def f(x):
            with open(log_file, "a") as log:
                log.write(x + "\n")

        x = [torch.ones((batch_size_nn, 3)).cuda()]
        model_trt = torch2trt(net, x)

        data = []
        for T in ts:
            torch.cuda.empty_cache()
            nrp = 1

            tf = 10
            dt = 1
            m = int(tf / dt)
            T = T * m

            sampleset = torch.rand((T, 3))

            net_time_tt = []
            for _ in range(nrp):
                pr, net_time_a, tpt, it, passt = runModel(sampleset, M=model_trt, batch_size=batch_size_nn)
                net_time_tt.append(net_time_a)

            cuda_time = net_time_tt
            bt = batch_size_nn
            data.append([
                T // m, bt, f"{T // m} X {m}", np.mean(cuda_time),
                np.mean(cuda_time) / tf, np.mean(cuda_time) / ((tf // 0.1 + 1) * T)
            ])

        # Save results to CSV in the subfolder
        df = pd.DataFrame(data, columns=["T/m", "Batch Size", "Description", "Avg Time", "Time/TF", "Normalized Time"])
        csv_path = os.path.join(subfolder_path, "speed_comp_results.csv")
        df.to_csv(csv_path, index=False)

        print(f"[{index+1}/{total_subfolders}] Saved results for {subfolder_path}. Remaining: {total_subfolders - (index + 1)}")
