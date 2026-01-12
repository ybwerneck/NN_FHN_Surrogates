import numpy as np
import matplotlib.pyplot as plt
import os
import chaospy as cp

import sys
import os


from FHNCUDAlib import FHNCUDA

# Constants
t_w = 100
nc = 12


def generateExactSolution(t, dt, x0, w0, rate, P, begin, end):
    n2 = int(t / dt) + 2
    n = int((end - begin) / (dt * rate))
    Sol = np.zeros((n, 3))
    Sol2 = np.zeros((n2, 2))
    Sol2[0] = x0, w0
    T = 0
    k = 0

    while k < n2 - 1:
        x, w = Sol2[k]
        Sol2[k + 1] = 10 * (x * (x - 0.4) * (1 - x) - w + P) * dt + x, 0.2 * (x * 0.2 - 0.8 * w) * dt + w

        if (k * dt == begin or ((k + 1) % rate == 0 and k * dt >= begin and k * dt <= end)) and T < n:
            Sol[T] = Sol2[k][0], Sol2[k][1], k * dt
            T += 1

        k += 1
        if k * dt > end:
            break
    return Sol

def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

def getDatasetFromDistA(sample_set, data_folder="./extrapol/", ti=0, tf=1, r=10, deltaT=0.01,norm=False):
    try:
        os.mkdir(data_folder)
    except:
        print("Overwriting")
    
    T = np.empty(0)
    K = np.empty(0)
    V = np.empty(0)
    U = np.empty(0)
    print("Generating dataset in", data_folder)

    rate = r
    num_samples = np.shape(sample_set)[1]
    print(np.shape(sample_set))

    u, v, t, p = FHNCUDA.run(sample_set.T, tf, deltaT, rate)
    t = t[0]
    u=np.array(u)
    v=np.array(v)

    u=u[:,ti:]
    v=v[:,ti:]
    t=t[ti:]
    SOLs = np.array(u).flatten()
    SOLw = np.array(v).flatten()
    
    T = np.tile(t, num_samples)
    K = np.repeat(sample_set[2, :], len(t))
    U = np.repeat(sample_set[0, :], len(t))
    V = np.repeat(sample_set[1, :], len(t))
    n = len(t)
    def min_max_scaling(arr, min_val, max_val):
        scaled_arr = (arr - min_val) / (max_val - min_val)
        return scaled_arr
    if(norm):
        T = min_max_scaling(T, 0, 50)
        K = min_max_scaling(K, 0, 1)
        U = min_max_scaling(U, -0.1, 0.81)
        V = min_max_scaling(V, 0, 0.12)
        SOLs = min_max_scaling(SOLs, -0.1, 0.81)
        SOLw = min_max_scaling(SOLw, 0, 0.12)

    print("Generated set")

    np.save(data_folder + "T.npy", T)
    np.save(data_folder + "K.npy", K)
    np.save(data_folder + "U.npy", U)
    np.save(data_folder + "V.npy", V)
    np.save(data_folder + "SOLs.npy", SOLs)
    np.save(data_folder + "SOLw.npy", SOLw)
    print(len(U))
    print(len(SOLs))
    nc = num_samples
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data=np.stack((U,V,K)).T
    print(np.shape(U))
    print(np.shape(data))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')

    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_zlabel('K')

    # Fix the range of the axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    plt.title('3D Scatter Plot')

    # Save the plot
    plt.savefig(f'{data_folder}/3d_scatter_plot_fixed_range.png')

    np.savetxt(f'{data_folder}/timepoints.txt', t, delimiter=',')

    #plt.show()

