import os
import sys
import chaospy as cp
import numpy as np
# Set up the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../data_generator'))
from set_generator import getDatasetFromDistA
# Define the ranges
krange = [0., 1]
vrange = [0., 0.2]
urange = [-0.1, 1]

k_dist = cp.Uniform(*krange)
v_dist = cp.Uniform(*vrange)
u_dist = cp.Uniform(*urange)


joint_dist = cp.J(u_dist, v_dist, k_dist)
T = int(5e5)
sample_set = joint_dist.sample(T, rule="L").T
np.random.shuffle(sample_set)
sample_set.T[2]=0
sample_set_aux=sample_set.copy()
sample_set_aux.T[1]=0.19*sample_set_aux.T[0] +0.01
sample_set_aux.T[:-1]+=np.random.random((np.shape(sample_set_aux.T[:-1])))*0.05


sample_set = sample_set.T
sample_set_aux=sample_set_aux.T
print(sample_set.T[0])
print("Generating training data")
getDatasetFromDistA(sample_set[:, int(T * 0.05):], data_folder="training_data/treino/", ti=1, tf=1,r=100,deltaT=0.01,norm=False)


getDatasetFromDistA(sample_set_aux[:, int(T * 0.05):], data_folder="training_data/treino_region/", ti=1, tf=1,r=100,deltaT=0.01,norm=False)


print("Generating validation data")
getDatasetFromDistA(sample_set[:, 0:int(T * 0.05)], data_folder="training_data/validation/", ti=1, tf=1,r=100,deltaT=0.01,norm=False)

print("Generating validation data (time series)")
getDatasetFromDistA(sample_set[:, 0:int(T * 0.05)], data_folder="training_data/validation_ts/", ti=0, tf=50,r=100,deltaT=0.01,norm=False)
