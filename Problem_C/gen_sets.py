import numpy as np
import matplotlib.pyplot as plt
import os
import chaospy as cp

import sys
import os
# Set up the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../data_generator'))

from set_generator import getDatasetFromDistA
# Define the ranges
krange = [0., 1]
vrange = [0., 0.12]
urange = [-0.1, 0.81]

k_dist = cp.Uniform(*krange)
v_dist = cp.Uniform(*vrange)
u_dist = cp.Uniform(*urange)


joint_dist = cp.J(u_dist, v_dist, k_dist)
T = int(1e5)
sample_set = joint_dist.sample(T, rule="L").T
np.random.shuffle(sample_set)
#sample_set.T[2]=0

sample_set = sample_set.T

print(sample_set.T[0])
print("Generating training data")
getDatasetFromDistA(sample_set[:, int(T * 0.05):], data_folder="training_data/treino/", ti=0, tf=20,r=10,norm=False)
print("Generating validation data")
getDatasetFromDistA(sample_set[:, 0:int(T * 0.01)], data_folder="training_data/validation/", ti=0, tf=20,r=10,norm=False)

