import numpy as np
import matplotlib.pyplot as plt
import os
import chaospy as cp

import sys
import os
# Set up the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/home/yan/PINN/data_generator')

from set_generator import getDatasetFromDistA


# Define the ranges
krange = [0., 1]
vrange = [0., 0.12]
urange = [-0.1, 0.81]

k_dist = cp.Uniform(*krange)
v_dist = cp.Uniform(*vrange)
u_dist = cp.Uniform(*urange)


joint_dist = cp.J(u_dist, v_dist, k_dist)
T = int(1e4)
sample_set = joint_dist.sample(T, rule="L").T
np.random.shuffle(sample_set)
sample_set.T[2]=0


# Assuming sample_set is already defined and transposed
sample_set = sample_set.T
T = len(sample_set.T)

print("Generating training data")
# Original training dataset
train_data = sample_set[:, int(T * 0.5):]
getDatasetFromDistA(train_data, data_folder="traininig_data/treino_s/", ti=0, tf=25, norm=False)

# Dropping 75% of rows for a new dataset
train_data_25 = train_data[:,np.random.choice(train_data.shape[1], int(train_data.shape[1] * 0.25), replace=False)]
print(train_data_25.shape,"ssss")
getDatasetFromDistA(train_data_25, data_folder="traininig_data/treino_s_25/", ti=0, tf=25, norm=False)

# Dropping 95% of rows for another new dataset
train_data_5 = train_data[:,np.random.choice(train_data.shape[1], int(train_data.shape[1] * 0.05), replace=False)]
getDatasetFromDistA(train_data_5, data_folder="traininig_data/treino_s_5/", ti=0, tf=25, norm=False)

# Dropping 99% of rows for another new dataset
train_data_1 = train_data[:,np.random.choice(train_data.shape[1], int(train_data.shape[1] * 0.01), replace=False)]
getDatasetFromDistA(train_data_1, data_folder="traininig_data/treino_s_1/", ti=0, tf=25, norm=False)


print("Generating validation data")
# Original validation dataset
validation_data = sample_set[:, 0:int(T * 0.5)]
getDatasetFromDistA(validation_data, data_folder="traininig_data/validation_s/", ti=0, tf=25, norm=False)


# Original training dataset
train_data = sample_set[:, int(T * 0.5):]
getDatasetFromDistA(train_data_5, data_folder="traininig_data/treino_s_lr/",r=5*100, ti=0, tf=25, norm=False)