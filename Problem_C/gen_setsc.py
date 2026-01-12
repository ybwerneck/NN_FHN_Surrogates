import numpy as np
import matplotlib.pyplot as plt
import os
import chaospy as cp
import pandas as pd
import sys
import os
# Set up the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../data_generator'))

from set_generator import getDatasetFromDistA

# Parameters
target_length = int(1e5)  # Desired number of rows
noise_level = 0.01  # Adjust the noise level to be "little noise"

# Read the dataset (assuming it has columns 'u', 'v', 'k')
df = pd.read_csv('filtered_df.csv')  # Replace 'your_dataset.csv' with the path to your file

# Extract the relevant columns (assuming the order is 'u', 'v', 'k')
sample_set = df[['u', 'w', 'k']].values  # Convert to a numpy array

# Shuffle the sample set
np.random.shuffle(sample_set)

# Check the current length of the sample set
current_length = sample_set.shape[0]

# If the current length is less than the target length, we need to add more samples
if current_length < target_length:
    # Determine how many rows are needed to reach the target length
    rows_needed = target_length - current_length

    # Sample rows randomly from the original set and add small noise
    additional_samples = np.tile(sample_set, (rows_needed // current_length + 1, 1))[:rows_needed]
    
    # Add small noise to the additional samples
    noise = noise_level * np.random.randn(*additional_samples.shape)
    additional_samples += noise

    # Concatenate the additional samples to the original sample set
    sample_set = np.vstack((sample_set, additional_samples))

# Shuffle the final set again
np.random.shuffle(sample_set)

# Transpose the set if needed
sample_set = sample_set

# Now sample_set has the desired length with added noise
print("Final sample_set shape:", sample_set.shape)
sample_set = sample_set.T

print(sample_set.T[0])
print("Generating training data")
getDatasetFromDistA(sample_set, data_folder="training_data/treino_r/", ti=0, tf=20,r=10,norm=False)
