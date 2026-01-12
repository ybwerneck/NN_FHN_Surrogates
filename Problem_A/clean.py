import os
import h5py
import shutil
import numpy as np
import pandas as pd
import sys
base_dir = sys.argv[1]
min_iterations = 1_000_000  # Set the minimum number of iterations

# Function to process the iterations and delete directories
def check_iterations(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        if 'error_stats' in f:
            iterations = len(f['error_stats'][()])
            return iterations
        return 0

deleted_folders = []

# Iterate through each subfolder
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    
    if os.path.isdir(subdir_path) and subdir.startswith('simulation'):
        h5_file_path = os.path.join(subdir_path, 'val_err.h5')

        # Check if the file exists
        if os.path.isfile(h5_file_path):
            iterations = check_iterations(h5_file_path)
            print(iterations)
            # If iterations are fewer than the threshold, delete the folder
            if iterations < 402:
                shutil.rmtree(subdir_path)
                deleted_folders.append(subdir)


print(f"Deleted {(deleted_folders)} folders with fewer than {min_iterations} iterations.")
