import os
import h5py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
# Load the dataset (assuming it's provided)
df = pd.read_csv("inf_data.csv")  # Uncomment if you need to load from a file

data=df
print(data)
# Call the function on the provided dataset
# Step 1: Filter rows for Problem C and extract 'Inf Speed' values
problem_c_speeds = df[df['Problem'] == 'C']['Inf speed'].reset_index(drop=True)

# Step 2: Update 'Inf Speed' for Problems A and B with the corresponding Problem C values
df.loc[df['Problem'] == 'A', 'Inf speed'] = problem_c_speeds.values
df.loc[df['Problem'] == 'B', 'Inf speed'] = problem_c_speeds.values

# Map problems to colors (example: 'C' -> blue, 'B' -> green, etc.)
problem_colors = {'C': 'blue', 'B': 'green', 'A': 'red'}
data['Color'] = data['Problem'].map(problem_colors)

# Scatter plot: Inf speed vs Accuracy, colored by Problem
plt.figure(figsize=(10,8))
#plt.grid(True, linestyle='--', alpha=0.6)

print(data)
fastest_A = data[data['Problem'] == 'A'].sort_values(by='Inf speed', ascending=True).iloc[0]
print(fastest_A)
# Find the fastest for problem B
fastest_B = data[data['Problem'] == 'B'].sort_values(by='Inf speed', ascending=True).iloc[0]

# Find the most accurate for problem B
most_accurate_B = data[data['Problem'] == 'B'].sort_values(by='Final Error 1').iloc[0]

# Create a list of points to annotate
points_to_annotate = [
    (fastest_A['Inf speed'], fastest_A['Final Error 1'], 'P1', 'red'),
    (fastest_B['Inf speed'], fastest_B['Final Error 1'], 'P3', 'green'),
    (most_accurate_B['Inf speed'], most_accurate_B['Final Error 1'], 'P2', 'green')
]
print(points_to_annotate)
# Highlight points to annotate with circles
# Highlight points to annotate with circles
for x, y, label, color in points_to_annotate:
    plt.scatter(x, y, facecolors='none', edgecolors='black', s=260, linewidth=3, zorder=3)

# Scatter plot for all data points
for problem, group in data.groupby('Problem'):
    plt.scatter(group['Inf speed'], group['Final Error 1'], label=f'Problem {problem}', 
                color=problem_colors.get(problem, 'gray'), alpha=0.7, edgecolors='k', s=40, zorder=2)

# Annotate points with text
for x, y, label, color in points_to_annotate:
    plt.text(x * 1.05, y, label, fontsize=25, color='black', zorder=4)

plt.xlabel('Inference Speed', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.legend(title='Problem', fontsize=18)
plt.yscale("log")
plt.xscale("log")
plt.axvline(x=0.528, color='grey', linestyle='--', label='Numerical Solution')
plt.axhline(y=0.003, color='grey', linestyle='--', label='Numerical Solution')
# Adjust the number of ticks on both axes

# Set major ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))

# Set minor ticks
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.minorticks_on()
plt.tick_params(axis='both', which='major', length=6, width=1, direction='in', labelsize=14)
plt.tick_params(axis='both', which='minor', length=4, width=0.5, direction='in', labelsize=14)




plt.tight_layout()
plt.savefig('inf_speed_vs_accuracy.png', dpi=300)
plt.show()

