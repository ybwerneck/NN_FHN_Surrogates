import os
import h5py
import numpy as np
import pandas as pd
import pickle


def custom_merge(df, reference_df, key_df="Hidden Layers", key_ref="layers"):
    """
    Custom merge function to handle lists in DataFrame columns.

    Parameters:
        df (pd.DataFrame): The main DataFrame to enrich.
        reference_df (pd.DataFrame): The reference DataFrame with additional information.
        key_df (str): The column name in df to match.
        key_ref (str): The column name in reference_df to match.

    Returns:
        pd.DataFrame: The enriched DataFrame.
    """
   # print(df)
    # Use a helper function to normalize lists for comparison
    def normalize(value):
        if isinstance(value, list):
            return str(value)  # Convert lists to strings for comparison
        return value

    # Create normalized keys for both DataFrames
    df["_merge_key"] = df[key_df].apply(normalize)
    reference_df["_merge_key"] = reference_df[key_ref].apply(normalize)

    # Perform the merge on the normalized keys
    merged_df = df.merge(reference_df, left_on="_merge_key", right_on="_merge_key", how="left").drop(columns=["_merge_key"])

    return merged_df
def enrich_with_reference(df, reference_df):
    """
    Enrich a DataFrame with additional information from a reference DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to enrich.
        reference_df (pd.DataFrame): The reference DataFrame with additional information.

    Returns:
        pd.DataFrame: The enriched DataFrame.
        
    """

    # Merge based on the 'Hidden Layers' field
    print(reference_df,df)
    enriched_df = custom_merge(df,reference_df)
    print(enriched_df)
    return enriched_df

def process_folder_to_csv(base_dir, output_csv_path,Prob="--"):
    """
    Process all simulation folders in the base directory and save results to a CSV file.

    Parameters:
        base_dir (str): Path to the base directory containing simulation folders.
        output_csv_path (str): Path to save the resulting CSV file.
    """
    columns = ["Folder", "Final Error 1", "Final Error 2", "Hidden Layers", "Pinn Info", "Batch Size","Problem"]
    results_table = pd.DataFrame(columns=columns)

    # Iterate through each subfolder
    for subdir in os.listdir(base_dir):
        try:
            if os.path.isdir(os.path.join(base_dir, subdir)) and subdir.startswith('simulation'):
                h5_file_path = os.path.join(base_dir, subdir, 'val_err.h5')
                pickle_file_path = os.path.join(base_dir, subdir, 'my_dict.pkl')

                if os.path.isfile(h5_file_path):
                    with h5py.File(h5_file_path, 'r') as f:
                        print(f"Processing folder: {subdir}")
                        learning_curve = np.array(f['error_stats'])
                        try:
                            final_value1 = learning_curve.T[0][-1]
                            final_value2 = learning_curve.T[1][-1]
                        except IndexError:
                            final_value1, final_value2 = np.nan, np.nan

                # Extract model information from pickle file
                if os.path.isfile(pickle_file_path):
                    with open(pickle_file_path, 'rb') as pf:
                        sim_data = pickle.load(pf)

                    model_info = sim_data.get('model_params', {})
                    pinn = model_info.get('pinn', "N/A")
                    bs = model_info.get('bs', "N/A")
                    hidden_layers = model_info.get('hidden_layers', "N/A")
                else:
                    pinn = "N/A"
                    bs = "N/A"
                    hidden_layers = "N/A"

                # Add the extracted data to the DataFrame
                results_table.loc[len(results_table)] = [
                    subdir, final_value1, final_value2, hidden_layers, pinn, bs,Prob
                ]

        except Exception as e:
            print(f"Skipping folder {subdir} due to an error: {e}")

    # Save the results to a CSV file
    print("aasdas")
    
    results=enrich_with_reference(results,pd.read_csv("model_zoo.csv"))
    results_table.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


def combine_csv_files(csv_files, combined_csv_path):
    """
    Combine multiple CSV files into a single CSV file.

    Parameters:
        csv_files (list of str): List of paths to the CSV files to combine.
        combined_csv_path (str): Path to save the combined CSV file.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    combined_df = pd.DataFrame()

    for csv_file in csv_files:
        try:
            print(f"Reading file: {csv_file}")
            df = pd.read_csv(csv_file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Skipping file {csv_file} due to an error: {e}")

    # Save the combined DataFrame to a new CSV
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined results saved to {combined_csv_path}")
    return combined_df

import os
import h5py
import numpy as np
import pandas as pd
import pickle

def process_and_combine_from_dict(input_dict, output_csv_path):
    """
    Processes subdirectories from a dictionary and concatenates the results into a single CSV.

    Parameters:
        input_dict (dict): Dictionary where keys are problem names and values are subdirectory paths.
        output_csv_path (str): Path to save the final concatenated CSV file.
    """
    combined_df = pd.DataFrame()

    # Process each problem and subdirectory
    for problem_name, subdir in input_dict.items():
        try:
            # Initialize a DataFrame for this problem
            columns = ["Problem", "Folder", "Final Error 1", "Final Error 2", "Hidden Layers", "Pinn Info", "Batch Size"]
            results_table = pd.DataFrame(columns=columns)

                    # Iterate through simulation folders
            # Iterate through subdirectories
            for folder in os.listdir(subdir):
                folder_path = os.path.join(subdir, folder)
                
                # Check if the folder is a valid directory and matches the criteria
                if os.path.isdir(folder_path) and folder.startswith('simulation'):
                    h5_file_path = os.path.join(folder_path, 'val_err.h5')
                    speed_path = os.path.join(folder_path, 'speed_comp_results.csv')
                    pickle_file_path = os.path.join(folder_path, 'my_dict.pkl')

                    # Initialize placeholders for data
                    final_value1, final_value2 = np.nan, np.nan
                    avg_time = np.nan

                    # Extract learning curve data from h5 file
                    if os.path.isfile(h5_file_path):
                        try:
                            with h5py.File(h5_file_path, 'r') as f:
                                learning_curve = np.array(f['error_stats'])
                                try:
                                    final_value1 = learning_curve.T[0][-1]  # Last value from the first column
                                    final_value2 = learning_curve.T[1][-1]  # Last value from the second column
                                except IndexError:
                                    pass  # Use default NaN values if indexing fails
                        except Exception as e:
                            print(f"Error reading {h5_file_path}: {e}")

                    # Read average time from speed CSV
                    if os.path.isfile(speed_path):
                        try:
                            speed_data = pd.read_csv(speed_path)
                            if 'Avg Time' in speed_data.columns:
                                avg_time = speed_data['Avg Time'].mean()  # Compute average time
                        except Exception as e:
                            print(f"Error reading {speed_path}: {e}")
                    # Read average time from speed CSV   
                    else:
                        final_value1, final_value2 = np.nan, np.nan
   
                    # Extract model info from pickle
                    if os.path.isfile(pickle_file_path):
                        with open(pickle_file_path, 'rb') as pf:
                            sim_data = pickle.load(pf)
                            model_info = sim_data.get('model_params', {})
                           # print(sim_data)
                           # print(model_info)
                            pinn = model_info.get('pinn', "N/A")
                            bs = model_info.get('training_set', "N/A")
                            hidden_layers = model_info.get('hidden_layers', "N/A")
                    else:
                        pinn = "N/A"
                        bs = "N/A"
                        hidden_layers = "N/A"

                    # Append data to results table
                    results_table.loc[len(results_table)] = [
                        problem_name, folder, final_value1, final_value2, hidden_layers, pinn, bs
                    ]

            # Concatenate results for this problem into the combined DataFrame
            combined_df = pd.concat([combined_df, results_table], ignore_index=True)
            print(f"Processed problem: {problem_name}")

        except Exception as e:
            print(f"Skipping problem {problem_name} due to an error: {e}")

    # Save the final combined CSV
    combined_df=enrich_with_reference(combined_df,pd.read_csv("model_zoo.csv"))
    combined_df=combined_df.drop(columns=["layers","id"])
    combined_df.to_csv(output_csv_path, index=False)
    print(f"Final combined CSV saved to {output_csv_path}")
    return combined_df




def plot_model_errors_vs_neurons(df, error_col, neuron_col, shape_col):
    """
    Plots model errors against total neurons, creating separate plots for each shape.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        error_col (str): Column name for error values.
        neuron_col (str): Column name for total neurons.
        shape_col (str): Column name for model shapes.

    Returns:
        None
    """
    basic_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    unique_shapes = df[shape_col].unique()
    
    # Assign colors based on shape
    shape_colors = {shape: basic_colors[i % len(basic_colors)] for i, shape in enumerate(unique_shapes)}
    s = 0
    plt.figure(figsize=(8, 5))

    # Create separate plots for each shape
    for shape in unique_shapes:
        shape_mask = df[shape_col] == shape

        plt.scatter(
            df[neuron_col][shape_mask], df[error_col][shape_mask],
            color=[shape_colors[shape]] * shape_mask.sum(),
            s=25,  # Marker size
            label=f'Shape: {shape}'
        )

        plt.title(f'Model Errors vs. Total Neurons ({shape})', fontsize=14)
        plt.xlabel('Total Neurons', fontsize=12)
        plt.ylabel(error_col, fontsize=12)
        plt.yscale("log")
        plt.ylim(1e-3, 1e-1)
        plt.legend(title="Shape", fontsize=10)
        plt.grid(alpha=0.3)
        s += 1

    plt.savefig("r.png")

# Example Usage:
# Assuming df is your DataFrame

import pandas as pd
import matplotlib.pyplot as plt

# Example Usage:
# Assuming `df` is your DataFrame

input_dict = {
    
    "A": "Problem_A2/Problem_A_results/",
    "C": "Problem_C/Problem_C_Results/",
   # "B": "Problem_B/Problem_B_Results/",
#    "B_it": "Problem_D/Problem_D_Results/",

}
models=process_and_combine_from_dict(input_dict, "all.csv")


print(models)
# Computing median errors without 'Pinn Info'
median_errors = models.groupby(["total_neurons"])["Final Error 1"].mean()
plot_model_errors_vs_neurons(models, error_col="Final Error 1", neuron_col="total_neurons", shape_col="shape")

# Print the result
print(median_errors)
print(models)

df=models
# Unique problems
problems = df["Problem"].unique()
plt.figure(figsize=(8, 5))

# Create individual plots for each problem
for problem in problems:
    
    # Filter data by problem
    subset = df[df["Problem"] == problem]
    
    # Scatter plot
    plt.scatter(subset["total_neurons"], subset["Final Error 1"], label=f"MSE-Problem {problem}", alpha=0.7)
    
    # Plot settings
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total Neurons")
    plt.ylabel("MAE")
    plt.title("Mae vs Total Neurons")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save each figure


# Create boxplots for Final Error 1 and Final Error 2 per problem
plt.figure(figsize=(12, 6))

# Extract unique problems
problems = ["A","B","C"]
# Create a single figure with subplots for each problem
fig, axes = plt.subplots(1, len(problems), figsize=(15, 6), sharey=True)

for ax, problem in zip(axes, problems):
    print(problems)
    subset = df[df["Problem"] == problem]
    print(subset)
    neuron_sizes = sorted(subset["total_neurons"].unique())

    # Create a boxplot of Final Error 1 for each neuron 
    print(neuron_sizes)
    ax.boxplot([subset[subset["total_neurons"] == neurons]["Final Error 1"] for neurons in neuron_sizes],
               labels=neuron_sizes)
    
    ax.set_xlabel("Total Neurons")
    ax.set_yscale("log")
    ax.set_title(f"Problem {problem}")
    ax.grid(True)

axes[0].set_ylabel("MAE 1")
fig.suptitle("MAE Distribution by Neuron Count for Each Problem", fontsize=14)

plt.tight_layout()
plt.savefig("rs.png")


# Ensure the correct column names
neuron_column = "total_neurons"  # Proxy for total neurons
shape_column = "shape"  # Architecture shape
error_column = "Final Error 1"  # MAE

# Select a specific problem (assuming user wants to specify)

for A in range(2):
    selected_problem = problems[A]  # Change this to the desired problem

    # Filter dataset for the selected problem
    subset = df[df["Problem"] == selected_problem]

    # Get unique neuron sizes and shapes
    neuron_sizes = sorted(subset[neuron_column].unique())
    shapes = sorted(subset[shape_column].unique())

    # Offset positions for side-by-side boxplots
    offsets = np.linspace(-0.2, 0.2, len(shapes))

    # Create figure
    plt.figure(figsize=(12, 6))

    for i, shape in enumerate(shapes):
        boxplot_data = [subset[(subset[neuron_column] == neurons) & (subset[shape_column] == shape)][error_column]
                        for neurons in neuron_sizes]
        
        # Calculate x positions with offsets
        x_positions = np.array(range(len(neuron_sizes))) + offsets[i]

        # Create boxplot
        plt.boxplot(boxplot_data, positions=x_positions, widths=0.1, patch_artist=True, boxprops=dict(facecolor=f"C{i}"))

    # Labels and formatting
    plt.xticks(range(len(neuron_sizes)), neuron_sizes, rotation=45)
    plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("Total Neurons (Batch Size)")
    plt.ylabel("MAE")
    if(A>0):
        plt.ylim((2e-3,5e-2))
    plt.title(f"MAE Distribution by Neuron Count & Shape for Problem {selected_problem}")
    plt.legend(handles=[plt.Line2D([0], [0], color=f"C{i}", lw=4, label=shape) for i, shape in enumerate(shapes)],
            title="Shape")
#    plt.grid(True, which="both", linestyle="--", linewidth=0.5)



    plt.savefig(f"r{A}.pdf")


import pandas as pd

def summarize_model_performance(df, error_col="Final Error 1", problem_col="Problem"):
    """
    Computes and prints the best, worst, and median model for each problem based on the error column.

    Parameters:
        df (pd.DataFrame): DataFrame containing model performance data.
        error_col (str): Column name for error values.
        problem_col (str): Column name for problem names.
    """
    summary = []

    for problem in df[problem_col].unique():
        subset = df[df[problem_col] == problem]
        
        # Compute best, worst, and median model
        best_model = subset.loc[subset[error_col].idxmin()]
        worst_model = subset.loc[subset[error_col].idxmax()]
        median_model = subset.loc[(subset[error_col] - subset[error_col].median()).abs().idxmin()]

        summary.append({
            "Problem": problem,
            "Best Model": best_model["Hidden Layers"],
            "Best Error": best_model[error_col],
            "Median Model": median_model["Hidden Layers"],
            "Median Error": median_model[error_col],
            "Worst Model": worst_model["Hidden Layers"],
            "Worst Error": worst_model[error_col],
        })
        print(summary)

    # Convert to DataFrame for better visualization
    summary_df = pd.DataFrame(summary)
    print(summary_df)
    summary_df.to_csv("summary.csv", index=False)

# Example usage
summarize_model_performance(df)
# Save the model performance summary
import pandas as pd

def summarize_model_performance_separated(df, error_col="Final Error 1", problem_col="Problem", layers_col="Hidden Layers", neurons_col="total_neurons"):
    """
    Computes and prints the best, worst, and median model for each problem and neuron count, separating two-layer and three-layer models.

    Parameters:
        df (pd.DataFrame): DataFrame containing model performance data.
        error_col (str): Column name for error values.
        problem_col (str): Column name for problem names.
        layers_col (str): Column name indicating number of hidden layers.
        neurons_col (str): Column name for total neurons.
    """
    summary = []

    for problem in df[problem_col].unique():
        for neurons in df[neurons_col].unique():
            for layer_count in [2, 3]:  # Separate models with two and three layers
                subset = df[(df[problem_col] == problem) & (df[neurons_col] == neurons) & (df[layers_col] == layer_count)]
                
                if subset.empty:
                    continue

                # Compute best, worst, and median model
                best_model = subset.loc[subset[error_col].idxmin()]
                worst_model = subset.loc[subset[error_col].idxmax()]
                median_model = subset.loc[(subset[error_col] - subset[error_col].median()).abs().idxmin()]

                summary.append({
                    "Problem": problem,
                    "Neurons": neurons,
                    "Layers": layer_count,
                    "Best Model": best_model[layers_col],
                    "Best Error": best_model[error_col],
                    "Median Model": median_model[layers_col],
                    "Median Error": median_model[error_col],
                    "Worst Model": worst_model[layers_col],
                    "Worst Error": worst_model[error_col],
                })

    # Convert to DataFrame for better visualization
    summary_df = pd.DataFrame(summary)
    

    summary_df.to_csv("summary.csv", index=False)

# Load the dataset (assuming it's provided)
# df = pd.read_csv("your_data.csv")  # Uncomment if you need to load from a file

# Call the function on the provided dataset
summarize_model_performance_separated(df)
