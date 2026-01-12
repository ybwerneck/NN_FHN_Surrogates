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
    print(df)
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
            columns = ["Problem", "Folder", "Final Error 1", "Final Error 2", "Hidden Layers", "Pinn Info", "Batch Size","Training_Tine"]
            results_table = pd.DataFrame(columns=columns)

            # Iterate through simulation folders
            for folder in os.listdir(subdir):
                folder_path = os.path.join(subdir, folder)
                if os.path.isdir(folder_path) and folder.startswith('simulation'):
                    h5_file_path = os.path.join(folder_path, 'val_err.h5')
                    pickle_file_path = os.path.join(folder_path, 'my_dict.pkl')
                    with open(os.path.join(folder_path, 'timing.txt'), "r") as file:

                        time = float(file.read().strip())
                    # Extract learning curve data
                try:
                    if os.path.isfile(h5_file_path):
                        with h5py.File(h5_file_path, 'r') as f:
                            learning_curve = np.array(f['error_stats'])
                            try:
                                final_value1 = learning_curve.T[0][-1]
                                final_value2 = learning_curve.T[1][-1]
                            except IndexError:
                                final_value1, final_value2 = np.nan, np.nan
                    else:
                        final_value1, final_value2 = np.nan, np.nan
   
                    # Extract model info from pickle
                    if os.path.isfile(pickle_file_path):
                        with open(pickle_file_path, 'rb') as pf:
                            sim_data = pickle.load(pf)
                            model_info = sim_data.get('model_params', {})
                            print(sim_data)
                            print(model_info)
                            pinn = model_info.get('pinn', "N/A")
                            bs = model_info.get('npt', "N/A")
                            hidden_layers = model_info.get('hidden_layers', "N/A")
                    else:
                        pinn = "N/A"
                        bs = "N/A"
                        hidden_layers = "N/A"

                    # Append data to results table
                    results_table.loc[len(results_table)] = [
                        problem_name, folder, final_value1, final_value2, hidden_layers, pinn, bs,time
                    ]
                except:
                     print("")
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

input_dict = {
    "Bp": "Problem_A_results/",
}
models=process_and_combine_from_dict(input_dict, "all.csv")


print(models)


import matplotlib.pyplot as plt

def plot_model_errors_vs_neurons(df, error_col, neuron_col, shape_col, pinn_col):
    """
    Plots model errors against total neurons, creating separate plots for each shape.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        error_col (str): Column name for error values.
        neuron_col (str): Column name for total neurons.
        shape_col (str): Column name for model shapes.
        pinn_col (str): Column name indicating PINN info (True/False).

    Returns:
        None
    """
    unique_shapes = df[shape_col].unique()
    colors = df[pinn_col].map({True: 'blue', False: 'orange'})
    s=0
    # Create separate plots for each shape
    for shape in unique_shapes:
        plt.figure(figsize=(8, 5))
        shape_mask = df[shape_col] == shape
        
        plt.scatter(
            df[neuron_col][shape_mask], df[error_col][shape_mask],
            c=colors[shape_mask],
            s=25,  # Marker size
            label=f'Shape: {shape}'
        )

        plt.title(f'Model Errors vs. Total Neurons ({shape})', fontsize=14)
        plt.xlabel('Total Neurons', fontsize=12)
        plt.ylabel('Final Error 1', fontsize=12)
        plt.yscale("log")
        plt.ylim(1e-3,1e-1)
        plt.legend(title="PINN Info", fontsize=10)
        plt.grid(alpha=0.3)
        s=s+1
        plt.savefig(f"r{s}.png")

# Example Usage:
# Assuming `df` is your DataFrame
plot_model_errors_vs_neurons(models, error_col="Final Error 1", neuron_col="total_neurons", shape_col="Batch Size", pinn_col="Pinn Info")

import pandas as pd

# Assuming `models` is your DataFrame
median_errors = models.groupby(["Pinn Info", "Batch Size"])["Final Error 1"].median()
median_t = models.groupby(["Pinn Info", "Batch Size"])["Training_Tine"].median()

# Print the result
print(median_errors)
print(median_t)
print(models)




import pandas as pd
import matplotlib.pyplot as plt

# Sample data (replace with your actual DataFrame)
# Option 1: Side-by-side subplots for Final Error and Training Time
def plot_side_by_side(df, save_path="side_by_side.png"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    markers = {True: 'o', False: 's'}
    colors = {True: 'blue', False: 'orange'}
    
    # Plot Final Error vs. Batch Size
    for pinn in [True, False]:
        subdf = df[df['Pinn Info'] == pinn]
        axs[0].plot(subdf['Batch Size'], subdf['Final Error 1'], 
                    marker=markers[pinn], color=colors[pinn],
                    label=f'PINN: {pinn}', linewidth=2)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Batch Size (log scale)')
    axs[0].set_ylabel('Final Error 1 (log scale)')
    axs[0].set_title('Final Error vs. Batch Size')
    axs[0].legend()
    axs[0].grid(True, which='both', ls='--', lw=0.5)
    
    # Plot Training Time vs. Batch Size
    for pinn in [True, False]:
        subdf = df[df['Pinn Info'] == pinn]
        axs[1].plot(subdf['Batch Size'], subdf['Training_Tine'], 
                    marker=markers[pinn], color=colors[pinn],
                    label=f'PINN: {pinn}', linewidth=2)
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Batch Size (log scale)')
    axs[1].set_ylabel('Training Time')
    axs[1].set_title('Training Time vs. Batch Size')
    axs[1].legend()
    axs[1].grid(True, which='both', ls='--', lw=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory
    print(f"Side-by-side plot saved as {save_path}")

# Option 2: Dual-axis plot combining Final Error and Training Time
def plot_dual_axis(df, save_path="dual_axis.png"):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    markers = {True: 'o', False: 's'}
    colors = {True: 'blue', False: 'orange'}
    
    # For each PINN condition, plot both metrics
    for pinn in [True, False]:
        subdf = df[df['Pinn Info'] == pinn]
        # Plot Final Error on left axis
        ax1.plot(subdf['Batch Size'], subdf['Final Error 1'], 
                 marker=markers[pinn], color=colors[pinn],
                 label=f'Final Error (PINN: {pinn})', linewidth=2)
        # Plot Training Time on right axis (using dashed line)
        ax2.plot(subdf['Batch Size'], subdf['Training_Tine'], 
                 marker=markers[pinn], linestyle='--', color=colors[pinn],
                 label=f'Training Time (PINN: {pinn})', linewidth=2)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size (log scale)')
    ax1.set_ylabel('Final Error 1 (log scale)')
    ax2.set_ylabel('Training Time')
    ax1.set_title('Final Error and Training Time vs. Batch Size')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory
    print(f"Dual-axis plot saved as {save_path}")

# Generate and save the figures
plot_side_by_side(models, save_path="side_by_side.png")
plot_dual_axis(models, save_path="dual_axis.png")
