import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import numpy as np


with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)
    
with open('./all_path.pkl', 'rb') as f:
    all_path = pickle.load(f)
    
num_agents = all_path[2]
ratio = all_path[4]
master_ratio = round(1/ratio, 3)
master_election = all_path[0]
pop_type = all_path[3]
master_r = f'1to{ratio}'
agents = num_agents
fold = f'{grid_size}x{grid_size}' 
x_axis_limit = 100
y_axis_limit = 100

if grid_size == 27 and pop_type == 'sparse' : 
    total_population = 41
    initial_aoi = 31
elif grid_size == 27 and pop_type == 'semi_dense' : 
    total_population = 105
    initial_aoi = 61
elif grid_size == 27 and pop_type == 'dense' : 
    total_population = 525
    initial_aoi = 103
elif grid_size == 45 and pop_type == 'sparse' : 
    total_population = 105
    initial_aoi = 90
elif grid_size == 63 and pop_type == 'sparse' : 
    total_population = 180
    initial_aoi = 129

save_path = f'./Analysis/{master_election}/{master_r}/{agents}/{pop_type}/{fold}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

        
def generate_random_color() -> str:
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def load_pickle_files(directory, file_pattern):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and file_pattern in filename:
            with open(os.path.join(directory, filename), 'rb') as f:
                data.append(pickle.load(f))
    return data

###### Methods for agent_averages ######

def agent_averages():
    max_exp_folders = 5
    file_pattern = "agent_averages.pkl"
    data_dict = {}
    for directory in directories:
        experiment_data = []
        exp_folder_count = 0
        for exp_folder in os.listdir(directory):
            if exp_folder_count >= max_exp_folders:
                break
            exp_folder_path = os.path.join(directory, exp_folder)
            if os.path.isdir(exp_folder_path):
                exp_data = load_pickle_files(exp_folder_path, file_pattern)
                experiment_data.extend(exp_data)
                exp_folder_count += 1
        if experiment_data:
            averaged_data = average_exp_data(experiment_data)
            data_dict[os.path.basename(directory)] = averaged_data
    print("exp_folder_count = ", exp_folder_count)
    return data_dict

def average_exp_data(exp_data_list):
    combined_data = defaultdict(lambda: [0, 0, 0, 0])  # [sum_metric1, sum_metric2, sum_metric3, count]
    for sublist in exp_data_list:
        for entry in sublist:
            key = entry[0]  # Index 0 value is the key
            metrics = entry[1:]  # Metrics are from index 1 onwards

            # Accumulate values
            combined_data[key][0] += metrics[0]
            combined_data[key][1] += metrics[1]
            combined_data[key][2] += metrics[2]
            combined_data[key][3] += 1  # Count occurrences

    # Calculate averages and format the result
    averaged_data = []
    for key, values in sorted(combined_data.items()):
        sum_metric1 = values[0]
        sum_metric2 = values[1]
        sum_metric3 = values[2]
        count = values[3]
        avg_metric1 = sum_metric1 / count if count > 0 else 0
        avg_metric2 = sum_metric2 / count if count > 0 else 0
        avg_metric3 = int(sum_metric3 / count) if count > 0 else 0
        averaged_data.append([key, avg_metric1, avg_metric2, avg_metric3])

    return averaged_data


def plot_data_subplots(data_dict):
    filename = f"p2_g1.pdf"
    full_path = f"{save_path}/{filename}"
    # print(full_path)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for key, values in data_dict.items():
        x = [entry[3] for entry in values]  # Index 0 values for x-axis
        y1 = [entry[1] for entry in values]  # Index 1 values for y-axis in first subplot
        y2 = [entry[2] for entry in values]  # Index 2 values for y-axis in second subplot

        if key == 'c1':
            key = 'PB-PAPP'
        
        axs[0].plot(x, y1, label=key)
        axs[1].plot(x, y2, label=key)

    title1 = f'Survivors per steps [Total Survivors = {total_population}] on {grid_size}x{grid_size} Grid'
    axs[0].set_title(title1)
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Survivors')
    axs[0].set_xlim(10, x_axis_limit)
    axs[0].set_ylim(0, y_axis_limit)
    axs[0].legend(loc = 'upper left')
    axs[0].grid(True)

    title2 = f'AOI per steps [Total AOI = {initial_aoi}] on {grid_size}x{grid_size} Grid'
    axs[1].set_title(title2)
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('AOI')
    axs[1].set_xlim(10, x_axis_limit)
    axs[1].set_ylim(0, y_axis_limit)
    axs[1].legend(loc = 'upper left')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(full_path)
    # plt.show()


def agents_survivor_detections_per_interval_plots2(data_dict):
    # First plot (x, y1)
    
    filename1 = f"p3_{grid_size}x{grid_size}_{pop_type}_single1.pdf"
    full_path1 = f"{save_path}/{filename1}"
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for key, values in data_dict.items():
        x = [entry[0] for entry in values]  # Index 0 values for x-axis
        y1 = [(entry[1]/total_population)*100 for entry in values] # Index 1 values for y-axis in first plot

        if key == 'c1':
            key = 'PB-PAPP'

        ax1.plot(x, y1, label=key)

    title1 = f'Survivors per steps [Total Survivors = {total_population}] on {grid_size}x{grid_size} Grid'
    ax1.set_title(title1)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Survivors')
    ax1.set_ylim(0, y_axis_limit)
    ax1.legend(loc='upper left')
    ax1.set_xlim(10, x_axis_limit)
    ax1.set_ylim(0, y_axis_limit)
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(full_path1)
    plt.close(fig1)

    # Second plot (x, y2)
    filename2 = f"p3_{grid_size}x{grid_size}_{pop_type}_single2.pdf"
    full_path2 = f"{save_path}/{filename2}"
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for key, values in data_dict.items():
        x = [entry[0] for entry in values]  # Index 0 values for x-axis
        y2 = [(entry[2]/total_population)*100 for entry in values]  # Index 2 values for y-axis in second plot

        if key == 'c1':
            key = 'PB-PAPP'

        ax2.plot(x, y2, label=key)

    title2 = f'AOI per steps [Total AOI = {initial_aoi}] on {grid_size}x{grid_size} Grid'
    ax2.set_title(title2)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('AOI')
    ax2.set_xlim(10, x_axis_limit)
    ax2.set_ylim(0, y_axis_limit)
    ax2.legend(loc='upper left')
    ax2.set_xlim(10, x_axis_limit)
    ax2.set_ylim(0, y_axis_limit)
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(full_path2)
    plt.close(fig2)

    # Third plot (side by side, combination of both graphs)
    filename3 = f"p3_{grid_size}x{grid_size}_{pop_type}_combined.pdf"
    full_path3 = f"{save_path}/{filename3}"
    fig3, axs = plt.subplots(1, 2, figsize=(15, 6))
    for key, values in data_dict.items():
        x = [entry[0] for entry in values]
        y1 = [(entry[1]/total_population)*100 for entry in values]
        y2 = [(entry[2]/total_population)*100 for entry in values]

        if key == 'c1':
            key = 'PB-PAPP'

        axs[0].plot(x, y1, label=key)
        axs[1].plot(x, y2, label=key)

    # First subplot settings
    axs[0].set_title(title1)
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Survivors')
    axs[0].set_ylim(0, y_axis_limit)
    axs[0].legend(loc='upper left')
    axs[0].set_xlim(10, x_axis_limit)
    axs[0].set_ylim(0, y_axis_limit)
    axs[0].grid(True)

    # Second subplot settings
    axs[1].set_title(title2)
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('AOI')
    axs[1].set_xlim(10, x_axis_limit)
    axs[1].set_ylim(0, y_axis_limit)
    axs[1].legend(loc='upper left')
    axs[1].set_xlim(10, x_axis_limit)
    axs[1].set_ylim(0, y_axis_limit)
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(full_path3)
    plt.close(fig3)



###### Methods for agents_step_counts ######

def agents_step_counts():
    file_pattern = "agents_step_counts.pkl"
    data_dict = {}
    for directory in directories:
        experiment_data = []
        for exp_folder in os.listdir(directory):
            exp_folder_path = os.path.join(directory, exp_folder)
            if os.path.isdir(exp_folder_path):
                exp_data = load_pickle_files(exp_folder_path, file_pattern)
                experiment_data.extend(exp_data)
        # print(experiment_data)
        if experiment_data:
            averaged_data = average_agents_step_counts(experiment_data)
            data_dict[os.path.basename(directory)] = averaged_data
    return data_dict

def average_agents_step_counts(exp_data_list):
    combined_data = defaultdict(float)
    count_data = defaultdict(int)

    # Iterate over each dictionary in the list
    for exp_data in exp_data_list:
        for key, value in exp_data.items():
            combined_data[key] += value
            count_data[key] += 1

    # Calculate averages
    averaged_data = {}
    for key, total in combined_data.items():
        averaged_data[key] = total / count_data[key]

    return averaged_data

def agents_step_counts_plot(data):
    filename = f"p2_{num_agents}_{grid_size}x{grid_size}_{pop_type}_agents_step_counts.pdf"
    full_path = f"{save_path}/{filename}"
    # print(full_path)
    keys = list(data.keys())
    metrics = list(next(iter(data.values())).keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    num_keys = len(keys)
    num_metrics = len(metrics)
    bar_width = 0.15
    indices = np.arange(num_keys)
    bar_positions = [indices + i * bar_width for i in range(num_metrics)]
    for i, metric in enumerate(metrics):
        values = [data[key][metric] for key in keys]
        ax.bar(bar_positions[i], values, bar_width, label=f'Agent {metric}')
    
    title = f'Average Agent Steps on {grid_size}x{grid_size} Grid with {grid_size*grid_size} cells'
    ax.set_xlabel('Baseline Methods')
    ax.set_ylabel('Steps')
    ax.set_title(title)
    ax.set_xticks(indices + bar_width * (num_metrics - 1) / 2)
    ax.set_xticklabels(keys, rotation=0)
    ax.legend()

    plt.tight_layout()
    plt.savefig(full_path)
    # plt.show()

####### Methods for agents_survior_detections_per_interval ########

def agents_survior_detections_per_interval():
    file_pattern = "agents_survior_detections_per_interval.pkl"
    data_dict = {}
    for directory in directories:
        experiment_data = []
        for exp_folder in os.listdir(directory):
            exp_folder_path = os.path.join(directory, exp_folder)
            if os.path.isdir(exp_folder_path):
                exp_data = load_pickle_files(exp_folder_path, file_pattern)
                experiment_data.extend(exp_data)
        # print(experiment_data)
        if experiment_data:
            averaged_data = average_agents_survior_detections_per_interval(experiment_data)
            data_dict[os.path.basename(directory)] = averaged_data
    return data_dict
    

def average_agents_survior_detections_per_interval(exp_data_list):
    combined_data = defaultdict(lambda: [0, 0, 0])
    count_data = defaultdict(lambda: [0, 0, 0])
    length_of_exp_data = len(exp_data_list)
    # Determine the minimum length of the sublists
    min_length = min(len(sublist) for sublist in exp_data_list)
    
    # Trim all sublists to the minimum length
    trimmed_exp_data_list = [sublist[:min_length] for sublist in exp_data_list]

    # Iterate over each trimmed sublist of data
    for sublist in trimmed_exp_data_list:
        for entry in sublist:
            key = entry[0]  # Assuming entry[0] is the key identifier
            metrics = entry[1:]  # Assuming metrics are from entry[1:] onwards

            for i in range(len(metrics)):
                combined_data[key][i] += metrics[i]
                count_data[key][i] += 1

    # Calculate averages
    averaged_data = []
    last_seen_key = -float('inf')
    last_seen_value1 = -float('inf')
    for key in sorted(combined_data.keys()):
        averaged_metrics = [combined_data[key][i] / length_of_exp_data for i in range(len(combined_data[key]))]
        if key >= last_seen_key and averaged_metrics[0] >= last_seen_value1 :
            averaged_data.append([key] + averaged_metrics)
            last_seen_key = key
            last_seen_value1 = averaged_metrics[0]
    return averaged_data

def agents_agents_survior_detections_per_interval_plots(data_dict):
    filename = f"p3_{grid_size}x{grid_size}_{pop_type}_agents_survior_detections_per_interval.pdf"
    full_path = f"{save_path}/{filename}"
    # print(full_path)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for key, values in data_dict.items():
        x = [entry[0] for entry in values]  # Index 0 values for x-axis
        y1 = [entry[1] for entry in values] # Index 1 values for y-axis in first subplot
        y2 = [entry[2] for entry in values]  # Index 2 values for y-axis in second subplot
        print(key)

        if key == 'c1':
            key = 'PB-PAPP'
        
        axs[0].plot(x, y1, label=key)
        axs[1].plot(x, y2, label=key)

    title1 = f'Survivors per steps [Total Survivors = {total_population}] on {grid_size}x{grid_size} Grid'
    axs[0].set_title(title1)
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Survivors')
    # axs[0].set_xlim(10, x_axis_limit)
    axs[0].set_ylim(0, y_axis_limit)
    axs[0].legend(loc = 'upper right')
    axs[0].grid(True)

    title2 = f'AOI per steps [Total AOI = {initial_aoi}] on {grid_size}x{grid_size} Grid'
    axs[1].set_title(title2)
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('AOI')
    axs[1].set_xlim(10, x_axis_limit)
    axs[0].set_ylim(0, y_axis_limit)
    axs[1].legend(loc = 'upper right')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(full_path)
    # plt.show()
    
data_path = f'./results'
directories = [
    f"{data_path}/{master_election}/{master_r}/{agents}/{pop_type}/{fold}/c1",
]

data_dict = agent_averages()
agents_survivor_detections_per_interval_plots2(data_dict)
