import os
from s2_1_agent_conf import num_agents, agent_configs
from s3_env_conf import GridWorld, _action_to_direction
from case1_s6_prediction_conf import PredictionCall
import random
import numpy as np
import pandas as pd
import networkx as nx
import math
import sys
import pickle
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import argparse
import time

# parser = argparse.ArgumentParser(description='Example script to pass multiple values.')

# # Define three arguments
# parser.add_argument('--lamb', type=float, required=True, help='An integer value for value1')
# parser.add_argument('--v', type=float, required=True, help='A float value for value2')
# parser.add_argument('--meu', type=float, required=True, help='A string value for value3')
# # parser.add_argument('--combination', type=str, required=True, help='A string value for value3')
# args = parser.parse_args()

# lamb = args.lamb
# v = args.v
# meu = args.meu
# # combination = args.combination
cue = 1
param = 0.001

with open('./Testing/result.pkl', 'rb') as f:
    result = pickle.load(f)
   
case = result[0]

with open('./total_population.pkl', 'rb') as f:
    tot_defined_population = pickle.load(f)

with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)

with open('./master_ratio.pkl', 'rb') as f:
    master_ratio = pickle.load(f)
    
with open('./Testing/locations.pkl', 'rb') as f:
    locations = pickle.load(f)

with open('./Testing/locations.pkl', 'rb') as f:
    locations_copy = pickle.load(f)
    
with open(result[3], 'rb') as f:
    df_with_hops_data = pickle.load(f)

# print(df_with_hops_data.columns)
with open(result[4], 'rb') as f:
    all_aoi = pickle.load(f)

with open('./all_path.pkl', 'rb') as f:
    all_path = pickle.load(f)
pop_type = f'{all_path[0]}/{all_path[1]}/{all_path[2]}/{all_path[3]}'
fold = f'{grid_size}x{grid_size}' 

def read_counter():
    try:
        with open('./Testing/counter1.txt', 'r') as file:
            return int(file.read())
    except FileNotFoundError:
        return 0

def update_counter(counter):
    with open('./Testing/counter1.txt', 'w') as file:
        file.write(str(counter + 1))

counter = read_counter()
print("c1 Counter:", counter)  

num_mds = 0
vehicle_capacity = 10000
eta = 1000
min_val = 0
max_val = grid_size

class ThreeTierArchitecture:
    def __init__(self) -> None:
        self.num_agents = num_agents
        self.num_mds = num_mds
        self.df_with_hops_data = df_with_hops_data
        self.avg_demand = 0
        self.eta = eta
        self.max_distance = self.max_dt()
        self.cue = cue
        self.param = param

    def euclidean_distance_heuristic(self, u, v):
        return np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
    
    def cal_astar_path(self, G, current_state, next_state):
        G = G
        shortest_path = nx.astar_path(G, current_state, next_state, heuristic=self.euclidean_distance_heuristic)
        return shortest_path

    def identify_action(self, current_state, next_state):
        direction_vector = np.subtract(next_state, current_state)
        for key, value in action_to_direction.items():
            # print(f'direction_vector :  {direction_vector} , value : {value} for key : {key}')
            if np.array_equal(direction_vector, value):
                return key
        return None

    def create_2d_network(self, rows, columns):
        G = nx.Graph()
        # Add nodes
        for i in range(rows):
            for j in range(columns):
                G.add_node((i, j))
        # Add horizontal and vertical edges
        for i in range(rows - 1):
            for j in range(columns - 1):
                G.add_edge((i, j), (i + 1, j))
                G.add_edge((i, j), (i, j + 1))
        # Add diagonal edges
        for i in range(rows - 1):
            for j in range(columns - 1):
                G.add_edge((i, j), (i + 1, j + 1))
                G.add_edge((i + 1, j), (i, j + 1))  # Add the second diagonal edge
        # Add edges for the last column and last row
        for i in range(rows - 1):
            G.add_edge((i, columns - 1), (i + 1, columns - 1))
        for j in range(columns - 1):
            G.add_edge((rows - 1, j), (rows - 1, j + 1))
        return G

    def distribute_agents(self, num_of_agents, grid_size):
        n = grid_size
        # Calculate the number of agents per row and column
        agents_per_row = int(math.sqrt(num_of_agents))
        agents_per_column = int(math.ceil(num_of_agents / agents_per_row))
        # Calculate the spacing between agents
        row_spacing = n // (agents_per_row + 1)
        col_spacing = n // (agents_per_column + 1)
        agent_positions = []
        # Distribute agents evenly across the grid
        for i in range(1, agents_per_row + 1):
            for j in range(1, agents_per_column + 1):
                # Calculate the position of the current agent
                x = i * row_spacing
                y = j * col_spacing
                # Ensure the position is within the grid boundaries
                if x <= n and y <= n:
                    agent_positions.append((x, y))
        return agent_positions

    def identify_fixed_mds(self, agent_configs, num_agents):
        md_agents = []
        masters = int(math.ceil(master_ratio * num_agents))
        for agent in agent_configs:
            agent.agent_type_md = ''
            if agent.idx <= masters:
                agent.agent_type_md = "MD" + str(agent.idx)
                md_agents.append(agent)
        return md_agents

    def identify_mds(self, agent_configs, num_agents):
        for agent in agent_configs:
            agent.agent_type_md = ''
        self.num_mds = max(1, int(math.ceil(master_ratio * num_agents)))
        agent_configs.sort(key=lambda x: x.battery_percentage_remaining, reverse=True)
        md_agents = agent_configs[:self.num_mds]
        for agent_config in md_agents:
            agent_config.agent_type_md = "MD" + str(agent_config.idx)
        return md_agents

    def haversine_distance(self, agent1, agent2):
        x1, y1 = agent1.agent_current_latidude, agent1.agent_current_longitude
        x2, y2 = agent2.agent_current_latidude, agent2.agent_current_longitude
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = np.radians([x1, y1, x2, y2])
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        # Radius of the Earth in kilometers
        R = 6371.0
        # Calculate the distance
        distance = R * c
        return distance


    def assign_md_to_sd(self, agents, md_agents):
        # Initialize connected_agents list for each MD agent and self-assign each MD
        for md_agent in md_agents:
            md_agent.connected_agents = [md_agent.agent_type]

        # Calculate distances and assign the nearest MD agent to each SD agent
        for sd_agent in agents:
            sd_agent.agent_distance_to_md = []
            for md_agent in md_agents:
                distance = self.euclidean_distance_heuristic(sd_agent.current_state, md_agent.current_state)
                sd_agent.agent_distance_to_md.append((distance, md_agent.agent_type_md))
            
            # Sort by distance to find the nearest MD agent
            sd_agent.agent_distance_to_md.sort(key=lambda x: x[0])
            
            # Assign the nearest MD agent
            nearest_md = sd_agent.agent_distance_to_md[0][1]
            sd_agent.agent_type_md = nearest_md

        # Update MD agents' connected agents based on assignments
        for sd_agent in agents:
            for md_agent in md_agents:
                if md_agent.agent_type_md == sd_agent.agent_type_md:
                    if sd_agent.agent_type not in md_agent.connected_agents:  # Ensure no duplicate assignments
                        md_agent.connected_agents.append(sd_agent.agent_type)
                    break  # Stop iterating once the SD agent is assigned to an MD agent

        # for md_agent in md_agents:
        #     print(f"MD Agent {md_agent.agent_type_md} has connected agents: {md_agent.connected_agents}")
        return


    def create_new_dataset_for_AOIprediction(self, agents, md_agents):
        sd_agents = [agent for agent in agents if agent.agent_type_md == '']
        for md_agent in md_agents:
            md_agent.agent_df = pd.DataFrame()
            data_list = []
            filtered_data_list = []
            for sd_agent in agents:
                if sd_agent.agent_type in md_agent.connected_agents :
                    md_agent.sd_agents_paths.extend(sd_agent.agent_path)
                    # for i in range(len(sd_agent.agent_current_interval_path)):
                    #     loc = sd_agent.agent_current_interval_path[i][1]
                    #     location_data = locations.get(loc)
                    #     if location_data: 
                    #         data_list.append(location_data)
                    # sd_agent.agent_current_interval_path.clear()
                    for i in range(len(sd_agent.agent_traversed_path)):
                        loc = sd_agent.agent_traversed_path[i][1]
                        location_data = locations.get(loc)
                        if location_data: 
                            data_list.append(location_data)
                    for data in data_list:
                        filtered_data = {key: data[key] for key in ['location', 'latitude', 'longitude','elevation', 'one_hopAOICount', 'one_hopSCount', 'one_hop_fraction','two_hopAOICount', 'two_hopSCount', 'two_hop_fraction', 'three_hopAOICount', 'three_hopSCount', 'three_hop_fraction', 'probability', 'population', 'is_aoi']}
                        filtered_data_list.append(filtered_data)
                    # print("filtered_data_list size = ", len(filtered_data_list))
                    unique_filtered_data_list = [dict(t) for t in {tuple(filtered_data.items()) for filtered_data in filtered_data_list}]
                    unique_filtered_data_list = [d for i, d in enumerate(unique_filtered_data_list) if d['location'] not in {x['location'] for x in unique_filtered_data_list[:i]}]
                    # print("unique_filtered_data_list size =", len(unique_filtered_data_list))
            frozen_tuples = [frozenset(t) for t in md_agent.sd_agents_paths]
            unique_frozen_tuples = set(frozen_tuples)
            md_agent.sd_agents_paths = [tuple(f) for f in unique_frozen_tuples]
            md_agent.agent_df = pd.DataFrame(unique_filtered_data_list)
            if not md_agent.agent_df.empty :
                md_agent.agent_df =  md_agent.agent_df[['location', 'latitude', 'longitude','elevation', 'one_hopAOICount', 'one_hopSCount', 'one_hop_fraction','two_hopAOICount', 'two_hopSCount', 'two_hop_fraction', 'three_hopAOICount', 'three_hopSCount', 'three_hop_fraction', 'probability', 'population', 'is_aoi']]
                md_agent.agent_df = self.update_one_hopSCount(md_agent.agent_df, md_agent.sd_agents_paths)
                md_agent.agent_df = self.update_two_hopSCount(md_agent.agent_df, md_agent.sd_agents_paths)
                md_agent.agent_df = self.update_three_hopSCount(md_agent.agent_df, md_agent.sd_agents_paths)
                md_agent.agent_df = self.update_population(md_agent.agent_df)
                # print("md_agent.agent_df.shape : ",md_agent.agent_df.shape)
        return

    def update_population(self, agent_df):
        for index, row in agent_df.iterrows():
            loc = row['location']
            agent_df.at[index, 'population'] = locations_copy.get(loc, {}).get('population')
        return agent_df
            

    def update_one_hopSCount(self, agent_df, only_traversed_path):
        for index, row in agent_df.iterrows():
            one_hopSCount_updated_value = 0
            traversed_count = 0
            loc = row['location'] 
            hop_locs_str = next(iter(self.df_with_hops_data.loc[self.df_with_hops_data['Location'] == loc, 'one_hop_locs'].values), [])
            set1 =  set(only_traversed_path)
            set2 = set(hop_locs_str)
            common_tuples = set1.intersection(set2)
            common_tuples_list = list(common_tuples)
            traversed_count = len(common_tuples_list)
            one_hopSCount_updated_value += sum(locations.get(tup, {}).get('population', 0) for tup in common_tuples_list if locations.get(tup, {}).get('population', 0) > 0)
            agent_df.at[index, 'one_hopSCount'] = one_hopSCount_updated_value
            agent_df.at[index, 'one_hop_fraction'] = traversed_count/9
        return agent_df
    
    def update_two_hopSCount(self, agent_df, only_traversed_path):
        for index, row in agent_df.iterrows():
            two_hopSCount_updated_value = 0
            traversed_count = 0
            loc = row['location'] 
            hop_locs_str = next(iter(self.df_with_hops_data.loc[self.df_with_hops_data['Location'] == loc, 'two_hop_locs'].values), [])
            set1 =  set(only_traversed_path)
            set2 = set(hop_locs_str)
            common_tuples = set1.intersection(set2)
            common_tuples_list = list(common_tuples)
            traversed_count = len(common_tuples_list)
            two_hopSCount_updated_value += sum(locations.get(tup, {}).get('population', 0) for tup in common_tuples_list if locations.get(tup, {}).get('population', 0) > 0)
            agent_df.at[index, 'two_hopSCount'] = two_hopSCount_updated_value
            agent_df.at[index, 'two_hop_fraction'] = traversed_count/16
        return agent_df
    
    def update_three_hopSCount(self, agent_df, only_traversed_path):
        for index, row in agent_df.iterrows():
            three_hopSCount_updated_value = 0
            traversed_count = 0
            loc = row['location'] 
            hop_locs_str = next(iter(self.df_with_hops_data.loc[self.df_with_hops_data['Location'] == loc, 'three_hop_locs'].values), [])
            set1 =  set(only_traversed_path)
            set2 = set(hop_locs_str)
            common_tuples = set1.intersection(set2)
            common_tuples_list = list(common_tuples)
            traversed_count = len(common_tuples_list)
            three_hopSCount_updated_value += sum(locations.get(tup, {}).get('population', 0) for tup in common_tuples_list if locations.get(tup, {}).get('population', 0) > 0)
            agent_df.at[index, 'three_hopSCount'] = three_hopSCount_updated_value
            agent_df.at[index, 'three_hop_fraction'] = traversed_count/25
        return agent_df
        
    def create_dataset_for_untraversedAOI(self, agents, md_agents):
        sd_agents = [agent for agent in agents if agent.agent_type_md == '']
        for md_agent in md_agents:
            md_agent.agent_untraversed_df = pd.DataFrame()
            data_list = []
            filtered_data_list = []
            for sd_agent in agents:
                if sd_agent.agent_type in md_agent.connected_agents :
                    md_agent.sd_agents_paths.extend(sd_agent.agent_path)
                    # print(f' {sd_agent.idx}  has {len(sd_agent.agent_untraversed_neighbors)} agent_untraversed_neighbors')
                    for i in range(len(sd_agent.agent_untraversed_neighbors)):
                        loc = sd_agent.agent_untraversed_neighbors[i]
                        location_data = locations.get(loc)
                        if location_data: 
                            data_list.append(location_data)
                    for data in data_list:
                        filtered_data = {key: data[key] for key in ['location', 'latitude', 'longitude', 'elevation', 'one_hopAOICount', 'one_hopSCount', 'one_hop_fraction','two_hopAOICount', 'two_hopSCount', 'two_hop_fraction', 'three_hopAOICount', 'three_hopSCount', 'three_hop_fraction', 'probability', 'population', 'is_aoi']}
                        filtered_data_list.append(filtered_data)
                    # print("filtered_data_list size = ", len(filtered_data_list))
                    unique_filtered_data_list = [dict(t) for t in {tuple(filtered_data.items()) for filtered_data in filtered_data_list}]
                    unique_filtered_data_list = [d for i, d in enumerate(unique_filtered_data_list) if d['location'] not in {x['location'] for x in unique_filtered_data_list[:i]}]
                    # print("unique_filtered_data_list size =", len(unique_filtered_data_list))
                else:
                    continue
            frozen_tuples = [frozenset(t) for t in md_agent.sd_agents_paths]
            unique_frozen_tuples = set(frozen_tuples)
            md_agent.sd_agents_paths = [tuple(f) for f in unique_frozen_tuples]
            if filtered_data_list: 
                # md_agent.agent_untraversed_df = pd.DataFrame(unique_filtered_data_list)
                md_agent.agent_untraversed_df = pd.DataFrame(filtered_data_list)
                md_agent.agent_untraversed_df =  md_agent.agent_untraversed_df[['location', 'latitude', 'longitude', 'elevation', 'one_hopAOICount', 'one_hopSCount', 'one_hop_fraction', 'two_hopAOICount', 'two_hopSCount', 'two_hop_fraction', 'three_hopAOICount', 'three_hopSCount', 'three_hop_fraction', 'probability', 'population', 'is_aoi']]
                md_agent.agent_untraversed_df = self.update_one_hopSCount(md_agent.agent_untraversed_df, md_agent.sd_agents_paths)
                md_agent.agent_untraversed_df = self.update_two_hopSCount(md_agent.agent_untraversed_df, md_agent.sd_agents_paths)
                md_agent.agent_untraversed_df = self.update_three_hopSCount(md_agent.agent_untraversed_df, md_agent.sd_agents_paths)
                # print("md_agent.agent_untraversed_df.shape : ",md_agent.agent_untraversed_df.shape)
            else:
                continue

    def get_agents_current_locations(self, agent_configs):
        agent_current_locations = []
        for agent in agent_configs:
            agent_current_locations.append([agent.idx, agent.current_state, agent.current_state_index ])
        return agent_current_locations

    def distribute_AOI_to_agents_old(self, current_locs, merged_predicted_AOI, radius=5):
        num_agents = len(current_locs)
        num_aois = len(merged_predicted_AOI)
        aois_per_agent = num_aois // num_agents
        extra_aois = num_aois % num_agents
        assigned_AOIs = []
        agent_aoi_count = defaultdict(int)
        agent_availability = {agent_id: aois_per_agent + (1 if i < extra_aois else 0) for i, (agent_id, _) in enumerate(current_locs)}
        agent_index = 0
        for loc in merged_predicted_AOI:
            agent_id, _ = current_locs[agent_index]
            assigned_AOIs.append((agent_id, loc))
            agent_aoi_count[agent_id] += 1
            agent_index = (agent_index + 1) % num_agents
        return assigned_AOIs
    
    def distribute_AOI_to_agents(self, current_locs, merged_predicted_AOI, radius=5):
        num_agents = len(current_locs)
        num_aois = len(merged_predicted_AOI)
        
        # Initialize dictionaries for tracking assignments
        assigned_AOIs = []
        agent_aoi_count = defaultdict(int)
        
        # Initialize availability for each agent
        aois_per_agent = num_aois // num_agents
        extra_aois = num_aois % num_agents
        agent_availability = {agent_id: aois_per_agent + (1 if i < extra_aois else 0) for i, (agent_id, _) in enumerate(current_locs)}
        
        # Calculate distances between each AOI and each agent
        aoi_to_agent_distances = []
        for loc in merged_predicted_AOI:
            distances = []
            for agent_id, agent_loc in current_locs:
                distance = abs(loc[0] - agent_loc[0]) + abs(loc[1] - agent_loc[1])
                if distance <= radius:  # Only consider agents within the specified radius
                    distances.append((distance, agent_id))
            distances.sort()  # Sort by distance (smallest distance first)
            aoi_to_agent_distances.append((loc, distances))
        
        # Assign AOIs to the closest agents within the radius, while balancing the load
        for loc, distances in aoi_to_agent_distances:
            if distances:  # If there are agents within the radius
                for distance, agent_id in distances:
                    if agent_availability[agent_id] > 0:
                        assigned_AOIs.append((agent_id, loc))
                        agent_aoi_count[agent_id] += 1
                        agent_availability[agent_id] -= 1
                        break  # Move to the next AOI once assigned
            else:
                # If no agent is within the radius, assign to the nearest agent
                min_distance = float('inf')
                closest_agent_id = None
                for agent_id, agent_loc in current_locs:
                    distance = abs(loc[0] - agent_loc[0]) + abs(loc[1] - agent_loc[1])
                    if distance < min_distance:
                        min_distance = distance
                        closest_agent_id = agent_id
                
                if agent_availability[closest_agent_id] > 0:
                    assigned_AOIs.append((closest_agent_id, loc))
                    agent_aoi_count[closest_agent_id] += 1
                    agent_availability[closest_agent_id] -= 1
        
        return assigned_AOIs



    def assign_AOI_to_agents(self, agents, md_agents, merged_predicted_AOI):
        current_locs = [[agent.agent_type, agent.current_state] for agent in agents]
        # print(f'Total AOI to assign : ', len(merged_predicted_AOI))
        # update AOI master list
        # for pred_aoi in merged_predicted_AOI:
        #     if pred_aoi in GridWorld.aoi: 
        #         continue
        #     else:
        #         GridWorld.aoi.append(pred_aoi)
        assigned_AOIs = self.distribute_AOI_to_agents(current_locs, merged_predicted_AOI)
        for agent in agents:
            # agent.agent_stops = []
            # print(f'assign_AOI_to_agents : Before : for {agent.idx} after final len of agent.agent_stops : {len(agent.agent_stops)} ')
            for aoi in assigned_AOIs:
                if agent.agent_type == aoi[0]:
                    # if aoi[1] not in agent.agent_stops:
                    agent.agent_stops.append(aoi[1])
            agent.agent_stops_count = len(agent.agent_stops)
            # print(f'assign_AOI_to_agents : After : for {agent.idx} after final len of agent.agent_stops : {len(agent.agent_stops)} ')
            # plt.figure(figsize=(8, 8))
            # aoi_x = [aoi[0] for aoi in GridWorld.aoi]
            # aoi_y = [aoi[1] for aoi in GridWorld.aoi]
            # plt.scatter(aoi_x, aoi_y, color='grey', label='AOIs', alpha=0.5)
        
            # for agent in agents:
            #     current_x, current_y = agent.current_state
            #     stops_x = [stop[0] for stop in agent.agent_stops]
            #     stops_y = [stop[1] for stop in agent.agent_stops]
                
            #     plt.scatter(current_x, current_y, label=f'{agent.agent_type} Current Location', s=100)  # Plot current location
            #     plt.scatter(stops_x, stops_y, label=f'{agent.agent_type} Stops', alpha=0.6)  # Plot agent stops
            
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Y Coordinate')
            # plt.title('Agents Current Locations and Stops')
            # plt.legend()
            # plt.grid(True)
            # path_template = './results/{}/{}/c1/exp{}/aoi_at_{}.png'
            # path = path_template.format(pop_type, fold, counter, i)
            # plt.savefig(path)
            # plt.close()
        
        return

    def create_agent_stops_dictionary(self):
        dict_of_agent_stops = {}
        for agent in agent_configs:
            agent.agent_stops = list(set(filter(lambda x: x != (), agent.agent_stops)))
            dict_of_agent_stops[agent.idx] = list(agent.agent_stops)
        return dict_of_agent_stops

    def update_agent_stops_tsp_dp(self, tsp_dp_optimal_paths):
        for agent in agent_configs:
            agent.agent_stops = tsp_dp_optimal_paths[agent.idx] 
            agent.agent_stops_count = len(agent.agent_stops)
        return


    def update_model_weights(self, agents, md_agents):
        averaged_coef = []
        averaged_intercept = 0
        vals = [agent.agent_model_weights[0][0] for agent in md_agents ]
        averaged_coef = np.mean(vals, axis=0)
        averaged_intercept = np.mean([agent.agent_model_weights[0][1] for agent in md_agents ])
        weights = [averaged_coef, averaged_intercept]
        for agent in agents:
            agent.agent_model_weights = weights

    def get_max_len_of_path(self, agents):
        # max_len_of_path = 0
        max_len_of_path = max([len(agent.agent_path_to_stops) for agent in agents])
        return max_len_of_path

    def get_min_len_of_path(self, agents):
        # max_len_of_path = 0
        min_len_of_path = min([len(agent.agent_path_to_stops) for agent in agents])
        return min_len_of_path

    def update_agent_stops(self, result_clusters, agent_configs):
        for agent_id, stops in result_clusters.items():
            agent = agent_configs[agent_id]
            for stop in stops:
                if stop not in agent.agent_stops:
                    agent.agent_stops.append(stop)
            agent.agent_stops_count = len(agent.agent_stops)
            # print(f'agent id : {agent.idx} have stops : {agent.agent_stops_count}')
        return


    def generate_colors(self, c_count):
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, c_count)]
        return colors

    def plot_agent_traversed_paths(self, agent_configs, i, counter):
        c_count = len(agent_configs)
        colors = self.generate_colors(c_count)
        # agent_colors = {agent.idx: colors[idx % len(colors)] for idx, agent in enumerate(agent_configs)}
        plt.figure(figsize=(10, 6))
        for agent in agent_configs:
            color_index = agent.idx % len(colors)
            x_values = [point[1][0] for point in agent.agent_traversed_path]
            y_values = [point[1][1] for point in agent.agent_traversed_path]
            plt.plot(x_values, y_values, label=f"Agent {agent.idx}", color=colors[color_index])
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.ylim(grid_size, 0)  
        plt.title('Agent Traversed Paths')
        plt.legend() 
        plt.grid(True) 
        path_template = './results/{}/{}/c1/exp{}/paths_at_{}.pdf'
        path = path_template.format(pop_type, fold, counter, i)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(path)
        plt.close()
        return

    def recalculate_agent_path_to_stops(self, agents, agent_current_locations, i):
        for agent in agents: 
            current_index = -1
            for agent_current_location in agent_current_locations:
                if agent.idx == agent_current_location[0]:
                    current_index = agent_current_location[2]
                    break
            if current_index != 0:
                agent.agent_path_to_stops = agent.agent_path_to_stops[:current_index + 1]
            else:
                agent.agent_path_to_stops = [agent.current_state]
            if agent.agent_stops:
                last_stop = agent.agent_path_to_stops[-1]
                for stop in agent.agent_stops:
                    path = self.cal_astar_path(G, last_stop, stop)
                    if path:
                        agent.agent_path_to_stops.extend(path[1:])  # Avoid duplicates
                    last_stop = stop
            # print(f'agent id : {agent.idx} has agent_path_to_stops length  :  {len(agent.agent_path_to_stops)}')
            min_len_of_path = arch.get_min_len_of_path(agent_configs)
            max_len_of_path = arch.get_max_len_of_path(agent_configs)
            if i == min_len_of_path - 1   : 
                arch.balance_agent_stops(agent_configs, min_len_of_path, max_len_of_path)
        return

    
    def create_customers(self, agent_stops, predictions, agent_current_state, battery_percentage_remaining):
        customers = []
        for agent_stop in agent_stops:
            idx = agent_stop
            x = idx[0]
            y = idx[1]
            prob_score = predictions[idx]['prob_score']
            k = self.euclidean_distance_heuristic(idx, agent_current_state)
            demand = self.importance_metric(prob_score, k, battery_percentage_remaining, self.max_distance)
            predictions[idx]['demand'] = demand
            # print(f'idx : {idx}, x : {x}, y : {y}, prob_score : {prob_score}, demand : {demand}')
            customer = Customer(idx, x, y, prob_score, demand)
            customers.append(customer)
        return customers
    
    
    def calculate_initial_predictions(self, result_clusters):
        predictions = []
        for agent_id, stops in result_clusters.items():
            agent = agent_configs[agent_id]
            for stop in stops:
                loc = stop
                value_array = np.array([1])
                first_tuple = stops[0]
                k = self.euclidean_distance_heuristic(stop, first_tuple)
                if k == 0 : 
                    probability_scores = np.array([[0, 0]])
                else: 
                    probability_scores = np.array([[0, 1/k]]) 
                prediction = [loc, value_array, probability_scores]
                predictions.append(prediction)
        return predictions
                

    def cws_v1(self, coordinates, agent_configs, predictions):
        final_routes = {}
        for agent, coords in coordinates.items():
            for agent_config in agent_configs:
                if agent_config.idx == agent:
                    if agent_config.current_state == None :
                        agent_config.current_state = agent_config.agent_stops[0]
                    agent_config.customers = self.create_customers(agent_config.agent_stops, predictions, agent_config.current_state, agent_config.battery_percentage_remaining)
                    routes, total_cost = self.clarke_wright_savings(agent_config.customers, agent_config.current_state, predictions)
                    flat_list = [item for sublist in routes for item in sublist]
                    agent_config.trip_costs.append(total_cost)
            final_routes[agent] = flat_list
        # print(final_routes)    
        return final_routes

    def euclidean_distance(self, customer1, customer2):
        return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)
    
    def euclidean_distance_with_current_state(self, customer1, customer2):
        return math.sqrt((customer1.x - customer2[0])**2 + (customer1.y - customer2[1])**2)

    def max_dt(self):
        return math.sqrt((min_val - max_val)**2 + (min_val - max_val)**2)

    def get_demand_for_coordinates(self, customers, target_x, target_y):
        for customer in customers:
            if customer.x == target_x and customer.y == target_y:
                return customer.demand
        return None
    
    def get_customer_data_for_coordinates(self, customers, target_x, target_y):
        for customer in customers:
            if customer.x == target_x and customer.y == target_y:
                return customer.demand, customer.prob_score
        return None

    def update_all_demand_values(self, customers):
        for customer in customers:
            s_io = round(self.euclidean_distance(customer, customers[0]), 2)
            if s_io == 0:
                customer.demand = 0
                continue
            customer.demand = round((self.eta * customer.prob_score) / s_io, 2)
        return

    def calculate_demand_for_a_customer(self, coord, prob_score, current_state):
        s_io = round(self.euclidean_distance(coord, current_state), 2)
        if s_io == 0:
            demand = 0
        else: 
            demand = round((eta * prob_score) / s_io, 2)
        return demand

    def calculate_average_demand(self, customers):
        if not customers:
            return 0
        total_demand = sum(customer.demand for customer in customers)
        average_demand = total_demand / len(customers)
        return average_demand
    
    def normalize_distance(self, distance, max_distance):
        return distance / max_distance

    def importance_metric(self, prob_score, distance, battery_remaining, max_distance):
        norm_distance = self.normalize_distance(distance, max_distance)
        demand = 1 - norm_distance
        weight_prob = 0.8
        weight_demand = 0.1
        weight_battery = 0.1
        return (weight_prob * prob_score) + (weight_demand * demand) + (weight_battery * battery_remaining / 100)

    
    def euclidean_distance_p(self, p1, p2):
        return np.sum(np.abs(p1 - p2))
    
    def compute_distance_matrix(self, points):
        num_points = len(points)
        distance_matrix = np.zeros((num_points, num_points))
        
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    distance_matrix[i, j] = np.sqrt(np.sum((np.array(points[i]) - np.array(points[j])) ** 2))
        return distance_matrix
    
    
    def compute_savings_matrix(self, customers, current_state, predictions):
        points = [(customer.x, customer.y) for customer in customers]
        num_points = len(points)
        savings_matrix = np.zeros((num_points, num_points))
        probability_matrix = np.zeros((num_points, num_points))  
        nnh_matrix = np.zeros((num_points, num_points)) 
        
        distance_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    distance_matrix[i, j] = self.euclidean_distance_p(np.array(points[i]), np.array(points[j]))
        for i in range(1, num_points):
            i_loc = (customers[i].x, customers[i].y)
            i_ps = predictions[i_loc]['prob_score']
            for j in range(i + 1, num_points):
                j_loc = (customers[j].x, customers[j].y)
                j_ps = predictions[j_loc]['prob_score']
                combined_ps = [i_ps, j_ps]
                p_score = np.sum(combined_ps)
                probability_matrix[i, j] = p_score
                probability_matrix[j, i] = p_score
        # print(probability_matrix)
        updated_savings_matrix =  probability_matrix  
        savings_list = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                updated_savings_value = updated_savings_matrix[i, j]
                savings_list.append(((i, j), updated_savings_value))
        
        sorted_savings = sorted(savings_list, key=lambda x: x[1], reverse=True)
        return sorted_savings, nnh_matrix


    def find_routes(self, customers, current_state, sorted_savings, distance_matrix):
        num_points = len(customers)
        
        # Initialize routes: each customer starts in its own route
        routes = {i: [i] for i in range(num_points)}
        
        # Function to find the route containing a given point
        def find_route(point):
            for route in routes.values():
                if point in route:
                    return route
            return None
        
        # Function to get the nearest neighbor of a point
        def get_nearest_neighbor(point, distance_matrix):
            distances = distance_matrix[point, :]
            valid_distances = distances[distances > 0]
            if len(valid_distances) > 0:
                nearest_neighbor_index = np.argmin(valid_distances)
                return np.where(distances == valid_distances[nearest_neighbor_index])[0][0]
            else:
                return None
        
        # Merge routes based on savings
        for (i, j), saving in sorted_savings:
            route_i = find_route(i)
            route_j = find_route(j)
            
            if route_i != route_j:
                # Merge the routes and update the routes dictionary
                merged_route = route_i + route_j
                for point in merged_route:
                    routes[point] = merged_route

        # Deduplicate routes
        unique_routes = []
        for route in routes.values():
            if route not in unique_routes:
                unique_routes.append(route)
        
        # Convert routes from indices to coordinates
        routes_with_coordinates = []
        for route in unique_routes:
            coord_route = [ (customers[i].x, customers[i].y) for i in route ]
            routes_with_coordinates.append(coord_route)
        
        # Calculate the total cost of the final routes
        total_cost = 0
        for route in routes_with_coordinates:
            # Include the trip from and to the depot (current_state)
            total_cost += self.euclidean_distance_p(np.array(current_state), np.array(route[0]))
            for i in range(len(route) - 1):
                total_cost += self.euclidean_distance_p(np.array(route[i]), np.array(route[i+1]))
        
        return routes_with_coordinates, total_cost



    def clarke_wright_savings(self, customers, current_state, predictions):
        sorted_savings, nnh_matrix = self.compute_savings_matrix(customers, current_state, predictions)
        routes, total_cost = self.find_routes(customers, current_state, sorted_savings, nnh_matrix)
        # print(nnh_matrix)
        return routes, total_cost


    def balance_agent_stops(self, agent_configs, min_stops, max_stops):
        for agent in agent_configs:
            stops_length = len(agent.agent_path_to_stops)
            # print(f'stops_length = {stops_length} | min_stops = {min_stops} | max_stops = {max_stops}')
            if stops_length == min_stops:
                while len(agent.agent_path_to_stops) < max_stops:
                    agent.agent_path_to_stops.append(agent.current_state)
        return
    
    def balance_all_agent_stops(self, agent_configs, max_stops):
        for agent in agent_configs:
            stops_length = len(agent.agent_path_to_stops)
            # print(f'before stops_length = {stops_length} | max_stops = {max_stops}')
            while len(agent.agent_path_to_stops) < max_stops:
                agent.agent_path_to_stops.append(agent.current_state)
            # print(agent.agent_path_to_stops)    
            # print(f'after stops_length = {len(agent.agent_path_to_stops)} | max_stops = {max_stops}')
            
        return

class Customer:
    def __init__(self, idx, x, y, prob_score, demand):
        self.id = idx
        self.x = x
        self.y = y
        self.prob_score = prob_score
        self.demand = demand


class PredictionsInitializer:
    def __init__(self):
        self.dict_of_dicts = self.initialize_dict()

    def initialize_dict(self):
        dict_of_dicts = {}
        for x in range(max_val):
            for y in range(max_val):
                loc = (x, y)
                dist_to_origin = self.euclidean_distance(loc, (0, 0))
                prob_score = 1 / dist_to_origin if dist_to_origin != 0 else 0  # Avoid division by zero
                dict_of_dicts[loc] = {
                    'loc': loc,
                    'prob_value': 0,
                    'prob_score': prob_score,
                    'visit_status': 0,
                    'prediction_status' : 0,
                    'demand' : 0
                }
        return dict_of_dicts

    @staticmethod
    def euclidean_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


predictions_initializer = PredictionsInitializer()
predictions = predictions_initializer.dict_of_dicts

arch = ThreeTierArchitecture()
predict = PredictionCall()
G = arch.create_2d_network(grid_size, grid_size)

action_to_direction = _action_to_direction
num_agents = num_agents
grid_size = grid_size

for agent in agent_configs:
    agent.current_state = (0, 0)
    agent.agent_step_count = 0

result_clusters = GridWorld.result_clusters
arch.update_agent_stops(result_clusters, agent_configs)
agent_current_locations = arch.get_agents_current_locations(agent_configs)
arch.recalculate_agent_path_to_stops(agent_configs, agent_current_locations, 0)

# step 2 : Identification of MDs 
# md_agents = arch.identify_fixed_mds(agent_configs, num_agents)
md_agents = arch.identify_mds(agent_configs, num_agents)

sd_agents = [agent for agent in agent_configs if agent.agent_type_md == '']
for md_agent in md_agents:
    print(f"Agent {md_agent.idx} is now an MD with battery level {md_agent.battery_percentage_remaining}")

max_steps = GridWorld.grid_size * GridWorld.grid_size
h = 1
n_steps_interval = h * int(max_steps / GridWorld.grid_size)
max_len_of_path = 0
min_len_of_path = 0

max_len_of_path = arch.get_max_len_of_path(agent_configs)
min_len_of_path = arch.get_min_len_of_path(agent_configs)
# print("max_len_of_path : ", max_len_of_path)
# print("min_len_of_path : ", min_len_of_path)

hop = 0
term = 0
agents_traversed_paths = {}
agents_step_counts = {}
agents_survior_detections = {}
agents_aoi_identified = {}
agents_survior_detections_per_interval = []
agents_metrics = {}
total_surv_identified = 0
population = 0
traversed_aois = []
traversed_aois_per_interval = []
aoi_traversal = []
predictions_per_interval = []
prediction_count = 0
min_len = 0 
previous_i = None
previous_i_val = 0
prediction_times = []
times = 0
all_predicted_aoi = []

with open('./Testing/traversed_aoi.pkl', 'wb') as f:
    pickle.dump(traversed_aois, f)
    
while max_len_of_path > 0 : 
    for i in range(min_len, min_len_of_path):
        if previous_i is not None and previous_i == i:
            same_i_counter += 1
            if same_i_counter > 3:
                print(f"c1 Value of i ({i}) has remained the same for more than 3 times. Breaking the loop.")
                term = 1
                break
        else:
            same_i_counter = 0  # Reset counter if i changes

        previous_i = i
        actions = []
        battery_levels = []
        for agent in agent_configs:
            previous_state = ()
            path = agent.agent_path_to_stops
            population = 0
            if i < len(path): 
                if i < len(path) - 1 :
                    current_state = path[i]
                    next_state = path[i+1]
                    action = arch.identify_action(current_state, next_state)
                    agent.current_state = current_state
                    agent.next_state = next_state
                    agent.agent_step_count += 1
                    
                else:
                    current_state = path[i]
                    next_state = path[i]
                    action = arch.identify_action(current_state, next_state)
                    agent.current_state = current_state
                    agent.next_state = next_state
                
                if locations[agent.current_state]['visit_count'] == -1:
                    population = locations[agent.current_state]['population']
                    agent.survivor_identified_count += int(population)
                    locations[agent.current_state]['population'] = 0
                    locations[agent.current_state]['visit_count'] += 1
                    
                else:
                    locations[agent.current_state]['visit_count'] += 1
                    population = locations[agent.current_state]['population']
                    agent.survivor_identified_count += int(population)

                if agent.current_state not in agent.agent_path: 
                    agent.agent_path.append(agent.current_state)
                agent.current_state_index = path.index(path[i])
                if agent.current_state_index != i :          
                    current_state_indices = [index for index, value in enumerate(path) if value == path[i]]
                    agent.current_state_index = min(current_state_indices, key=lambda x: abs(x - i))

                actions.append(action)
            else:
                actions.append(8)

            if predictions[agent.current_state]['visit_status'] == 0:
                predictions[agent.current_state]['visit_status'] = 1
            
            if agent.current_state in agent.agent_stops: 
                agent.agent_stops.remove(agent.current_state)
            
            if agent.current_state in all_aoi and agent.current_state not in traversed_aois:
                traversed_aois.append(agent.current_state)
                agent.aoi_identified_count += 1
            if agent.survivor_identified_count <= tot_defined_population and agent.aoi_identified_count <= len(all_aoi):
                agent.agent_traversed_path.append([i, agent.current_state, agent.survivor_identified_count, agent.aoi_identified_count, agent.agent_step_count])
                agent.agent_current_interval_path.append([i, agent.current_state, agent.survivor_identified_count, agent.aoi_identified_count, agent.agent_step_count])
            # find total energy consumption and remaining battery percentage
            agent.distance = agent.latlon_mapping_for_distance_cal(agent.current_state, agent.next_state)
            agent.energy_consumption, battery_percentage_remaining = agent.total_energy_and_battery_percentage(agent.distance)
            agent.battery_percentage_remaining = battery_percentage_remaining 
            agent.agent_current_latidude, agent.agent_current_longitude = agent.getLatitudeLongitude(agent.current_state)
            agent.agent_next_latitude, agent.agent_next_longitude = agent.getLatitudeLongitude(agent.next_state)
            battery_levels.append(agent.battery_percentage_remaining)

        # print(actions)
        # print("battery_levels : ", battery_levels)
        is_less_than_20 = any(level < 20 for level in battery_levels)
        if is_less_than_20:
            print("Battery levels dropped beyond threshold. Terminating..c1 counter : ", counter)
            term = 1
            break
        
        if set(traversed_aois) == set(all_aoi):
            print("All the AOI are traversed.. Terminating exp c1 counter : ", counter)
            term = 1
            break

        if round(total_surv_identified/tot_defined_population,2) >= 0.85:
            print(f'85 % survivors detected. terminating experiement')
            term = 1
            break
        
        next_observation, reward, done = GridWorld.step(actions)

        if i == min_len_of_path - 1 or i % n_steps_interval == 0 and i != 0 and abs(previous_i_val - i) > n_steps_interval:
        # if i == min_len_of_path - 1 and i != 0 :
            
            with open('./Testing/traversed_aoi.pkl', 'wb') as f:
                pickle.dump(traversed_aois, f)
            traversed_aois_per_interval.append(len(traversed_aois))
            aoi_traversal.append([i, traversed_aois_per_interval[-1], traversed_aois_per_interval[-1] / len(all_aoi)])
            # print("battery_levels : ", battery_levels)
            
            hop += 1
            total_surv_identified = 0
            for agent in agent_configs:
                # print(f'agent {agent.agent_type} has identified {agent.survivor_identified_count} survivors')
                total_surv_identified += agent.survivor_identified_count
            spercentage = total_surv_identified/tot_defined_population
            if [i, total_surv_identified, spercentage] not in agents_survior_detections_per_interval: 
                agents_survior_detections_per_interval.append([i, total_surv_identified, spercentage])
            arch.plot_agent_traversed_paths(agent_configs, hop, counter)
            # md_agents = arch.identify_mds(agent_configs, num_agents)
            # for md_agent in md_agents:
                # print(f"Agent {md_agent.idx} is now an MD with battery level {md_agent.battery_percentage_remaining}")
            arch.assign_md_to_sd(agent_configs, md_agents)
            # start_time = time.time()
            arch.create_new_dataset_for_AOIprediction(agent_configs, md_agents)
            predict.train_test_newAOI(agent_configs, md_agents)
            predict.get_untraversed_neighbors(agent_configs, grid_size)
            arch.create_dataset_for_untraversedAOI(agent_configs, md_agents)
            start_time = time.time()
            merged_predicted_AOI, predictions, prediction_accuracies, std_errors = predict.predict_newAOI_logreg(md_agents, agent_configs, predictions)
            end_time = time.time()
            all_predicted_aoi.extend(merged_predicted_AOI)
            with open('./Testing/first_aoi.pkl', 'wb') as f:
                pickle.dump(all_predicted_aoi, f)
            predictions_per_interval.append(len(merged_predicted_AOI))
            prediction_count = len(merged_predicted_AOI)
            arch.update_model_weights(agent_configs, md_agents)
            arch.assign_AOI_to_agents(agent_configs, md_agents, merged_predicted_AOI)
            dict_of_agent_stops = arch.create_agent_stops_dictionary()
            cws_v1_optimal_paths = arch.cws_v1(dict_of_agent_stops, agent_configs, predictions)
            arch.update_agent_stops_tsp_dp(cws_v1_optimal_paths)
            agent_current_locations = arch.get_agents_current_locations(agent_configs)
            arch.recalculate_agent_path_to_stops(agent_configs, agent_current_locations, i)
            end_time = time.time()
            time_taken = end_time - start_time
            prediction_times.append(time_taken)
            min_len = i
            min_len_of_path = arch.get_min_len_of_path(agent_configs)
            h += 1
            if prediction_count == 0: 
                print("c1 Prediction counts are very less ", prediction_count) 
                min_len_of_path = arch.get_max_len_of_path(agent_configs)
                arch.balance_all_agent_stops(agent_configs, min_len_of_path)
            
            if min_len_of_path < i : 
                print("c1 i >= min_len_of_path ")
                term = 1
                break
            
            print(f'c1 i : {i} | Iteration = {h-1} | interval size = {abs(previous_i_val - i)} | preds = {prediction_count} | survivors_detected :  {round(total_surv_identified/tot_defined_population,2)} | psl_traversed = {round(traversed_aois_per_interval[-1]/len(all_aoi), 2)}')
            previous_i_val = i
        if term == 1 :
            break
    
    # Terminal condition 
    if term == 1: 
        total_surv_identified = 0
        all_master_accuracies = {}
        total_trip_costs = {}
        for agent in agent_configs:
            total_surv_identified += agent.survivor_identified_count
            agents_traversed_paths[agent.idx] = agent.agent_traversed_path
            total_trip_costs[agent.idx] = agent.trip_costs
            agents_step_counts[agent.idx] = agent.agent_step_count
            agents_survior_detections[agent.idx] = agent.survivor_identified_count
            agents_aoi_identified[agent.idx] = agent.aoi_identified_count
            if agent.agent_type_md.startswith('MD') :
                agents_metrics[agent.idx] = agent.agent_mse
                all_master_accuracies[agent.idx] = agent.prediction_accuracies
        break

traversed_aois_per_interval.append(len(traversed_aois))
spercentage = total_surv_identified/tot_defined_population
agents_survior_detections_per_interval.append([i, total_surv_identified, spercentage])
aoi_traversal.append([i, traversed_aois_per_interval[-1], traversed_aois_per_interval[-1] / len(all_aoi)])
arch.plot_agent_traversed_paths(agent_configs, hop, counter)

# ####### Plot 1 : Agent wise total steps ##########
path_template = './results/{}/{}/c1/exp{}/agents_step_counts.pkl'
path = path_template.format(pop_type, fold, counter)
import os 
directory = os.path.dirname(path)
if not os.path.exists(directory):
    os.makedirs(directory)
with open(path, 'wb') as f:
    pickle.dump(agents_step_counts, f)

agent_ids = list(agents_step_counts.keys())
steps = list(agents_step_counts.values())

plt.figure(figsize=(8, 6))
plt.bar(agent_ids, steps, color='skyblue')
plt.xlabel('Agent ID')
plt.ylabel('Steps Count')
plt.title('Steps Count per Agent')
plt.xticks(agent_ids)
path_template = './results/{}/{}/c1/exp{}/agents_step_counts.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path) 


# ##### Plot 2 : Agent_averages i wise #######
    
def aggregate_agent_data(agent_data):
    aggregated_data = {}
    max_length = max(len(agent.agent_traversed_path) for agent in agent_configs)
    for i in range(max_length):
        for agent in agent_configs:
            if i not in aggregated_data:
                aggregated_data[i] = [0, 0, 0, 0]
            aggregated_data[i][0] = agent.agent_traversed_path[i][0]
            aggregated_data[i][1] += agent.agent_traversed_path[i][2]
            aggregated_data[i][2] += agent.agent_traversed_path[i][3]
            aggregated_data[i][3] += agent.agent_traversed_path[i][4]
    result = []
    for i_value, values in aggregated_data.items():
        # print(f'{i_value} :  {values}')
        sum_survivor_identified_count = values[1]
        sum_aoi_identified_count = values[2]
        avg_agent_step_count = int(values[3] / len(agent_configs))
        result.append([i_value, sum_survivor_identified_count, sum_aoi_identified_count, avg_agent_step_count])
    return result

def plot_aggregated_data(aggregated_data):
    x_values = [data[0] for data in aggregated_data]
    y_values_survivor_identified_count = [data[1] for data in aggregated_data]
    y_values_aoi_identified_count = [data[2] for data in aggregated_data]
    y_values_agent_step_count = [data[3] for data in aggregated_data]

    plt.figure(figsize=(15, 5))

    # Plot for avg_survivor_identified_count
    plt.subplot(1, 3, 1)
    plt.plot(x_values, y_values_survivor_identified_count, marker='', linestyle='-')
    plt.xlabel('i_value')
    plt.ylabel('Average Survivor Identified Count')
    plt.title('Average Survivor Identified Count per Step')

    # Plot for avg_aoi_identified_count
    plt.subplot(1, 3, 2)
    plt.plot(x_values, y_values_aoi_identified_count, marker='', linestyle='-')
    plt.xlabel('i_value')
    plt.ylabel('Average AOI Identified Count')
    plt.title('Average AOI Identified Count per Step')

    # Plot for avg_agent_step_count
    plt.subplot(1, 3, 3)
    plt.plot(x_values, y_values_agent_step_count, marker='', linestyle='-')
    plt.xlabel('i_value')
    plt.ylabel('Average Agent Step Count')
    plt.title('Average Agent Step Count per Step')

    
    path_template = './results/{}/{}/c1/exp{}/agent_averages.pdf'
    path = path_template.format(pop_type, fold, counter)
    plt.tight_layout()
    plt.savefig(path)  
    plt.close()

agent_averages = aggregate_agent_data(agent_configs)
plot_aggregated_data(agent_averages)

path_template = './results/{}/{}/c1/exp{}/agent_averages.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(agent_averages, f)


# ####### Plot 3 : agents_survior_detections ######
path_template = './results/{}/{}/c1/exp{}/agents_survior_detections.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(agents_survior_detections, f)

agent_ids = list(agents_survior_detections.keys())
survivors = list(agents_survior_detections.values())

plt.figure(figsize=(8, 6))
plt.bar(agent_ids, survivors, color='skyblue')
plt.xlabel('Agent ID')
plt.ylabel('Survivors Count')
plt.title('Survivors Count per Agent')
plt.xticks(agent_ids)
path_template = './results/{}/{}/c1/exp{}/agents_survior_detections.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path) 
plt.close()

# ####### plot 4 : agents_metrics ######
path_template = './results/{}/{}/c1/exp{}/agents_metrics.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(agents_metrics, f)

# ###### plot 5 : traversed_aois ######
path_template = './results/{}/{}/c1/exp{}/traversed_aois.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(traversed_aois_per_interval, f)

values = [len(all_aoi), traversed_aois_per_interval[-1]]
labels = ['Total AOI', 'Traversed AOI']
plt.figure(figsize=(4, 3))
plt.bar(labels, values, color=['blue', 'green'])
plt.title('AOI Traversal')
plt.xlabel('')
plt.ylabel('AOI Count')
path_template = './results/{}/{}/c1/exp{}/traversed_aois_bar.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close()

x_values = range(1, len(traversed_aois_per_interval) + 1)
plt.figure(figsize=(8, 6))
plt.plot(x_values, traversed_aois_per_interval, marker='.', linestyle='-', color='b')
plt.title('Traversed AOIs per Interval')
plt.xlabel('Interval')
plt.ylabel('Number of AOIs Traversed')
path_template = './results/{}/{}/c1/exp{}/traversed_aois_line.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close()

# ###### Plot 6 : predictions_per_interval #######
path_template = './results/{}/{}/c1/exp{}/predictions_per_interval.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(predictions_per_interval, f)

x_values = range(1, len(predictions_per_interval) + 1)
plt.figure(figsize=(8, 6))
plt.plot(x_values, predictions_per_interval, marker='', linestyle='-', color='g')
plt.title('Predictions per Interval')
plt.xlabel('Interval')
plt.ylabel('Number of Predictions')
path_template = './results/{}/{}/c1/exp{}/predictions_per_interval.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close()

# ###### Plot 7 : prediction_accuracies ######
path_template = './results/{}/{}/c1/exp{}/prediction_accuracies.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(prediction_accuracies, f)
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(prediction_accuracies) + 1), prediction_accuracies, marker='', linestyle='-', color='g', label='Prediction Accuracy')
plt.xlabel('Intervals')
plt.ylabel('Prediction Accuracy')
plt.title('Prediction Accuracies per interval')
path_template = './results/{}/{}/c1/exp{}/prediction_accuracies.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close() 
    

# ##### Plot 8 : agents_survior_detections_per_interval ######
path_template = './results/{}/{}/c1/exp{}/agents_survior_detections_per_interval.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(agents_survior_detections_per_interval, f)

i_values = [data[0] for data in agents_survior_detections_per_interval]
spercentage_values = [data[2] for data in agents_survior_detections_per_interval]
plt.figure(figsize=(8, 6))
plt.plot(i_values, spercentage_values, marker='o', linestyle='-', color='b')
plt.title('Survivor Detections Percentage per Interval')
plt.xlabel('Interval (i)')
plt.ylabel('Survivor Detection Percentage (%)')
path_template = './results/{}/{}/c1/exp{}/agents_survior_detections_per_interval.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close() 

# ###### Plot 9 : prediction_times ######
path_template = './results/{}/{}/c1/exp{}/prediction_times.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(prediction_times, f)   

plt.figure(figsize=(8, 6))
plt.plot(prediction_times, marker='', linestyle='-', color='b')
plt.title('Prediction Times')
plt.xlabel('Index')
plt.ylabel('Time (units)')
path_template = './results/{}/{}/c1/exp{}/prediction_times.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close() 

# ##### Plot 10 : aoi_traversal Percentages ######
path_template = './results/{}/{}/c1/exp{}/aoi_traversal.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(aoi_traversal, f)   

i_values = [data[0] for data in aoi_traversal]
spercentage_values = [data[2] for data in aoi_traversal]
plt.figure(figsize=(8, 6))
plt.plot(i_values, spercentage_values, marker='', linestyle='-', color='b')
plt.title('AOI Traversal Percentage per Interval')
plt.xlabel('Interval (i)')
plt.ylabel('AOI Traversal Percentage (%)')
path_template = './results/{}/{}/c1/exp{}/aoi_traversal.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close() 

# ###### Plot 11 : agents_traversed_paths #######
path_template = './results/{}/{}/c1/exp{}/agents_traversed_paths.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(agents_traversed_paths, f) 
    
plt.figure(figsize=(10, 6))
for agent, path in agents_traversed_paths.items():
    values = [t[2] for t in path]
    plt.plot(values, label=agent)

plt.xlabel('Steps')
plt.ylabel('Survivors Counts')
plt.title('Cumulative Survivor Detections of All Agents')
plt.legend()
plt.grid(True)
path_template = './results/{}/{}/c1/exp{}/agents_traversed_paths.pdf'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close()

# ###### Plot 12 : #######
path_template = './results/{}/{}/c1/exp{}/std_errors.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(std_errors, f) 

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(std_errors) + 1), std_errors, marker='', linestyle='-', color='g', label='std_errors')
plt.xlabel('Intervals')
plt.ylabel('std_error')
plt.title('std_error per interval')
# plt.ylim(0.8, 0.95)  
plt.legend()
path_template = './results/{}/{}/c1/exp{}/std_errors.png'
path = path_template.format(pop_type, fold, counter)
plt.savefig(path)  
plt.close()

# ###### Plot 13 : #######
path_template = './results/{}/{}/c1/exp{}/all_master_accuracies.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(all_master_accuracies, f) 
    
# ###### Plot 14 : #######
path_template = './results/{}/{}/c1/exp{}/total_trip_costs.pkl'
path = path_template.format(pop_type, fold, counter)
with open(path, 'wb') as f:
    pickle.dump(total_trip_costs, f) 

def update_exp_details():
    path_template = './results/{}/{}/c1/exp{}/exp_details.pkl'
    path = path_template.format(pop_type, fold, counter)
    exp_details = {
        'grid_size' : grid_size,
        'fold' : fold,
        'folder_name' : all_path[0],
        'master_r' : all_path[1],
        'agents' : all_path[2],
        'pop_type' : all_path[3],
        'ratio' : all_path[4],
        'all_aoi' : all_aoi,
        'all_aoi_count' : len(all_aoi),
        'traversed_aois' : traversed_aois,
        'traversed_aois_count' : len(traversed_aois),
        'traversed_aois_percentage' : (len(traversed_aois) / len(all_aoi)) * 100,
        'tot_defined_population' : tot_defined_population,
        'counter' : counter,
        'path' : path,
        'total_surv_identified' : total_surv_identified,
        'surv_percentage' : (total_surv_identified / tot_defined_population) * 100
    }
    with open(path, 'wb') as f:
        pickle.dump(exp_details, f)  
    return

update_exp_details()
update_counter(counter)

file_path = "./Testing/first_aoi.pkl"

# Check if the file exists before attempting to delete it
if os.path.exists(file_path):
    os.remove(file_path)
print("==================================================================================")