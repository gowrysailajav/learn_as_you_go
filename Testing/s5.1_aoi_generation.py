import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from s2_1_agent_conf import agent_configs, num_agents
import pickle
import random

with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)

with open('./Testing/aoi_without_elevation_testing.pkl', 'rb') as f:
    initial_aoi = pickle.load(f)

# print("initial_aoi : ", initial_aoi)

initial_deployed_aoi = []

class AOI_Generation():
    def __init__(self) -> None:
        self.clusters = 20
        self.grid_size = grid_size

    def get_agents_current_locations(self, agent_configs):
        agent_current_locations = []
        for agent in agent_configs:
            agent_current_locations.append(agent.current_state)
        return agent_current_locations

    def remove_duplicates(self, input_list):
        unique_set = set(tuple(inner_list) for inner_list in input_list)
        unique_list = [list(inner_tuple) for inner_tuple in unique_set]
        return unique_list

    def convert_to_tuples(self, input_list):
        tuple_list = [tuple(inner_list) for inner_list in input_list]
        return tuple_list

    def generate_non_overlapping_clusters(self, k=10):
        grid_size = self.grid_size
        if k > (grid_size - 2) * (grid_size - 2) or grid_size < 3:
            print("Invalid input. Unable to generate clusters.")
            return None
        clusters = []
        taken_positions = set()
        for _ in range(k):
            while True:
                start_row = np.random.randint(0, grid_size - 2)
                start_col = np.random.randint(0, grid_size - 2)
                if all((start_row + i, start_col + j) not in taken_positions for i in range(3) for j in range(3)):
                    break
            taken_positions.update((start_row + i, start_col + j) for i in range(3) for j in range(3))
            cluster_coordinates = []
            for i in range(3):
                for j in range(3):
                    cluster_coordinates.append([start_row + i, start_col + j])
            clusters.append(cluster_coordinates)
        return clusters


    def cluster_and_distribute(self, aoi, agent_current_locations):
        aoi_np = np.array(aoi)
        num_clusters = num_agents
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(aoi_np)
        cluster_labels = kmeans.labels_
        clusters = {i: [] for i in range(num_clusters)}
        for i, aoi_location in enumerate(aoi_np):
            cluster_id = cluster_labels[i]
            clusters[cluster_id].append(aoi_location)
        # output_tuples = {k: [tuple(item) for item in v] for k, v in clusters.items()}
        output_tuples = {k: [tuple(item) for item in v[:1]] for k, v in clusters.items()}
        output_tuples_all = [tuple(item) for sublist in clusters.values() for item in sublist[:5]]
        return output_tuples, output_tuples_all
    
    def populate_tuples(self, result_clusters):
        tuples_list = []
        for cluster_id, tuples in result_clusters.items():
            tuples_list.extend(tuples)
        return tuples_list
    
    def generate_random_clusters(self, grid_size, k):
        result_clusters = {}
        for i in range(k):
            random_tuple = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            result_clusters[i] = [random_tuple]
        return result_clusters

aoi_gen = AOI_Generation()
agent_current_locations = aoi_gen.get_agents_current_locations(agent_configs)
result_clusters, first_aoi = aoi_gen.cluster_and_distribute(initial_aoi, agent_current_locations)

result_clusters = aoi_gen.generate_random_clusters(grid_size, num_agents)
# print("result_clusters : ", result_clusters)
with open('./Testing/result_clusters.pkl', 'wb') as f:
    pickle.dump(result_clusters, f)

# with open('./Testing/first_aoi.pkl', 'wb') as f:
#     pickle.dump(first_aoi, f)