import pickle

folder_name = 'ps_sum_check'

with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)

with open('./total_population.pkl', 'rb') as f:
    total_population = pickle.load(f)

with open('./aoi_quota.pkl', 'rb') as f:
    aoi_quota = pickle.load(f)

with open('./pop_type.pkl', 'rb') as f:
    pop_type = pickle.load(f)
  
with open('./image_path.pkl', 'rb') as f:
    image_path = pickle.load(f)

with open('./image_path_training.pkl', 'rb') as f:
    image_path_training = pickle.load(f)  
    
with open('./ratio.pkl', 'rb') as f:
    r = pickle.load(f)

with open('./num_agents.pkl', 'rb') as f:
    ag = pickle.load(f)

num_agents = ag
ratio = r
master_ratio = round(1/ratio, 3)
master_election = folder_name
master_r = f'1to{ratio}'
agents = f'{num_agents}_agents'
result_folder_path = f'./results/{pop_type}/{grid_size}x{grid_size}'


print(f'grid_size = {grid_size} ')
print(f'total_population = {total_population}')
print(f'aoi_quota = {aoi_quota}')
print(f'pop_type = {pop_type}')
print(f'num_agents = {num_agents}')
print(f'master_ratio = {master_ratio}')
print(f'result_folder_path = {result_folder_path}')


def generate_lawnmower_paths(num_agents, grid_size):
    rows, cols = grid_size, grid_size
    segment_height = rows // num_agents
    paths = {}
    max_length = 0
    for agent in range(1, num_agents + 1):
        path = []
        start_row = (agent - 1) * segment_height
        end_row = start_row + segment_height if agent != num_agents else rows
        for r in range(start_row, end_row):
            if r % 2 == 0:
                for c in range(cols):
                    path.append((r, c))
            else:
                for c in range(cols - 1, -1, -1):
                    path.append((r, c))
        paths[agent] = path
        path_length = len(path)
        max_length = max(max_length, path_length)
    return max_length
max_length = generate_lawnmower_paths(num_agents, grid_size)


with open('./Testing/x_axis_limit.pkl', 'wb') as f:
    pickle.dump(max_length, f)
    
with open('./master_ratio.pkl', 'wb') as f:
    pickle.dump(master_ratio, f)
    
with open('./result_folder_path.pkl', 'wb') as f:
    pickle.dump(result_folder_path, f)

with open('./num_agents.pkl', 'wb') as f:
    pickle.dump(num_agents, f)

all_path = [master_election, master_r, agents, pop_type, ratio]
    
with open('./all_path.pkl', 'wb') as f:
    pickle.dump(all_path, f)    
