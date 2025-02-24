import pickle
import subprocess
import time 

def run_sh_file(sh_file_path):
    try:
        result = subprocess.run(['bash', sh_file_path], check=True)
        print(f"Successfully ran {sh_file_path}")
        return result.returncode == 0  # Return True if the script ran successfully
    except subprocess.CalledProcessError as e:
        print(f"Error running {sh_file_path}: {e}")
        return False
    

experiments = [
    {
        "grid_size": 27,
        "total_population": 41,
        "aoi_quota": 0.98,
        "pop_type": "sparse",
        "image_path": "./3x3_1_post.jpg",
        "image_path_training": "./3x3_1_pre.jpg",
    },
    # {
    #     "grid_size": 27,
    #     "total_population": 105,
    #     "aoi_quota": 0.95,
    #     "pop_type": "semi_dense",
    #     "image_path": "./3x3_1_post.jpg",
    #     "image_path_training": "./3x3_1_pre.jpg",
    # },
    # {
    #     "grid_size": 27,
    #     "total_population": 525,
    #     "aoi_quota": 0.86,
    #     "pop_type": "dense",
    #     "image_path": "./3x3_1_post.jpg",
    #     "image_path_training": "./3x3_1_pre.jpg",
    # },
    # {
    #     "grid_size": 45,
    #     "total_population": 105,
    #     "aoi_quota": 0.98,
    #     "pop_type": "sparse",
    #     "image_path": "./5x5_1_post.jpg",
    #     "image_path_training": "./5x5_1_pre.jpg",
    # },
    # {
    #     "grid_size": 45,
    #     "total_population": 283,
    #     "aoi_quota": 0.95,
    #     "pop_type": "semi_dense",
    #     "image_path": "./5x5_1_post.jpg",
    #     "image_path_training": "./5x5_1_pre.jpg",
    # },
    # {
    #     "grid_size": 45,
    #     "total_population": 1413,
    #     "aoi_quota": 0.9,
    #     "pop_type": "dense",
    #     "image_path": "./5x5_1_post.jpg",
    #     "image_path_training": "./5x5_1_pre.jpg",
    # },
    # {
    #     "grid_size": 63,
    #     "total_population": 180,
    #     "aoi_quota": 0.98,
    #     "pop_type": "sparse",
    #     "image_path": "./7x7_1_post.jpg",
    #     "image_path_training": "./7x7_1_pre.jpg",
    # },
    # {
    #     "grid_size": 63,
    #     "total_population": 378,
    #     "aoi_quota": 0.95,
    #     "pop_type": "semi_dense",
    #     "image_path": "./7x7_1_post.jpg",
    #     "image_path_training": "./7x7_1_pre.jpg",
    # },
    # {
    #     "grid_size": 63,
    #     "total_population": 1890,
    #     "aoi_quota": 0.9,
    #     "pop_type": "dense",
    #     "image_path": "./7x7_1_post.jpg",
    #     "image_path_training": "./7x7_1_pre.jpg",
    # },
]

def run_experiment(config):
    grid_size = config["grid_size"]
    total_population = config["total_population"]
    aoi_quota = config["aoi_quota"]
    pop_type = config["pop_type"]
    image_path = config["image_path"]
    image_path_training = config["image_path_training"]
    
    values_list = [(2, 6)]
    # values_list = [(2, 4), (2, 6), (2, 8), (2, 10), (2, 12), (3, 6), (3, 9), (3, 12), (3, 15), (3, 8), (4, 8), (4, 12), (4, 16), (4, 20)]
    sh_file = 'full_experiment_run.sh'

    with open('./grid_size.pkl', 'wb') as f:
        pickle.dump(grid_size, f)
    
    with open('./total_population.pkl', 'wb') as f:
        pickle.dump(total_population, f)

    with open('./aoi_quota.pkl', 'wb') as f:
        pickle.dump(aoi_quota, f)

    with open('./pop_type.pkl', 'wb') as f:
        pickle.dump(pop_type, f)

    with open('./image_path.pkl', 'wb') as f:
        pickle.dump(image_path, f)
    
    with open('./image_path_training.pkl', 'wb') as f:
        pickle.dump(image_path_training, f)
    
    for i, (ratio, num_agents) in enumerate(values_list):
        print(f'i = {i} | ratio = {ratio} | num_agents = {num_agents}')
        with open('./ratio.pkl', 'wb') as f:
            pickle.dump(values_list[i][0], f)
        
        with open('./num_agents.pkl', 'wb') as f:
            pickle.dump(values_list[i][1], f)
        
        time.sleep(10)
        
        success = run_sh_file(sh_file)
        
        if not success:
            print("Terminating the process due to an error.")
            break

        print("Completed current iteration, moving to the next set of variables.\n")

    print("All sets of variables have been processed.")

# Iterate over each experiment configuration and run the operations
for experiment in experiments:
    run_experiment(experiment)
    time.sleep(10)




    





