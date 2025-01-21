# Population distriibution based on house count with color coding

import cv2
import numpy as np
import pandas as pd
import pickle
import requests
import math
import os

def delete_pkl_files(file_path1, file_path2):
    # Delete the first file
    if os.path.exists(file_path1):
        try:
            os.remove(file_path1)
            print(f"Deleted {file_path1}")
        except Exception as e:
            print(f"Error deleting {file_path1}: {e}")
    else:
        print(f"{file_path1} does not exist.")
    
    # Delete the second file
    if os.path.exists(file_path2):
        try:
            os.remove(file_path2)
            print(f"Deleted {file_path2}")
        except Exception as e:
            print(f"Error deleting {file_path2}: {e}")
    else:
        print(f"{file_path2} does not exist.")
        
path_to_file1 = "./Training/df_aoi.pkl"
path_to_file2 = "./Training/df_not_aoi.pkl"

delete_pkl_files(path_to_file1, path_to_file2)

    
    
with open('./total_population.pkl', 'rb') as f:
    total_population = pickle.load(f)
print("total_population : ", total_population)

with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)
print("grid_size : ", grid_size)

with open('./aoi_quota.pkl', 'rb') as f:
    aoi_quota = pickle.load(f)
print("aoi_quota : ", aoi_quota)

with open('./Testing/cc_dataframe.pkl', 'rb') as f:
    cc_dataframe = pickle.load(f)

# print(cc_dataframe.columns)

with open('./image_path.pkl', 'rb') as f:
    image_path = pickle.load(f)
    
cell_area_global = 0
min_val = 0
max_val = grid_size
initial_latitude = 21.1938
initial_longitude = 81.3509
cell_size_mts = 40
R = 6378137

def calculate_grid_area(cell_size_mts, grid_size):
    cell_area = cell_size_mts * cell_size_mts
    total_cells = grid_size * grid_size
    grid_area = cell_area * total_cells
    return grid_area
grid_area = calculate_grid_area(cell_size_mts, grid_size)
print(f"The total grid area is {grid_area} square meters.")

def new_latitude(latitude, distance):
    return latitude + (distance / R) * (180 / math.pi)

def new_longitude(latitude, longitude, distance):
    return longitude + (distance / (R * math.cos(math.pi * latitude / 180))) * (180 / math.pi)


def calculate_lat_lon_grid(initial_lat, initial_lon, cell_size_mts, grid_size):
    lat_lon_grid = np.empty((grid_size, grid_size, 2))
    
    # Calculate the physical dimensions of the grid
    total_distance = cell_size_mts * (grid_size - 1)
    
    # Adjust latitude and longitude calculations to ensure coverage
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the lat/lon of each cell
            lat = new_latitude(initial_lat, i * cell_size_mts)
            lon = new_longitude(initial_lat, initial_lon, j * cell_size_mts)
            lat_lon_grid[i, j] = (lat, lon)
    
    # Adjust the outermost coordinates to ensure exact coverage
    lat_lon_grid[-1, :, 0] = new_latitude(initial_lat, total_distance)
    lat_lon_grid[:, -1, 1] = new_longitude(initial_lat, initial_lon, total_distance)
    
    return lat_lon_grid

lat_lon_grid = calculate_lat_lon_grid(initial_latitude, initial_longitude, cell_size_mts, grid_size)
output_file = './Testing/c2sparse_lat_lon_testing.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(lat_lon_grid, f)

with open('./Testing/c2sparse_lat_lon_testing.pkl', 'rb') as f:
    c2sparse_lat_lon = pickle.load(f)
    


def append_to_pickle(existing_pickle_file, new_data):
    try:
        # Step 1: Load the existing data from the pickle file
        with open(existing_pickle_file, 'rb') as f:
            existing_data = pickle.load(f)
        
        # Step 2: Append the new data
        updated_data = pd.concat([existing_data, new_data])

        # Step 3: Save the updated data back to the pickle file
        with open(existing_pickle_file, 'wb') as f:
            pickle.dump(updated_data, f)
    
    except FileNotFoundError:
        # If the file does not exist, create a new one
        with open(existing_pickle_file, 'wb') as f:
            pickle.dump(new_data, f)
    except Exception as e:
        print(f"An error occurred: {e}")


def find_neighbors_updated(x, y, min_val, max_val):
    zero_hop = [(x, y)]
    one_hop = []
    two_hop = []
    three_hop = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # skip the given cell itself
            nx = x + dx
            ny = y + dy
            if nx < min_val or nx >= max_val or ny < min_val or ny >= max_val:
                continue  # skip out-of-bounds cells
            one_hop.append((nx, ny))

            for dx2 in [-1, 0, 1]:
                for dy2 in [-1, 0, 1]:
                    if dx2 == 0 and dy2 == 0:
                        continue  # skip the given cell itself
                    nx2 = nx + dx2
                    ny2 = ny + dy2
                    if nx2 < min_val or nx2 >= max_val or ny2 < min_val or ny2 >= max_val:
                        continue  # skip out-of-bounds cells
                    two_hop.append((nx2, ny2))

                    for dx3 in [-1, 0, 1]:
                        for dy3 in [-1, 0, 1]:
                            if dx3 == 0 and dy3 == 0:
                                continue  # skip the given cell itself
                            nx3 = nx2 + dx3
                            ny3 = ny2 + dy3
                            if nx3 < min_val or nx3 >= max_val or ny3 < min_val or ny3 >= max_val:
                                continue  # skip out-of-bounds cells
                            three_hop.append((nx3, ny3))

    return (zero_hop, one_hop, two_hop, three_hop)


def hopData(x,y):
    zero_hop, one_hop, two_hop, three_hop = find_neighbors_updated(x, y, min_val, max_val)
    zero_hop = list(set(zero_hop))
    one_hop = list(set(one_hop))
    two_hop = list(set(two_hop)-set(one_hop)-set(zero_hop))
    three_hop = list(set(three_hop)-set(two_hop))
    all_hops = [one_hop, two_hop, three_hop]
    return all_hops


def get_pixel_counts(df, location):
    location_data = df[df['location'] == location]
    if location_data.empty:
        return None
    red_pixels = location_data['red_count'].values[0]
    blue_pixels = location_data['blue_count'].values[0]
    yellow_pixels = location_data['yellow_count'].values[0]
    green_pixels = location_data['green_count'].values[0]
    image_pixels = red_pixels + blue_pixels + yellow_pixels + green_pixels
    pixel_counts = {
        'red_pixels': red_pixels,
        'blue_pixels': blue_pixels,
        'yellow_pixels': yellow_pixels,
        'green_pixels': green_pixels,
        'image_pixels': image_pixels
    }
    return pixel_counts


def detect_color_regions(image, color_ranges):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_data = {}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
        pixel_sum = np.sum(mask)
        color_data[f'{color}_count'] = count
        color_data[f'{color}_pixels'] = pixel_sum
    return color_data

def divide_image_into_cells(grid_size, color_ranges):
    reference_latitude = initial_latitude
    reference_longitude = initial_longitude
    latitude_step = cell_size_mts / 111_000 
    longitude_step = cell_size_mts / (111_000 * np.cos(np.radians(reference_latitude)))
    
    cell_data = []
    for i in range(grid_size):
        for j in range(grid_size):
            location = (i, j)
            color_data = get_pixel_counts(cc_dataframe, location)
            ele = 0
            cell_data.append({
                'Location': (i, j),
                'lat_long': tuple(c2sparse_lat_lon[i,j]),
                'elevation': ele,
                'one_hopAOICount': 0,
                'one_hopSCount': 0,
                'one_hop_locs' : [],
                'one_hop_fraction': 0,
                'two_hopAOICount': 0,
                'two_hopSCount': 0,
                'two_hop_locs' : [],
                'two_hop_fraction': 0,
                'three_hopAOICount': 0,
                'three_hopSCount': 0,
                'three_hop_locs' : [],
                'three_hop_fraction': 0,
                **color_data
            })
    df = pd.DataFrame(cell_data)
    return df

def update_s_counts_to_df(df, initial_aoi):
    for index, row in df.iterrows():
        x, y = row['Location']
        all_hops = hopData(x, y)  # Assuming hopData is defined
        for i, given_locations in enumerate(all_hops):
            total_aoi, total_population = 0, 0
            population_values = distributed_data[distributed_data['Location'].isin(given_locations)]['population'].tolist()
            total_population = sum(population_values)
            list1 = initial_aoi
            list2 = given_locations
            aoi_values = set(list1) & set(list2)
            # print("Common tuples:", aoi_values)
            total_aoi = len(aoi_values)
            # print("Common tuples len:", total_aoi)
            if i == 0:
                df.at[index, 'one_hopSCount'] = total_population
                df.at[index, 'one_hopAOICount'] = total_aoi
                df.at[index, 'one_hop_locs'] = given_locations
            elif i == 1:
                df.at[index, 'two_hopSCount'] = total_population
                df.at[index, 'two_hopAOICount'] = total_aoi
                df.at[index, 'two_hop_locs'] = given_locations
            elif i == 2:
                df.at[index, 'three_hopSCount'] = total_population
                df.at[index, 'three_hopAOICount'] = total_aoi
                df.at[index, 'three_hop_locs'] = given_locations
    return df


def get_elevation(latitude, longitude):
    baseurl = "https://api.open-elevation.com/api/v1/lookup"
    params = {
        "locations": f"{latitude},{longitude}",
    }
    try:
        response = requests.get(baseurl, params=params)
        data = response.json()
        elevation = data["results"][0]["elevation"]
        return elevation
    except Exception as e:
        print("Error retrieving elevation data : ", e)
        return None


def distribute_population(total_population, cell_data):
    cell_data['total_weight'] = (
        0.6 * (cell_data['green_pixels']/cell_data['image_pixels']) +
        0.1 * (cell_data['yellow_pixels']/cell_data['image_pixels']) +
        0.3 * (cell_data['blue_pixels']/cell_data['image_pixels']) +
        0.0 * (cell_data['red_pixels']/cell_data['image_pixels']) 
    ) * (cell_data['green_pixels'] + cell_data['blue_pixels'] + cell_data['yellow_pixels'] + cell_data['red_pixels'] ) 
    cell_data['total_weight'].fillna(0, inplace=True)
    # print(cell_data[cell_data['total_weight'].isna()])
    cell_data['probability'] = cell_data['total_weight'] / cell_data['total_weight'].sum()
    cell_data['population'] = (total_population * cell_data['probability']).astype(int)
    remaining_population = total_population - cell_data['population'].sum()
    if remaining_population > 0:
        cell_data = cell_data.sort_values(by='probability', ascending=False)
        for i in range(remaining_population):
            cell_data.iloc[i % len(cell_data), cell_data.columns.get_loc('population')] += 1
    return cell_data


def fetch_locations_with_population_gt_k(distributed_data, k):
    filtered_tuples = distributed_data[distributed_data['population'] >= k]
    locations_gt_k = list(filtered_tuples['Location'])
    return locations_gt_k

# Define color ranges
color_ranges = {
    'red': ([0, 100, 100], [10, 255, 255]),  
    'blue': ([100, 50, 50], [140, 255, 255]),
    'yellow': ([28, 150, 150], [35, 255, 255]),
    'green': ([30, 70, 150], [60, 255, 255]) 
}

image_path = image_path

# Divide the image into cells and store data in a DataFrame
cell_data = divide_image_into_cells(grid_size, color_ranges)
distributed_data = distribute_population(total_population, cell_data)

sorted_population = distributed_data['population'].sort_values()
index_80th_percentile = int(len(sorted_population) * aoi_quota)
k = sorted_population.iloc[index_80th_percentile]
print("k : ", k)

initial_aoi = fetch_locations_with_population_gt_k(distributed_data, k)
# print(initial_aoi)
print("Initital aoi length : ",len(initial_aoi))
distributed_data['is_aoi'] = [1 if location in initial_aoi else 0 for location in distributed_data['Location']]
distributed_data1 = update_s_counts_to_df(distributed_data, initial_aoi)
print(distributed_data1)
print(sum(distributed_data1['population']))
max_pop_deployed = max(distributed_data1['population'])

with open('./Testing/df_without_elevation_testing.pkl', 'wb') as f:
    pickle.dump(distributed_data1, f)
with open('./Testing/aoi_without_elevation_testing.pkl', 'wb') as f:
    pickle.dump(initial_aoi, f)

    
df_aoi = distributed_data1[distributed_data1['is_aoi'] == 1]
df_not_aoi = distributed_data1[distributed_data1['is_aoi'] == 0]

with open('./Training/df_aoi.pkl', 'wb') as f:
    pickle.dump(df_aoi, f)
    
with open('./Training/df_full.pkl', 'wb') as f:
    pickle.dump(distributed_data1, f)
    
with open('./Training/df_not_aoi.pkl', 'wb') as f:
    pickle.dump(df_not_aoi, f)

append_to_pickle('./Training/df_aoi.pkl', df_aoi)
append_to_pickle('./Training/df_not_aoi.pkl', df_not_aoi)

with open('./Testing/max_pop_deployed.pkl', 'wb') as f:
    pickle.dump(max_pop_deployed, f)