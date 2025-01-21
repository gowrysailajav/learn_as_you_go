import pandas as pd
import numpy as np
import pickle
import ast

with open('./Testing/result.pkl', 'rb') as f:
    result = pickle.load(f)   

with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)

with open(result[4], 'rb') as f:
    aoi = pickle.load(f)

csv_file_path = "./Testing/dataset_output5_c2sparse2_testing.csv"
df = pd.read_csv(csv_file_path)
locations = {}

class CellConfiguration:
    def __init__(self):
        self.size = grid_size
        # BS positioning
        self.bs_position = (0, 0)
        # locations = self.cell_config(df)


    def cell_config(self, df):
        locations = {}
        inner_keys = ['location', 'latitude', 'longitude', 'elevation', 'one_hopAOICount', 'one_hopSCount', 'one_hop_fraction','two_hopAOICount', 'two_hopSCount', 'two_hop_fraction', 'three_hopAOICount', 'three_hopSCount', 'three_hop_fraction', 'probability', 'population', 'is_aoi', 'visit_status', 'visit_count', 'reward', 'visited_by']
        for i in range(len(df)):
            row = df.iloc[i]
            loc = ast.literal_eval(row['Location'])
            inner_dict = {}
            for k in inner_keys:
                inner_dict[k] = row[k] if k in row else 0
            locations[loc] = inner_dict
            self.update_cell_config(locations, row)
        return locations


    def update_cell_config(self, locations, row):
        loc = ast.literal_eval(row['Location'])
        loc_tuple = ast.literal_eval(row['Location'])
        tuple_representation = ast.literal_eval(row['lat_long'])
        locations[loc]['location'] = loc_tuple
        locations[loc]['latitude'] = tuple_representation[0]
        locations[loc]['longitude'] = tuple_representation[1]
        locations[loc]['elevation'] = row['elevation']
        locations[loc]['one_hopAOICount'] = row['one_hopAOICount']
        locations[loc]['one_hopSCount'] = row['one_hopSCount']
        locations[loc]['one_hop_fraction'] = row['one_hop_fraction']
        locations[loc]['two_hopAOICount'] = row['two_hopAOICount']
        locations[loc]['two_hopSCount'] = row['two_hopSCount']
        locations[loc]['two_hop_fraction'] = row['two_hop_fraction']
        locations[loc]['three_hopAOICount'] = row['three_hopAOICount']
        locations[loc]['three_hopSCount'] = row['three_hopSCount']
        locations[loc]['three_hop_fraction'] = row['three_hop_fraction']
        locations[loc]['probability'] = row['probability']
        locations[loc]['population'] = row['population']
        locations[loc]['is_aoi'] = row['is_aoi']
        locations[loc]['visit_status'] = 0
        locations[loc]['visit_count'] = -1
        locations[loc]['visited_by'] = []


cell = CellConfiguration()
locations = cell.cell_config(df)
total_population = sum(locations[loc]['population'] for loc in locations)
print("Total Survivors Deployed:", total_population)

with open('./Testing/locations.pkl', 'wb') as f:
    pickle.dump(locations, f)

   