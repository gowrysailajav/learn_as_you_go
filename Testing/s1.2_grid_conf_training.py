import numpy as np
import pandas as pd
import pickle

with open('./Testing/result.pkl', 'rb') as f:
    result = pickle.load(f)   

with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)

with open(result[3], 'rb') as f:
    df = pickle.load(f)

selected_columns = ['Location', 'lat_long', 'elevation', 'one_hopAOICount', 'one_hopSCount', 'one_hop_fraction','two_hopAOICount', 'two_hopSCount', 'two_hop_fraction', 'three_hopAOICount', 'three_hopSCount', 'three_hop_fraction', 'probability', 'population', 'is_aoi']
df[selected_columns].to_csv('./Testing/dataset_output5_c2sparse2_testing.csv', index=False)
df = pd.read_csv('./Testing/dataset_output5_c2sparse2_testing.csv')
# print(df)