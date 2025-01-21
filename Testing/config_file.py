import pickle

case = 10
var1 = 'case10'
var2 = ['location', 'latitude', 'longitude', 'probability', 'is_aoi', 'elevation', 'population']  # s6
var3 = ['Location', 'lat_long', 'probability', 'is_aoi', 'elevation', 'population'] # s6_1 
var4 = './Testing/df_without_elevation_testing.pkl'
var5 = './Testing/aoi_without_elevation_testing.pkl'
        
result = [var1, var2, var3, var4, var5]

# print(result) 

with open('./Testing/result.pkl', 'wb') as f:
    pickle.dump(result, f)
    