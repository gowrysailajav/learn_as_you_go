import numpy as np
import pandas as pd
import random
import sympy as sp
import mpmath
import math
import sys
import pickle

with open('./num_agents.pkl', 'rb') as f:
    num_agents = pickle.load(f)
    
with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)
    
with open('./Testing/locations.pkl', 'rb') as f:
    locations = pickle.load(f)
mpmath.mp.dps = 5
battery_capacity = 3300 # 3.3  # in amp-hours
voltage = 11.1  # in volts
m_drone = 3.6   # Mass of drone
m_battery = 2.7  # mass of battery
num_rotors = 4 # Quadcopter
power_watts = 100 
diameter = 0.895
pressure = 100726  # 50 meters above sea level
R = 287.058
temperature = 15 + 273.15  # 15 degrees in Kelvin
rho = pressure / (R*temperature)
g = 9.81
# power efficiency
eta = 0.7
drag_coefficient_drone = 1.49
drag_coefficient_battery = 1
drag_coefficient_package = 2.2
projected_area_drone = 0.224
projected_area_battery = 0.015
projected_area_package = 0.0929

# drone parameters
payload = 0.5
# wind's parameters
wind_speed = 0
wind_direction = 0
# Agent Configurations
num_agents = num_agents
agent_configs = []

message = "Hello, world!"
c = 3e8  # Speed of light in m/s
F = 2.4e9  # Frequency in Hz (e.g., 2.4 GHz)
eLoS = 20  # Loss in dB for LoS scenario
eNLoS = 30  # Loss in dB for NLoS scenario
Capacity = 100  # Channel capacity in bits per second
beta = 0.1  # Define beta as a constant
a = 0.5  # Define 'a' as a constant or adjust its value
tx = 20 #fixed transmission power
I = 1.0  # Define 'I' as a constant or adjust its value
sigma = 1.0  # Define 'sigma' as a constant or adjust its value

# Define the altitude range in meters
min_altitude_meters = 11
max_altitude_meters = 12
# Convert meters to feet
min_altitude_feet = min_altitude_meters * 3.28084
max_altitude_feet = max_altitude_meters * 3.28084


class Agent:
    def __init__(self, idx, agent_type, current_state, next_state, max_steps, agent_speed, payload = payload, wind_speed = wind_speed, wind_direction = wind_direction):
        self.idx = idx
        self.agent_type = agent_type
        self.current_state = current_state
        self.next_state = next_state
        self.max_steps = max_steps
        self.agent_speed = agent_speed
        self.payload = payload
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

class AgentConfiguration(Agent):
    def __init__(self, idx, agent_type, current_state, next_state, max_steps, agent_speed, start_state=(0, 0)):
        super().__init__(idx, agent_type, current_state, next_state, max_steps, agent_speed)
        self.agent_stops = [(0, 0)]
        self.agent_path_to_stops = []
        self.agent_stops_count = 0 
        self.agent_path_to_stops_count = 0
        self.start_state = start_state
        self.initial_battery_capacity = battery_capacity * voltage  
        # Convert the random altitude value from feet back to meters
        self.agent_height = random.uniform(min_altitude_feet, max_altitude_feet)/ 3.28084 
        self.agent_bandwidth = 100
        self.agent_current_latidude, self.agent_current_longitude = self.getLatitudeLongitude(self.current_state)
        self.agent_next_latitude, self.agent_next_longitude = self.getLatitudeLongitude(self.next_state)
        self.side_distance = self.haversine_distance(self.agent_current_latidude, self.agent_current_longitude, self.agent_next_latitude, self.agent_next_longitude)
        self.distance = self.side_distance
        self.hyp_distance = self.calculate_hypotenuse(self.agent_height, self.side_distance)
        self.time_to_reach = self.calTime_to_reach(self.distance, self.agent_speed)
        self.agent_energy_hovering = self.get_energy_hovering(self.time_to_reach, payload, wind_speed)
        # Calculate direction only if distance is not zero
        if self.distance > 0:
            self.direction = self.calculate_direction(self.current_state, self.next_state)
        else:
            self.direction = 0
        self.relative_wind_direction = self.get_relative_wind_direction(self.direction, wind_direction)
        self.agent_energy_moving = self.get_energy_movement(self.distance, payload,
                                                            self.agent_speed, wind_speed,
                                                            self.relative_wind_direction)
        # self.energy_consumption = self.calTotal_Energy_Consumption(self.agent_energy_hovering, self.agent_energy_moving)
        self.energy_consumption = 0
        # self.remaining_battery_capacity = self.cal_remaining_battery_capacity(self.initial_battery_capacity, self.energy_consumption)
        self.remaining_battery_capacity = 0
        # self.battery_percentage_remaining = self.cal_battery_percentage_remaining(self.remaining_battery_capacity, self.initial_battery_capacity)
        self.battery_percentage_remaining = 100 
        
        # self.agent_data_size = self.calculate_wireless_data_size(message)
        self.agent_data_size = 0
        # self.agent_transmission_time = self.calculate_transmission_time(self.agent_height, self.hyp_distance, self.agent_data_size, Capacity)
        self.agent_transmission_time = 0
        # self.agent_transmission_rate = self.calculate_transmission_rate(self.agent_height, self.hyp_distance, Capacity)
        self.agent_transmission_rate = 0
        # self.agent_error_probability = self.calculate_error_probability(self.agent_height, self.hyp_distance)
        self.agent_error_probability = 0
        # self.agent_total_delay = self.calculate_total_delay(self.agent_height, self.hyp_distance, self.agent_data_size, Capacity)
        self.agent_total_delay = 0
        if self.agent_type == 'BS' :
            self.locations = locations
        else:
            self.locations = {}
        self.agent_type_md = ''
        self.agent_traversed_path = []
        self.agent_current_interval_path = []
        self.agent_path = []
        self.sd_agents_paths = []
        self.agent_connected_md = ''
        self.agent_distance_to_md = 0
        self.agent_transmission_cost = 0
        self.connected_agents = []
        self.agent_df = pd.DataFrame()
        self.agent_untraversed_df = pd.DataFrame()
        self.agent_model_weights = []
        self.agent_untraversed_neighbors = []
        self.predicted_AOI = []
        self.agent_mse = [] 
        self.survivor_identified_count = 0 
        self.aoi_identified_count = 0
        self.agent_step_count = 0 
        self.customers = []
        self.current_state_index = 0
        self.trip_costs = []
        self.prediction_accuracies = []

    def latlon_mapping_for_distance_cal(self, current_state, next_state):
        self.agent_current_latidude, self.agent_current_longitude = self.getLatitudeLongitude(current_state)
        self.agent_next_latitude, self.agent_next_longitude = self.getLatitudeLongitude(next_state)
        self.side_distance = self.haversine_distance(self.agent_current_latidude, self.agent_current_longitude, self.agent_next_latitude, self.agent_next_longitude)
        # print(f'self.agent_current_latidude : {self.agent_current_latidude}, self.agent_current_longitude : {self.agent_current_longitude}, self.agent_next_latitude : {self.agent_next_latitude}, self.agent_next_longitude : {self.agent_next_longitude}, self.side_distance : {self.side_distance}')
        return self.side_distance

    def total_energy_and_battery_percentage(self, distance):
        self.time_to_reach = self.calTime_to_reach(distance, self.agent_speed)
        self.agent_energy_hovering = self.get_energy_hovering(self.time_to_reach, self.payload, self.wind_speed)
        if distance > 0:
            self.direction = self.calculate_direction(self.current_state, self.next_state)
        else:
            self.direction = 0
        self.relative_wind_direction = self.get_relative_wind_direction(self.direction, wind_direction)
        self.relative_wind_direction = self.get_relative_wind_direction(self.direction, self.wind_direction)
        self.agent_energy_moving = self.get_energy_movement(distance, self.payload,
                                                       self.agent_speed, self.wind_speed,
                                                       self.relative_wind_direction)
        energy_consumption = self.calTotal_Energy_Consumption(self.agent_energy_hovering, self.agent_energy_moving)
        # print("energy_consumption = ", energy_consumption)
        self.energy_consumption = self.energy_consumption + energy_consumption 
        self.remaining_battery_capacity = self.cal_remaining_battery_capacity(self.initial_battery_capacity, self.energy_consumption)
        battery_percentage_remaining = self.cal_battery_percentage_remaining(self.remaining_battery_capacity, self.initial_battery_capacity)
        return self.energy_consumption, battery_percentage_remaining
    

    def calculate_transmission_cost(self, message, distance):
        if distance > 0 :
            self.agent_data_size = self.calculate_wireless_data_size(message)
            self.agent_transmission_time = self.calculate_transmission_time(self.agent_height, self.agent_distance_to_md, self.agent_data_size, Capacity)
            self.agent_transmission_rate = self.calculate_transmission_rate(self.agent_height, self.agent_distance_to_md, Capacity)
            self.agent_error_probability = self.calculate_error_probability(self.agent_height, self.agent_distance_to_md)
            self.agent_total_delay = self.calculate_total_delay(self.agent_height, self.agent_distance_to_md, self.agent_data_size, Capacity)
            transmission_cost = self.agent_transmission_time * (1 + self.agent_error_probability) + self.agent_total_delay
            return transmission_cost
        else:
            return 0


    def cal_battery_percentage_remaining(self, remaining_battery_capacity, initial_battery_capacity):
        return (remaining_battery_capacity / initial_battery_capacity) * 100

    def cal_remaining_battery_capacity(self, initial_battery_capacity, energy_consumption):
        return initial_battery_capacity - energy_consumption

    def calTotal_Energy_Consumption(self, energy_hovering=0, energy_moving=0):
        return energy_hovering + energy_moving

    def calculate_direction(self, current_state, next_state):
        direction = int(np.rad2deg(float(np.arctan2(next_state[1] - current_state[1], next_state[0] - current_state[0]))))
        return direction

    def get_relative_wind_direction(self, direction, wind_direction):
        return abs(direction - wind_direction)

    def calTime_to_reach(self, distance, speed):
        return distance/speed 

    def getLatitudeLongitude(self, current_state):
        lat = locations[current_state]['latitude']
        lon = locations[current_state]['longitude']
        return lat,lon 

    def get_energy_hovering(self, time, payload_weight, wind_speed):
        m_package = payload_weight
        v_air = wind_speed
        # Drag force
        F_drag_drone = 0.5 * rho * (v_air**2) * drag_coefficient_drone * projected_area_drone
        F_drag_battery = 0.5 * rho * (v_air**2) * drag_coefficient_battery * projected_area_battery
        F_drag_package = 0.5 * rho * (v_air**2) * drag_coefficient_package * projected_area_package
        F_drag = F_drag_drone + F_drag_battery + F_drag_package
        # Thrust
        T = (m_drone + m_battery + m_package)*g + F_drag
        # Power min hover
        P_min_hover = (T**1.5) / (np.sqrt(0.5 * np.pi * num_rotors * (diameter**2) * rho))
        # expended power
        P = P_min_hover / eta
        # Energy consumed
        E = P * time
        return E
    
    def get_energy_movement(self, distance, payload_weight, drone_speed, wind_speed, relative_wind_direction):
        m_package = payload_weight
        v_north = self.agent_speed - wind_speed*np.cos(np.deg2rad(relative_wind_direction))
        v_east = - wind_speed*np.sin(np.deg2rad(relative_wind_direction))
        v_air = np.sqrt(v_north**2 + v_east**2)
        # Drag force
        F_drag_drone = 0.5 * rho * (v_air**2) * drag_coefficient_drone * projected_area_drone
        F_drag_battery = 0.5 * rho * (v_air**2) * drag_coefficient_battery * projected_area_battery
        F_drag_package = 0.5 * rho * (v_air**2) * drag_coefficient_package * projected_area_package
        F_drag = F_drag_drone + F_drag_battery + F_drag_package
        alpha = np.arctan(F_drag / ((m_drone + m_battery + m_package)*g))
        # Thrust
        T = (m_drone + m_battery + m_package)*g + F_drag
        tmp_a = 2*T
        tmp_b = np.pi * num_rotors * (diameter**2) * rho
        tmp_c = (self.agent_speed*sp.cos(alpha))**2
        tmp_d = self.agent_speed*sp.sin(alpha)
        tmp_e = tmp_a / tmp_b
        coeff = [1, (2*tmp_d), (tmp_c+tmp_d**2), 0, -tmp_e**2]
        sol = np.roots(coeff)
        induced_speed = float(max(sol[np.isreal(sol)]).real)
        # Power min to go forward
        P_min = T*(self.agent_speed*np.sin(alpha) + induced_speed)
        # expended power
        P = P_min / eta
        # energy efficiency of travel
        mu = P / self.agent_speed
        # Energy consumed
        E = mu * distance
        return E     
    
    def get_object_size(self, obj):
        seen_objects = set()
        def _get_size(obj):
            if id(obj) in seen_objects:
                # Avoid counting the same object multiple times
                return 0
            seen_objects.add(id(obj))
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(_get_size(v) for v in obj.values())
                size += sum(_get_size(k) for k in obj.keys())
            elif hasattr(obj, '__dict__'):
                size += _get_size(obj.__dict__)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum(_get_size(item) for item in obj)
            return size
        return _get_size(obj)

    def calculate_wireless_data_size(self, message):
        data_size = sys.getsizeof(message)
        return data_size

    def pathLoS(self, distance):
        wavelength = c / F
        return 20 * math.log10((4 * math.pi * distance) / wavelength) + eLoS

    def pathNLoS(self, distance):
        wavelength = c / F
        return 20 * math.log10((4 * math.pi * distance) / wavelength) + eNLoS

    def calculate_transmission_rate(self, alti, distance, Capacity):
        avg_path_loss = self.calculate_avg_path_loss(alti, distance)
        return Capacity / math.log(1 + self.calculate_SINR(alti, avg_path_loss))

    def calculate_transmission_time(self, alti, distance, data_size, Capacity):
        rate = self.calculate_transmission_rate(alti, distance, Capacity)
        return data_size / rate

    def calculate_error_probability(self, alti, distance):
        avg_path_loss = self.calculate_avg_path_loss(alti, distance)
        return 1 / (1 + math.exp(avg_path_loss / 10))

    def calculate_total_delay(self, alti, distance, data_size, Capacity):
        tx_time = self.calculate_transmission_time(alti, distance, data_size, Capacity)
        propagation_delay = distance / c
        return tx_time + propagation_delay

    def calculate_avg_path_loss(self, alti, distance):
        p_los = self.calculate_PLoS(alti, distance)
        return p_los * self.pathLoS(distance) + (1 - p_los) * self.pathNLoS(distance)

    def calculate_PLoS(self, alti, distance):
        angle = self.calculate_angle_db(alti, distance)
        return 1 / (1 + math.exp(-beta * (angle - a)))

    def calculate_SINR(self, alti, distance):
        return self.dBm2W(tx) * (10 ** -(self.calculate_avg_path_loss(alti, distance) / 10)) / (self.dBm2W(I) + self.dBm2W(sigma))

    def dBm2W(self, dBm):
        return 10 ** ((dBm - 30) / 10.)

    def calculate_angle_db(self, alti, distance):
        return np.arcsin(alti / distance)

    def radians_to_degrees(self, radians_list):
        degrees_list = []
        for radians in radians_list:
            degrees = radians * (180 / math.pi)
            degrees_list.append(degrees)
        return np.array(degrees_list, dtype=np.float32)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
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
    
    def calculate_hypotenuse(self, side1, side2):
        hypotenuse = math.sqrt(side1 ** 2 + side2 ** 2)
        return hypotenuse

    

# Initialization of agent configurations
for i in range(num_agents):
    agent_config = AgentConfiguration(
        idx=i+1,
        agent_type="SD" + str(i),
        current_state=(0, 0),
        next_state=(0, 1),
        # distance=0, 
        max_steps=100,
        agent_speed=1,
    )
    agent_configs.append(agent_config)

# print(agent_configs)
# for agent_config in agent_configs:
#     print("\n",agent_config.__dict__)