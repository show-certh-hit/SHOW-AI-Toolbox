import numpy as np
import pandas as pd

# Define parameters
median_value = 4.2 # km/h
std_value =0.95 # km/h
num_points = 20
num_iterations = 100

# Create an empty list to store the resulting DataFrames
df_list = []

# Iterate the code block and append the resulting DataFrame to the list
for i in range(num_iterations):
    # Generate speed profile
    v = median_value + std_value * np.random.randn(num_points, 1)
    v = np.maximum(v, median_value + std_value)
    max_value = np.max(v)
    v = np.vstack((v, np.maximum(max_value, median_value + std_value) + std_value * np.random.randn(2, 1)))
    v = np.vstack((v, np.zeros((3, 1))))
    
    # Perform dead reckoning
    # (Assuming a GPS data file is available, otherwise the code needs to be modified)
    positions = pd.read_csv('gps_data.csv')  
    m, n = positions.shape
    rand_row = np.random.randint(0, m)
    lat1 = positions.iloc[rand_row, 0]
    lon1 = positions.iloc[rand_row, 1]
    speed = v/3.6 # Define the speed in m/s
    dt = 1 # Define the time step for each iteration
    R = 6371e3 # Define the Earth's radius in meters
    w = 2 * np.pi / 86164 # Define the angular velocity of the Earth
    lat = np.zeros(speed.shape) # Define the latitude array
    lon = np.zeros(speed.shape) # Define the longitude array
    lat[0] = lat1 # Set the starting latitude
    lon[0] = lon1 # Set the starting longitude
    for i in range(1, len(speed)): # Calculate the next latitude and longitude
        lat[i] = lat[i-1] + speed[i-1] * dt / R * np.cos(w * dt) / np.cos(lat[i-1])
        lon[i] = lon[i-1] + speed[i-1] * dt / (R * np.cos(lat[i-1]))

    alinear1 = np.zeros_like(speed)
    aquad1 = np.zeros_like(speed)
    alinear2 = np.zeros_like(speed)
    aquad2 = np.zeros_like(speed)
    alinear3 = np.zeros_like(speed)
    aquad3 = np.zeros_like(speed)

    for j in range(20):
      # SCP
      alinear1[j] = 2.782 - 0.154 * speed[j]
      aquad1[j] = (1.745 - 0.09 * speed[j])**2
    
      # LTAP/OD
      alinear2[j] = 2.924 - 0.247 * speed[j]
      aquad2[j] = (1.791 - 0.099 * speed[j])**2
    
      # LTAP/LD
      alinear3[j] = 2.167 - 0.057 * speed[j]
      aquad3[j] = (1.489 - 0.025 * speed[j])**2
    
    df = pd.DataFrame({'median_lat': np.median(lat),
                       'mean_lat': np.mean(lat),
                       'min_lat': np.min(lat),
                       'max_lat': np.max(lat),
                       'var_lat': np.var(lat),
                       'median_lon': np.median(lon),
                       'mean_lon': np.mean(lon),
                       'min_lon': np.min(lon),
                       'max_lon': np.max(lon),
                       'var_lon': np.var(lon),
                       'median_v': np.median(v),
                       'mean_v': np.mean(v),
                       'min_v': np.min(v),
                       'max_v': np.max(v),
                       'var_v': np.var(v)}, index=[0])
    df_list.append(df)

# Concatenate the list of DataFrames into a single DataFrame
accident_data = pd.concat(df_list, ignore_index=True)


print (accident_data)
# export the DataFrame to a csv file
#accident_data.to_csv('accident_data.csv')

