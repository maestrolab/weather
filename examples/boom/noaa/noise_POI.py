import platform
import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import interpn, RegularGridInterpolator, interp1d

from weather.boom import  boom_runner_eq
from weather.scraper.noaa import process, output_for_sBoom
from weather.scraper.geographic import elevation_function

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

year = '2018'
month = '04'
day = '16'
hour = '12'


directory = '../../../matlab/'
filename = directory + year + month + day + '_' + hour + '.mat'
output_directory = '../../../data/noise/'
# eq_area_file = 'Mach1.583_Alpha0.173_HL5.dat'

alt_ft = 50000

# Process weather data
data = process(filename)

# lat_cities = [47.6062, 43.6150, 39.7392, 32.7555, 25.7617, ]
# lon_cities = [-122.3321, -116.2023, -104.9903, -97.3308, -80.1918]
lat_all = [26.05]
lon_all = [-80.906]

# Setting up path
path = np.array([lat_all,lon_all]).T

# Interpolate elevation
LON, LAT = np.meshgrid(data.lon, data.lat)
lonlat = np.array([LON.flatten(), LAT.flatten()]).T
lon = np.array(data.lon)
lat = np.array(data.lat)

# Preapre interpolation functions
elevation = np.flip(data.elevation, 0)
humidity = np.flip(np.transpose(data.humidity, (1, 2, 0)), 0)
height = np.flip(np.transpose(data.height, (1, 2, 0)), 0)
temperature = np.flip(np.transpose(data.temperature, (1, 2, 0)), 0)
wind_x = np.flip(np.transpose(data.wind_x, (1, 2, 0)), 0)
wind_y = np.flip(np.transpose(data.wind_y, (1, 2, 0)), 0)

f_elevation = RegularGridInterpolator((lat[::-1],lon), elevation)
f_humidity = RegularGridInterpolator((lat[::-1],lon), humidity)
f_height = RegularGridInterpolator((lat[::-1],lon), height)
f_temperature = RegularGridInterpolator((lat[::-1],lon), temperature)
f_wind_x = RegularGridInterpolator((lat[::-1],lon), wind_x)
f_wind_y = RegularGridInterpolator((lat[::-1],lon), wind_y)

# Interpolating
path_elevation = f_elevation(path)
path_humidity = f_humidity(path)
path_height = f_height(path)
path_temperature = f_temperature(path)
path_wind_x = f_wind_x(path)
path_wind_y = f_wind_y(path)


path_noise = []

# Consider elevation and round up (because of sboom input) for altitude
elevation_ft = elevation_function(lat_all, lon_all)['elev_feet'][0]
print(elevation_ft)
height_above_ground = np.around(path_height[0].tolist(), decimals=1)

# Convert temperature from Kelvin to Farenheight
temperature = (path_temperature[0] - 273.15) * 9/5. + 32
weather = {}
weather['wind'] = np.array([height_above_ground,
                            path_wind_x[0].tolist(),
                            path_wind_y[0].tolist()]).T

weather['temperature'] = np.array([height_above_ground,
                                   temperature.tolist()]).T
weather['humidity'] = np.array([height_above_ground,
                                path_humidity[0].tolist()]).T

for key in weather:
    weather[key] = weather[key].tolist()

sBoom_data = [weather['temperature'], weather['wind'], weather['humidity']]

# Run sBoom
try:
    noise = boom_runner_eq(sBoom_data, alt_ft, 0)# nearfield_file=eq_area_file)
except:
    # Remove highest wind point in case of failure. Usually the reason
    sBoom_data[1] = sBoom_data[1][:-1]
    try:
        noise = boom_runner_eq(sBoom_data, alt_ft, 0)# nearfield_file=eq_area_file)
    except(FileNotFoundError):
        noise = np.nan
print(noise)