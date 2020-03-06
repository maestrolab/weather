import platform
import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import interpn, RegularGridInterpolator, interp1d

from weather.boom import boom_runner
from weather.scraper.noaa import process, output_for_sBoom

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
month = '06'
day = '21'
hour = '00'
directory = '../../../matlab/'
filename = directory + year + month + day + '_' + hour + '.mat'
output_directory = '../../../data/noise/'

alt_ft = 50000

# Process weather data
data = process(filename)

lat_cities = [47.6062, 43.6150, 39.7392, 30.6280, 25.7617, ]
lon_cities = [-122.3321, -116.2023, -104.9903, -96.3344, -80.1918]

lat_all = []
lon_all = []
distance_all = [0]
distance_cities = [0]
for i in range(len(lat_cities)-1):
    j = i+1
    lon_path = np.linspace(lon_cities[i], lon_cities[j], 2)
    lat_path = lat_cities[i] + (lon_path-lon_cities[i])/(lon_cities[j]-lon_cities[i])*(lat_cities[j]-lat_cities[i])
    lon_all += list(lon_path)
    lat_all += list(lat_path)

for i in range(len(lat_all)-1):
    j = i+1
    distance_all.append(distance_all[-1] + haversine(lon_all[i], lat_all[i],
                                                     lon_all[j], lat_all[j]))
for i in range(len(lat_cities)-1):
    j = i+1
    distance_cities.append(distance_cities[-1] + haversine(lon_cities[i], lat_cities[i],
                                                     lon_cities[j], lat_cities[j]))
                                                     
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
for i in range(len(lon_all)):
    # Consider elevation and round up (because of sboom input) for altitude
    elevation = path_elevation[i]

    height_above_ground = np.around(path_height[i].tolist(), decimals=1)

    # Convert temperature from Kelvin to Farenheight
    temperature = (path_temperature[i] - 273.15) * 9/5. + 32
    weather = {}
    weather['wind'] = np.array([height_above_ground,
                                path_wind_x[i].tolist(),
                                path_wind_y[i].tolist()]).T

    weather['temperature'] = np.array([height_above_ground,
                                       temperature.tolist()]).T
    weather['humidity'] = np.array([height_above_ground,
                                    path_humidity[i].tolist()]).T

    for key in weather:
        weather[key] = weather[key].tolist()

    sBoom_data = [weather['temperature'], weather['wind'], weather['humidity']]
    altitude = alt_ft

    # Run sBoom
    noise = boom_runner(sBoom_data, altitude, elevation)
    print(noise)
    path_noise.append(noise)
f = open(output_directory + 'path_' + year + month + day + '_' + hour + '_'
         + str(alt_ft) + ".p", "wb")
pickle.dump(data, f)
plt.figure()
plt.plot(distance_all, path_noise, 'r')
plt.ylabel('Perceived level in dB')
plt.xlabel('Longitude')
plt.show()