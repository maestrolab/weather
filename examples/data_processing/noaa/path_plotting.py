import pickle
import platform
import random
import numpy as np
from matplotlib import cm
from scipy import interpolate
from numpy import linalg as LA
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import interp1d, RegularGridInterpolator, interp1d

from weather.scraper.noaa import process


# Haversine formula: function to compute great circle distance between point lat1
# and lon1 and arrays of points given by lons, lats
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

def scanner(x, y, spacing = 10):
    points = []
    f = interp1d(x, y)
    min = np.min(np.array(x))
    max = np.max(np.array(x)-spacing)
    for i in range(100000):
        x_sample = random.uniform(min, max)
        y_sample = f(x_sample) - f(x_sample + spacing)
        # print(x_sample, x_sample + spacing, f(x_sample), f(x_sample + spacing))
        points.append([x_sample, y_sample])
    points = np.array(points)
    return points

year = '2018'
month = '06'
day = '21'
hour = '00'
alt_ft = 50000
directory = '../../../data/noise/'
# filename = directory + year + month + day + '_' + hour + '_' + \
    # str(alt_ft) + ".p"
filename = directory + 'standard_' + \
    str(alt_ft) + ".p"
f = open(filename, "rb")
data = pickle.load(f)

# Process Nan numbers
array = np.ma.masked_invalid(np.array(data.noise).reshape(data.lon_grid.shape))

#get only the valid values
lon1 = data.lon_grid[~array.mask]
lat1 = data.lat_grid[~array.mask]
newarr = array[~array.mask]

noise = interpolate.griddata((lon1, lat1), newarr.ravel(),
                          (data.lon_grid, data.lat_grid),
                             method='cubic')

# Setting up path
# Seattle, Boise, Denver, College Station, and Miami
lat_cities = [47.6062, 43.6150, 39.7392, 30.6280, 25.7617, ]
lon_cities = [-122.3321, -116.2023, -104.9903, -96.3344, -80.1918]

lat_all = []
lon_all = []
distance_all = [0]
for i in range(len(lat_cities)-1):
    j = i+1
    lon_path = np.linspace(lon_cities[i], lon_cities[j], 50)
    lat_path = lat_cities[i] + (lon_path-lon_cities[i])/(lon_cities[i]-lon_cities[j])*(lat_cities[j]-lat_cities[i])
    lon_all += list(lon_path)
    lat_all += list(lat_path)

for i in range(len(lat_all)-1):
    j = i+1
    distance_all.append(distance_all[-1] + haversine(lon_all[i], lat_all[i],
                                                     lon_all[j], lat_all[j]))
path = np.array([lon_all, lat_all]).T

# Preapre interpolation functions
elevation = np.flip(np.transpose(data.elevation), 1)
noise = np.flip(np.transpose(noise), 1)
humidity = np.flip(np.transpose(data.humidity, (2, 1, 0)), 1)
height = np.flip(np.transpose(data.height, (2, 1, 0)), 1)
temperature = np.flip(np.transpose(data.temperature, (2, 1, 0)), 1)
wind_x = np.flip(np.transpose(data.wind_x, (2, 1, 0)), 1)
wind_y = np.flip(np.transpose(data.wind_y, (2, 1, 0)), 1)

lonlat = np.array([data.lon, data.lat[::-1]])
f_elevation = RegularGridInterpolator(lonlat, elevation)
f_humidity = RegularGridInterpolator(lonlat, humidity)
f_height = RegularGridInterpolator(lonlat, height)
f_temperature = RegularGridInterpolator(lonlat, temperature)
f_wind_x = RegularGridInterpolator(lonlat, wind_x)
f_wind_y = RegularGridInterpolator(lonlat, wind_y)
f_noise = RegularGridInterpolator(lonlat, noise)

# Interpolating
path_elevation = f_elevation(path)
path_humidity = f_humidity(path)
path_height = f_height(path)
path_temperature = f_temperature(path)
path_wind_x = f_wind_x(path)
path_wind_y = f_wind_y(path)
path_noise = f_noise(path)

# Contour plot
pressures = np.array([100000, 97500, 95000, 92500, 90000, 85000, 80000, 75000, 70000,
                       65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000,
                       20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 700, 500, 300,
                       200, 100])
altitude = 10.**5/2.5577*(1-(pressures/101325)**(1/5.2558)) / 0.3048
LON, HEIGHT = np.meshgrid(altitude, distance_all)

plt.figure(figsize=(12, 6))
plt.contourf(HEIGHT, LON, path_humidity, cmap=cm.Blues, levels=np.linspace(0, 100, 21), extend = 'max')
plt.xlabel('Distance (km)')
plt.ylabel('Sea-level altitude (ft)')
clb = plt.colorbar()
clb.set_label('Relative Humidity')
plt.fill_between(distance_all, np.zeros(len(path_elevation)), path_elevation,
                facecolor = ".5", edgecolor = 'k', lw=1)
plt.ylim([0, 50000])
# plt.yticks(range(31), ylabels)



plt.figure(figsize=(12, 6))
plt.plot(distance_all, path_noise, 'k')
plt.ylabel('Perceived level in dB')
plt.xlabel('Distance (km)')

# Important for Contour map
# gradient = np.gradient(path_noise, distance_all)

# plt.figure(figsize=(12, 6))
# plt.plot(distance_all, gradient, 'k')
# plt.ylabel('PLdB gradient')
# plt.xlabel('Distance (km)')
#
#
# plt.figure(figsize=(12, 6))
# speed_sound = 294.9 # (m/s) at 50000ft
# v = 1.6*speed_sound
# time = 1000*np.array(distance_all)/v/60 #minutes
# gradient = np.gradient(path_noise, time)
# plt.plot(time, gradient, 'k')
# plt.ylabel('Noise gradient (PldB/min)')
# plt.xlabel('Time (minutes)')

print('Distance for 0.5: ', haversine(0, 0, 0, 0.5))
print('Distance for 0.3: ', haversine(0, 0, 0, 0.3))
print('Distance for 0.25: ', haversine(0, 0, 0, 0.25))

# plt.figure()
max_values = []
spacings = np.linspace(5, 400, 40)
for spacing in spacings:
    points = scanner(distance_all, path_noise, spacing)
    points = points[points[:,0].argsort()]
    [x,y] = points.T
    print(np.max(y), np.mean(y), np.std(y))
    max_values.append(np.max(np.abs(y)))
#     plt.plot(x,y, label = '%i km' % spacing)
# plt.xlabel('Distance (km)')
# plt.ylabel('PldB Delta')
# plt.legend()

plt.figure()
plt.plot(spacings, max_values)
plt.xlabel('Distance between samples (km)')
plt.ylabel('Change in PLdB')

plt.figure()
speed_sound = 294.9 # (m/s) at 50000ft
v = 1.6*speed_sound
time = 1000*np.array(spacings)/v/60 #minutes
plt.plot(time, max_values, 'k', label='Correlation', lw=2)
plt.axvline(x=1000*haversine(0, 0, 0, 0.5)/v/60, color = 'b', ls = '--', label = 'Weather resolution')
plt.axvline(x=1000*30/v/60, color = 'r', ls = '--', label = 'LIDAR capability')
plt.xlabel('Time in between sample (min)')
plt.ylabel('Change in PLdB')
plt.xscale('log')
plt.legend()
plt.show()
