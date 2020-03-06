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
day = '21'
alt_ft = 50000
directory = '../../../data/noise/'

# Determining path
# Setting up path
# Seattle, Boise, Denver, College Station, and Miami
lat_cities = [47.6062, 43.6150, 39.7392, 30.6280, 25.7617, ]
lon_cities = [-122.3321, -116.2023, -104.9903, -96.3344, -80.1918]

lat_all = []
lon_all = []
distance_all = [0]
distance_cities = [0]
for i in range(len(lat_cities)-1):
    j = i+1
    lon_path = np.linspace(lon_cities[i], lon_cities[j], 50)
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
print(distance_cities)
path = np.array([lat_all, lon_all]).T
# Time
speed_sound = 294.9 # (m/s) at 50000ft
v = 1.6*speed_sound
time_all = 1000*np.array(distance_all)/v/60 #minutes
time_cities = 1000*np.array(distance_cities)/v/60 #minutes

# Extracting all data
all_data = {}
for month in ["06", "12"]:
    for hour in ["00", "12"]:
        filename = directory + year + month + day + '_' + hour + '_' + \
            str(alt_ft) + ".p"
        f = open(filename, "rb")
        data = pickle.load(f)

        # Process Nan numbers
        array = np.ma.masked_invalid(np.array(data.noise).reshape(data.lon_grid.shape))

        #get only the valid values
        lon1 = data.lon_grid[~array.mask]
        lat1 = data.lat_grid[~array.mask]
        newarr = array[~array.mask]

        noise = interpolate.griddata((lat1, lon1 ), newarr.ravel(),
                                  ( data.lat_grid, data.lon_grid),
                                     method='cubic')


        print(np.shape(data.elevation))
        # Preapre interpolation functions
        elevation = np.flip(data.elevation, 0)
        noise = np.flip(noise, 0)
        humidity = np.flip(np.transpose(data.humidity, (1, 2, 0)), 0)
        height = np.flip(np.transpose(data.height, (1, 2, 0)), 0)
        temperature = np.flip(np.transpose(data.temperature, (1, 2, 0)), 0)
        wind_x = np.flip(np.transpose(data.wind_x, (1, 2, 0)), 0)
        wind_y = np.flip(np.transpose(data.wind_y, (1, 2, 0)), 0)

        lonlat = np.array([data.lat[::-1], data.lon])
        f_elevation = RegularGridInterpolator(lonlat, elevation)
        f_humidity = RegularGridInterpolator(lonlat, humidity)
        f_height = RegularGridInterpolator(lonlat, height)
        f_temperature = RegularGridInterpolator(lonlat, temperature)
        f_wind_x = RegularGridInterpolator(lonlat, wind_x)
        f_wind_y = RegularGridInterpolator(lonlat, wind_y)
        f_noise = RegularGridInterpolator(lonlat, noise)

        # Interpolating
        all_data[(month, day)] = {}
        all_data[(month, day)]['elevation'] = f_elevation(path)
        all_data[(month, day)]['humidity'] = f_humidity(path)
        all_data[(month, day)]['height'] = f_height(path)
        all_data[(month, day)]['temperature'] = f_temperature(path)
        all_data[(month, day)]['wind_x'] = f_wind_x(path)
        all_data[(month, day)]['wind_y'] = f_wind_y(path)
        all_data[(month, day)]['noise'] = f_noise(path)

# Contour plot
pressures = np.array([100000, 97500, 95000, 92500, 90000, 85000, 80000, 75000, 70000,
                       65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000,
                       20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 700, 500, 300,
                       200, 100])
altitude = 10.**5/2.5577*(1-(pressures/101325)**(1/5.2558)) / 0.3048
LON, HEIGHT = np.meshgrid(altitude, distance_all)

for month in ["06", "12"]:
    for hour in ["00"]:
        plt.figure(figsize=(12, 4))
        plt.contourf(HEIGHT, LON, all_data[(month, day)]['humidity'], cmap=cm.Blues, levels=np.linspace(0, 100, 21), extend = 'max')
        plt.xlabel('Distance (km)')
        plt.ylabel('Sea-level altitude (ft)')
        clb = plt.colorbar()
        clb.set_label('Relative Humidity')
        plt.fill_between(distance_all, np.zeros(len(all_data[(month, day)]['elevation'])), all_data[(month, day)]['elevation'],
                        facecolor = ".5", edgecolor = 'k', lw=1)
        for distance in distance_cities:
            plt.axvline(x=distance, color = 'k', ls = '--', lw=2)
        plt.ylim([0, 50000])
        plt.title('Month: ' + month + ', Day:' + day + ', Hour: ' + hour)
plt.show()

LON, HEIGHT = np.meshgrid(altitude, time_all)
for month in ["06", "12"]:
    for hour in ["00"]:
        plt.figure(figsize=(12, 4))
        plt.contourf(HEIGHT, LON, all_data[(month, day)]['humidity'], cmap=cm.Blues, levels=np.linspace(0, 100, 21), extend = 'max')
        plt.xlabel('Time (min)')
        plt.ylabel('Sea-level altitude (ft)')
        clb = plt.colorbar()
        clb.set_label('Relative Humidity')
        plt.fill_between(time_all, np.zeros(len(all_data[(month, day)]['elevation'])), all_data[(month, day)]['elevation'],
                        facecolor = ".5", edgecolor = 'k', lw=1)
        for time in time_cities:
            plt.axvline(x=time, color = 'k', ls = '--', lw=2)
        plt.ylim([0, 50000])
        plt.title('Month: ' + month + ', Day:' + day + ', Hour: ' + hour)
plt.show()

plt.figure(figsize=(12, 6))
for month in ["06", "12"]:
    for hour in ["00"]:
        label = 'Month: ' + month + ', Day:' + day + ', Hour: ' + hour
        plt.plot(time_all, all_data[(month, day)]['noise'],
                 label= label)
        for time in time_cities:
            plt.axvline(x=time, color = 'k', ls = '--', lw=2)
plt.ylabel('Perceived level in dB')
plt.xlabel('Time (min)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for month in ["06", "12"]:
    for hour in ["00"]:
        label = 'Month: ' + month + ', Day:' + day + ', Hour: ' + hour
        max_values = []
        spacings = np.linspace(5, 100, 39)
        for spacing in spacings:
            points = scanner(distance_all, all_data[(month, day)]['noise'], spacing)
            points = points[points[:,0].argsort()]
            [x,y] = points.T
            print(month, hour, spacing, np.max(y), np.mean(y), np.std(y))
            max_values.append(np.max(np.abs(y)))

        speed_sound = 294.9 # (m/s) at 50000ft
        v = 1.6*speed_sound
        time = 1000*np.array(spacings)/v/60 #minutes
        plt.plot(time, max_values,  label=label, lw=2)
plt.axvline(x=1000*haversine(0, 0, 0, 0.5)/v/60, color = 'b', ls = '--', label = 'Grid Resolution')
plt.axvline(x=1000*30/v/60, color = 'r', ls = '--', label = 'LIDAR capability')
plt.xlabel('Time in between samples (min)')
plt.ylabel('Change in PLdB')
# plt.xscale('log')
plt.legend()
plt.show()
