import pickle
import platform
import random
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.neighbors import KernelDensity
from scipy import interpolate
from numpy import linalg as LA
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import interp1d, RegularGridInterpolator, interp1d
from scipy.stats import norm, spearmanr

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
    for i in range(1000):
        x_sample = random.uniform(min, max)
        y_sample = f(x_sample) - f(x_sample + spacing)
        # print(x_sample, x_sample + spacing, f(x_sample), f(x_sample + spacing))
        points.append([x_sample, y_sample])
    points = np.array(points)
    return points

year = '2018'
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'];
days = ['01', '07', '14', '21', '28'];
hour = '12'
alt_ft = 50000
directory = '../../../data/noise/'

design = ''
# Determining path
# Setting up path
# Seattle, Boise, Denver, College Station, and Miami
# lat_cities = [47.6062, 43.6150, 39.7392, 30.6280, 25.7617, ]
# lon_cities = [-122.3321, -116.2023, -104.9903, -96.3344, -80.1918]
lat_cities = [47.6062, 43.6150, 39.7392, 32.7555, 25.7617, ]
lon_cities = [-122.3321, -116.2023, -104.9903, -97.3308, -80.1918]
lat_all = []
lon_all = []
distance_all = [0]
distance_cities = [0]
for i in range(len(lat_cities)-1):
    j = i+1
    lon_path = np.linspace(lon_cities[i], lon_cities[j], 25)
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
print('D',distance_cities)
path = np.array([lat_all, lon_all]).T
# Time
speed_sound = 294.9 # (m/s) at 50000ft
v = 1.6*speed_sound
time_all = 1000*np.array(distance_all)/v/60 #minutes
time_cities = 1000*np.array(distance_cities)/v/60 #minutes

# Contour plot
pressures = np.array([100000, 97500, 95000, 92500, 90000, 85000, 80000, 75000, 70000,
                       65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000,
                       20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 700, 500, 300,
                       200, 100])
altitude = 10.**5/2.5577*(1-(pressures/101325)**(1/5.2558)) / 0.3048
print(altitude)
BRAKE
LON, HEIGHT = np.meshgrid(altitude, distance_all)

all_data = {}
df = pd.DataFrame({'location': [], 'noise': [], 'elevation': [], 'month':[], 'day':[], 'humidity':[], 'temperature':[], 'wind_x':[], 'wind_y':[]})
indexes = range(9,21)
for index in indexes:
    for month in range(1,13):
        month = '%02i' % month
        for day in range(1,32):
            try:
                day = '%02i' % day
                all_data[(month, day)] = pickle.load(open(directory + '/' + design + 'path_' + year + month + day + '_' + hour + '_' + str(alt_ft) + '.p', 'rb'))
                # Getting rid of Nan numbers
                array = np.ma.masked_invalid(all_data[(month, day)]['noise'])

                #get only the valid values
                d = np.array(distance_all)[~array.mask]
                newarr = array[~array.mask]
                f = interpolate.interp1d(d, newarr, fill_value="extrapolate")
                all_data[(month, day)]['noise'] = f(distance_all)
                rh_index = []
                for i in range(len(all_data[(month, day)]['humidity'])):
                    rh_i = all_data[(month, day)]['humidity'][i]
                    rh_index.append(rh_i[index])

                temperature_index = []
                for i in range(len(all_data[(month, day)]['temperature'])):
                    rh_i = all_data[(month, day)]['temperature'][i]
                    temperature_index.append(rh_i[index])
                    
                wind_x_index = []
                for i in range(len(all_data[(month, day)]['wind_x'])):
                    rh_i = all_data[(month, day)]['wind_x'][i]
                    wind_x_index.append(rh_i[index])
                    
                wind_y_index = []
                for i in range(len(all_data[(month, day)]['wind_y'])):
                    rh_i = all_data[(month, day)]['wind_y'][i]
                    wind_y_index.append(rh_i[index])
                    
                # Checking for Nan numbers
                df2 = pd.DataFrame({'location': distance_all,
                                    'month': month,
                                    'day': day,
                                    'noise': all_data[(month, day)]['noise'],
                                    'elevation': all_data[(month, day)]['elevation'],
                                    'humidity': rh_index,
                                    'temperature': temperature_index,
                                    'wind_x': wind_x_index,
                                    'wind_y': wind_y_index})
                df = df.append(df2)
                print(directory +design+'pressure_' +'%02d' % index +'_'+ year + month + day + '_' + hour + '_' + str(alt_ft) + '.p')
            except(FileNotFoundError):
                pass
    df.reset_index(drop=True, inplace=True)
    pickle.dump(df, open(directory + design+ 'pressure_' +'%02d' % index + '.p', 'wb'))
