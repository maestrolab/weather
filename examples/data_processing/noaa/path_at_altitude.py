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
print(lat_all[-2], lon_all[-2])
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
LON, HEIGHT = np.meshgrid(altitude, distance_all)

all_data = {}
df = pd.DataFrame({'location': [], 'noise': [], 'elevation': [], 'month':[], 'day':[]})
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
            rh_average = []
            rh_max = []
            for i in range(len(all_data[(month, day)]['humidity'])):
                rh_i = all_data[(month, day)]['humidity'][i]
                rh_average.append(np.average(rh_i))
                rh_max.append(np.max(rh_i))

            # Checking for Nan numbers
            df2 = pd.DataFrame({'location': distance_all,
                                'month': month,
                                'day': day,
                                'noise': all_data[(month, day)]['noise'],
                                'elevation': all_data[(month, day)]['elevation'],
                                'average_rh': rh_average,
                                'max_rh': rh_max})
            df = df.append(df2)
            print(directory + '/'+design+'path_' + year + month + day + '_' + hour + '_' + str(alt_ft) + '.p')
        except(FileNotFoundError):
            pass
df.reset_index(drop=True, inplace=True)
pickle.dump(df, open(directory + design+'path_noise.p', 'wb'))
# Section to get data for Points of Interest
# POI = {'location': [0, 0, 133.6, 133.6, 133.6, 133.6, 133.6, 133.6, 651,
#                     651, 1503, 1580, 1580, 1580, 1676, 1676, 1676, 2957,
#                     2957, 4627, 4627],
#         'noise': [83.81, 78.23, 88.66, 86.62, 83.92, 81.46, 80.13, 79.45,
#                   88.20, 79.52, 84.32, 90.33, 88.37, 83.80, 89.80, 85.36,
#                   81.03, 82.87, 80.47, 82.10, 80.13],
#         'quantile': [99.5, 0.5, 99.5, 95, 75, 25, 5, 0.5, 99.5, 0.5, 0.5, 99.5,
#                      50, 0.5, 99.5, 50, 0.5, 99.5, 0.05, 99.5, 50]}
# indexes = len(POI['location'])*[0]
# residuals = len(POI['location'])*[99999]
# for i in range(len(POI['location'])):
#     for month in range(1,13):
#         month = '%02i' % month
#         for day in range(1,31):
#             try:
#                 day = '%02i' % day
#                 index = np.argmin(np.abs(np.array(distance_all) - POI['location'][i]))
#                 residual = np.min(np.abs(np.array(distance_all) - POI['location'][i]))
#                 if residual < residuals[i]:
#                     residuals[i] = residual
#                     indexes[i] = index
#             except(KeyError):
#                 pass
#
# months = len(POI['location'])*[0]
# days = len(POI['location'])*[0]
# residuals = len(POI['location'])*[99999]
# for i in range(len(POI['location'])):
#     for month in range(1,13):
#         month = '%02i' % month
#         for day in range(1,31):
#                 day = '%02i' % day
#                 try:
#                     residual = np.abs(all_data[(month, day)]['noise'][indexes[i]] - POI['noise'][i])
#                     if residual < residuals[i]:
#                         residuals[i] = residual
#                         months[i] = month
#                         days[i] = day
#                 except(KeyError):
#                     pass

#================================================
# PLOT NOISE ALONG THE path_height
#================================================

# Along location
mean = df.groupby('location')['noise'].mean()
q_25 = df.groupby('location')['noise'].quantile(0.25)
q_75 = df.groupby('location')['noise'].quantile(0.75)
q_05 = df.groupby('location')['noise'].quantile(0.05)
q_95 = df.groupby('location')['noise'].quantile(0.95)
q_005 = df.groupby('location')['noise'].quantile(0.005)
q_995 = df.groupby('location')['noise'].quantile(0.995)
max = df.groupby('location')['noise'].max()
min = df.groupby('location')['noise'].min()

# Along elevation
mean_e = df.groupby('elevation')['noise'].mean()
q_25_e = df.groupby('elevation')['noise'].quantile(0.25)
q_75_e = df.groupby('elevation')['noise'].quantile(0.75)
q_05_e = df.groupby('elevation')['noise'].quantile(0.05)
q_95_e = df.groupby('elevation')['noise'].quantile(0.95)
q_005_e = df.groupby('elevation')['noise'].quantile(0.005)
q_995_e = df.groupby('elevation')['noise'].quantile(0.995)
max_e = df.groupby('elevation')['noise'].max()
min_e = df.groupby('elevation')['noise'].min()

# Distance based
plt.figure(figsize=(12, 4))
x = df['location'].unique()

city_names = ['Seattle', 'Boise', 'Denver', 'Dallas', 'Miami']
plt.fill_between(x, q_005, q_995, color='.4', alpha=0.2, lw=0, label = '99% Percentile')
plt.fill_between(x, q_05, q_95, color='.4', alpha=0.2, lw=0, label = '90% Percentile')
plt.fill_between(x, q_25, q_75, color='.4', alpha=0.2, lw=0, label = '50% Percentile')
plt.plot(x, mean, 'k', label = 'Mean')
plt.plot(x, max, 'k', linestyle = '--', label = 'Maximum')
plt.plot(x, min, 'k', linestyle = '-.', label = 'Minimum')
for i in range(len(distance_cities)):
    distance = distance_cities[i]
    plt.axvline(x=distance, color = 'b', ls = '--', lw=2)
    if i==4:
        plt.text(distance - 250, 76, ' ' + city_names[i], color = 'b')
    else:
        plt.text(distance, 76, ' ' + city_names[i], color = 'b')
plt.legend()
plt.ylabel('Perceived level in dB')
plt.xlabel('Distance (km)')
plt.xlim([0, distance])

plt.figure(figsize=(12, 4))
zeros = np.zeros(len(x))
plt.fill_between(x, zeros, q_995-q_005, color='.4', alpha=0.2, lw=0, label = '99% Percentile')
plt.fill_between(x, zeros, q_95 - q_05, color='.4', alpha=0.2, lw=0, label = '90% Percentile')
plt.fill_between(x, zeros, q_75 - q_25, color='.4', alpha=0.2, lw=0, label = '50% Percentile')
plt.plot(x, max - min, 'k', linestyle = '--', label = 'Maximum')

for i in range(len(distance_cities)):
    distance = distance_cities[i]
    plt.axvline(x=distance, color = 'b', ls = '--', lw=2)
    if i==4:
        plt.text(distance - 250, 0, ' ' + city_names[i], color = 'b')
    else:
        plt.text(distance, 0, ' ' + city_names[i], color = 'b')

plt.legend()
plt.ylabel('Change in Perceived level in dB')
plt.xlabel('Distance (km)')
plt.ylim([0, 14])
plt.xlim([0, distance])
plt.show()

# Time based
plt.figure(figsize=(12, 4))
x = np.unique(time_all)

plt.fill_between(x, q_005, q_995, color='.4', alpha=0.2, lw=0, label = '99% Percentile')
plt.fill_between(x, q_05, q_95, color='.4', alpha=0.2, lw=0, label = '90% Percentile')
plt.fill_between(x, q_25, q_75, color='.4', alpha=0.2, lw=0, label = '50% Percentile')
plt.plot(x, mean, 'k', label = 'Mean')
plt.plot(x, max, 'k', linestyle = '--', label = 'Maximum')
plt.plot(x, min, 'k', linestyle = '-.', label = 'Minimum')
for i in range(len(time_cities)):
    t = time_cities[i]
    plt.axvline(x=t, color = 'b', ls = '--', lw=2)
    if i==4:
        plt.text(t - 1000*250/v/60 , 64, ' ' + city_names[i], color = 'b')
    else:
        plt.text(t, 64, ' ' + city_names[i], color = 'b')
# plt.legend()
plt.ylim([64, 86])
plt.ylabel('Perceived level in dB')
plt.xlabel('Time (min)')
plt.xlim([0, t])

plt.figure(figsize=(12, 4))
zeros = np.zeros(len(x))
plt.fill_between(x, zeros, q_995-q_005, color='.4', alpha=0.2, lw=0, label = '99% Percentile')
plt.fill_between(x, zeros, q_95 - q_05, color='.4', alpha=0.2, lw=0, label = '90% Percentile')
plt.fill_between(x, zeros, q_75 - q_25, color='.4', alpha=0.2, lw=0, label = '50% Percentile')
plt.plot(x, max - min, 'k', linestyle = '--', label = 'Maximum')

for i in range(len(time_cities)):
    t = time_cities[i]
    plt.axvline(x=t, color = 'b', ls = '--', lw=2)
    if i==4:
        plt.text(t - 1000*250/v/60, 0, ' ' + city_names[i], color = 'b')
    else:
        plt.text(t, 0, ' ' + city_names[i], color = 'b')

plt.legend()
plt.ylabel('Change in Perceived level in dB')
plt.xlabel('Time (min)')
plt.ylim([0, 14])
plt.xlim([0, t])
plt.show()

#===============================
# PLOT RELATION BETWEEN ELEVATION AND NOISE
plt.figure()
import statsmodels.formula.api as sm
result = sm.ols(formula="noise ~ elevation", data=df).fit()
  
print('Spearman correlation elevation and noise ', spearmanr(df['elevation'], df['noise']))

print(result.params)
x = np.unique(df['elevation'])
print('x', x)
print('q005', q_005_e)
print('q995', q_995_e)
plt.fill_between(np.array(x, dtype=float),
                 np.array(q_005_e, dtype=float),
                 np.array(q_995_e, dtype=float),
                 color='.4', alpha=0.2, lw=0, label = '99% Percentile')
plt.fill_between(np.array(x, dtype=float),
                 np.array(q_05_e, dtype=float),
                 np.array(q_95_e, dtype=float),
                 color='.4', alpha=0.2, lw=0, label = '90% Percentile')
plt.fill_between(np.array(x, dtype=float),
                 np.array(q_25_e, dtype=float),
                 np.array(q_75_e, dtype=float),
                 color='.4', alpha=0.2, lw=0, label = '50% Percentile')
# plt.plot(x, mean_e, 'k', label = 'Mean')
# plt.plot(x, max_e, 'k', linestyle = '--', label = 'Maximum')
# plt.plot(x, min_e, 'k', linestyle = '-.', label = 'Minimum')
plt.plot(x, result.params[0] + result.params[1]*x, 'k', label='Linear fit')
# plt.scatter(df['elevation'], df['noise'], alpha=0.3, label = 'Data')
plt.xlabel('Elevation (ft)')
plt.ylabel('PLdB')
plt.legend()
plt.show()

#===============================
# WEATHER
# for month in months:
#     for day in days:
#         plt.figure(figsize=(12, 4))
#         plt.contourf(HEIGHT, LON, all_data[(month, day)]['humidity'], cmap=cm.Blues, levels=np.linspace(0, 100, 21), extend = 'max')
#         plt.xlabel('Distance (km)')
#         plt.ylabel('Sea-level altitude (ft)')
#         clb = plt.colorbar()
#         clb.set_label('Relative Humidity')
#         plt.fill_between(distance_all, np.zeros(len(all_data[(month, day)]['elevation'])), all_data[(month, day)]['elevation'],
#                         facecolor = ".5", edgecolor = 'k', lw=1)
#         for distance in distance_cities:
#             plt.axvline(x=distance, color = 'k', ls = '--', lw=2)
#         plt.ylim([0, 50000])
#         plt.title('Month: ' + month + ', Day:' + day + ', Hour: ' + hour)
# plt.show()

# LON, HEIGHT = np.meshgrid(altitude, time_all)
# for month in months:
#     for day in days:
#         plt.figure(figsize=(12, 4))
#         plt.contourf(HEIGHT, LON, all_data[(month, day)]['humidity'], cmap=cm.Blues, levels=np.linspace(0, 100, 21), extend = 'max')
#         plt.xlabel('Time (min)')
#         plt.ylabel('Sea-level altitude (ft)')
#         clb = plt.colorbar()
#         clb.set_label('Relative Humidity')
#         plt.fill_between(time_all, np.zeros(len(all_data[(month, day)]['elevation'])), all_data[(month, day)]['elevation'],
#                         facecolor = ".5", edgecolor = 'k', lw=1)
#         for time in time_cities:
#             plt.axvline(x=time, color = 'k', ls = '--', lw=2)
#         plt.ylim([0, 50000])
#         plt.title('Month: ' + month + ', Day:' + day + ', Hour: ' + hour)
# plt.show()

spacings = np.linspace(5, 100, 20)

data_p = {'mean':[], 'q25':[], 'q75':[], 'q05':[], 'q95':[], 'q005':[],
          'q995':[], 'location_f':[],'max':[], 'locations':[]}
for spacing in spacings:
    data_i = {'location': [], 'noise':[]}
    for month in range(1,13):
        month = '%02i' % month
        for day in range(1,32):
            try:
                day = '%02i' % day
                label = 'Month: ' + month + ', Day:' + day + ', Hour: ' + hour
                points = scanner(distance_all, all_data[(month, day)]['noise'], spacing)
                points = points[points[:,0].argsort()]
                [x,y] = points.T
                print(spacing, month, day, np.max(y), np.mean(y), np.std(y))
                data_i['noise']+= list(y)
                data_i['location'].append(x[np.argmax(np.abs(y))])
            except(KeyError):
                pass
    df_i = pd.DataFrame({'noise': np.abs(data_i['noise'])})
    data_p['mean'].append(df_i['noise'].mean())
    data_p['q25'].append(df_i['noise'].quantile(0))
    data_p['q75'].append(df_i['noise'].quantile(0.50))
    data_p['q05'].append(df_i['noise'].quantile(0.0))
    data_p['q95'].append(df_i['noise'].quantile(0.90))
    data_p['q005'].append(df_i['noise'].quantile(0.00))
    data_p['q995'].append(df_i['noise'].quantile(0.99))
    data_p['max'].append(df_i['noise'].max())
    data_p['locations'] += data_i['location']
    data_p['location_f'].append(KernelDensity(kernel='gaussian', bandwidth=0.75).fit(np.array(data_i['location'])[:, np.newaxis]))

del all_data
# df = pd.DataFrame(data)

# Distance based
plt.figure(figsize=(12, 4))
x = spacings
speed_sound = 294.9 # (m/s) at 50000ft
v = 1.6*speed_sound
x = 1000*np.array(x)/v/60 #minutes

plt.fill_between(x, data_p['q005'], data_p['q995'], color='.4', alpha=0.2, lw=0, label = '99% Percentile')
plt.fill_between(x, data_p['q05'], data_p['q95'], color='.4', alpha=0.2, lw=0, label = '90% Percentile')
plt.fill_between(x, data_p['q25'], data_p['q75'], color='.4', alpha=0.2, lw=0, label = '50% Percentile')
plt.plot(x, data_p['mean'], 'k', label = 'Mean')
plt.plot(x, data_p['max'], 'k', linestyle = '--', label = 'Maximum')

plt.legend()
plt.ylabel('Change in PLdB')
plt.xlabel('Time in between samples (min)')
plt.show()

# Plot as a function of time
plt.figure(figsize=(12, 4))
plt.hist(data_p['locations'], bins=100, stacked =True)
p = 1./len(data_p['locations'])
values = np.array([0, 0.1, 0.2, 0.3])
plt.yticks(values/p, values)
plt.xlabel('Distance (km)')
plt.ylabel('Probability')
plt.show()
# fig, ax = plt.subplots()
# for i in range(len(spacings)):
#     rho = np.exp(data_p['location_f'][i].score_samples(np.array(distance_all)[:, np.newaxis]))
#     ax.fill(distance_all, rho, color='.4', alpha=0.2, label = spacing[i])
# plt.legend()
# plt.show()


# plt.figure(figsize=(12, 6))
# for month in months:
#     for day in days:
#         label = 'Month: ' + month + ', Day:' + day + ', Hour: ' + hour
#         max_values = []
#         spacings = np.linspace(5, 100, 39)
#         for spacing in spacings:
#             points = scanner(distance_all, all_data[(month, day)]['noise'], spacing)
#             points = points[points[:,0].argsort()]
#             [x,y] = points.T
#             print(month, hour, spacing, np.max(y), np.mean(y), np.std(y))
#             max_values.append(np.max(np.abs(y)))
#
#         speed_sound = 294.9 # (m/s) at 50000ft
#         v = 1.6*speed_sound
#         time = 1000*np.array(spacings)/v/60 #minutes
#         plt.plot(time, max_values,  label=label, lw=2)
# plt.axvline(x=1000*haversine(0, 0, 0, 0.5)/v/60, color = 'b', ls = '--', label = 'Grid Resolution')
# plt.axvline(x=1000*30/v/60, color = 'r', ls = '--', label = 'LIDAR capability')
# plt.xlabel('Time in between samples (min)')
# plt.ylabel('Change in PLdB')
# # plt.xscale('log')
# plt.legend()
# plt.show()
