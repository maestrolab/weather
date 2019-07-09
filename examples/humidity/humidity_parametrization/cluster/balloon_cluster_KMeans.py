import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

from misc_humidity import prepare_standard_profiles, calculate_vapor_pressures,\
                          convert_to_celcius

# Load standard profiles (need standard pressure profile)
path = './../../../../data/weather/standard_profiles/standard_profiles.p'
standard_profiles = prepare_standard_profiles(standard_profiles_path = path)

# Load atmospheric data
data = pickle.load(open('./../../../boom/72469_profiles.p','rb'))
humidity_profiles = np.array([np.array(p) for p in data['humidity']])
temperature_profiles = np.array(data['temperature'])
pressure_profiles = np.array(standard_profiles['original_pressures'])

cruise_altitude = 13500
n = 200
n_clusters = 4

# Interpolate profiles
alts_interpolate = np.linspace(0, cruise_altitude, n)
keys = ['rh','temp','vps']
data_interpolated = {key:np.zeros((len(humidity_profiles), len(alts_interpolate)
                     , 2)) for key in keys}
for i in range(len(humidity_profiles)):
    alts, rh = np.array(humidity_profiles[i]).T
    rh_fun = interp1d(alts, rh)
    alts, temp = np.array(temperature_profiles[i]).T
    temp_fun = interp1d(alts,temp)
    rh_interp = rh_fun(alts_interpolate)
    temp_interp = temp_fun(alts_interpolate)
    data_interpolated['rh'][i] = np.array([alts_interpolate, rh_interp]).T
    data_interpolated['temp'][i] = np.array([alts_interpolate, temp_interp]).T

alts, pressures = pressure_profiles.T
pres_fun = interp1d(alts,pressures)
pressures = pres_fun(alts_interpolate)
data_interpolated['pressure'] = np.array([alts_interpolate, pressures]).T

# Clustering
mixture_vector = data_interpolated['rh'][:,:,1]
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(mixture_vector)
y_km = kmeans.fit_predict(mixture_vector)

# Vapor Pressure Calculations
alts, pres = data_interpolated['pressure'].T
for i in range(len(humidity_profiles)):
    alts, rh = data_interpolated['rh'][i].T
    alts, temp = data_interpolated['temp'][i].T
    temp = convert_to_celcius(temp)
    vps = calculate_vapor_pressures(rh, temp, pres)
    data_interpolated['vps'][i] = np.array([alts, vps[0]]).T

# Metrics
average_rh = np.array([np.average(profile[:,1]) for profile in
                      data_interpolated['rh']])
average_vps = np.array([np.average(profile[:,1]) for profile in
                       data_interpolated['vps']])
average_rh_profile = np.array([np.average(data_interpolated['rh'][:,i,1]) for i
                              in range(n)])
average_vps_profile = np.array([np.average(data_interpolated['vps'][:,i,1]) for
                               i in range(n)])
deviation_rh = np.array([np.average(np.absolute(data_interpolated['rh'][i,:,1]-
             average_rh_profile)) for i in range(len(data_interpolated['rh']))])
deviation_vps = np.array([np.average(np.absolute(data_interpolated['vps'][i,:,1]
          -average_vps_profile)) for i in range(len(data_interpolated['vps']))])

# Plot average vs. deviation
averages = [average_rh, average_vps]
deviations = [deviation_rh, deviation_vps]

plt.figure(figsize=(12,5))
xlabels = ['$\mu_{rh}$', '$\mu_{vps}$']
ylabels = ['$deviation_{rh}$', '$deviation_{vps}$']

for i in range(len(averages)):
    plt.subplot(1,2,i+1,xlabel=xlabels[i],ylabel=ylabels[i])
    plt.scatter(averages[i], deviations[i], c=y_km)

# Plot clusters
data = [data_interpolated['rh'][y_km == i] for i in range(n_clusters)]
average_plot = np.array([[np.average(data[i][:,:,1], axis = 0)] for i in range(n_clusters)])

plt.figure()
for j in range(n_clusters):
    plt.subplot(2,2,j+1,xlabel='Relative Humidity [%]',ylabel='Altitude [m]')
    data_i = data_interpolated['rh'][y_km == j]
    for profile in data_i:
        alts, rh = profile.T
        plt.plot(rh, alts, j, alpha=0.2, color = 'k')
    plt.plot(average_plot[j][0,:], data[j][0,:,0], color = 'r')

# Plot clusters (vapor pressure profiles)
data = [data_interpolated['vps'][y_km == i] for i in range(n_clusters)]
average_plot = np.array([[np.average(data[i][:,:,1], axis = 0)] for i in range(n_clusters)])

plt.figure()
for j in range(n_clusters):
    plt.subplot(2,2,j+1,xlabel='Vapor Pressure [kPa]',ylabel='Altitude [m]')
    data_i = data_interpolated['vps'][y_km == j]
    for profile in data_i:
        alts, vps = profile.T
        plt.plot(vps, alts, j, alpha=0.2, color = 'k')
    plt.plot(average_plot[j][0,:], data[j][0,:,0], color = 'r')
plt.show()
