import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.signal import savgol_filter

from misc_humidity import package_data, prepare_standard_profiles
from misc_cluster import predict_clusters, calculate_average_profile,\
    prepare_profiles_GMM, vapor_pressures_GMM

colors = [[0, 0.4470, 0.7410],
          [0.8500, 0.3250, 0.0980],
          [0.9290, 0.6940, 0.1250],
          [0.4940, 0.1840, 0.5560],
          [0.4660, 0.6740, 0.1880],
          [0.3010, 0.7450, 0.9330],
          [0.6350, 0.0780, 0.1840]]

# Load atmospheric data ./../../../72469_profiles.p
data = pickle.load(open('./72469_noise.p', 'rb'))
n = 200
n_clusters = 4
rh = np.array(data['humidity'])
m = len(rh)
alt_interpolated = np.linspace(0, 13500, n)

data_interpolated = np.zeros((len(rh), len(alt_interpolated), 2))
for i in range(m):
    alt, values = np.array(rh[i]).T
    # values = savgol_filter(values, 11, 3)
    fun = interp1d(alt, values)
    values_interpolated = fun(alt_interpolated)
    data_interpolated[i] = np.array([alt_interpolated, values_interpolated]).T

# metrics
average = np.array([np.average(data_interpolated[i, :, 1]) for i in range(m)])

maximum = np.array([np.max(data_interpolated[i, :, 1]) for i in range(m)])
ground = data_interpolated[:, 0, 1]
indices_of_maximum = [np.argmax(data_interpolated[i, :, 1]) for i in range(m)]
location_of_maximum = data_interpolated[i, indices_of_maximum, 0]

average_profile = np.array([np.average(data_interpolated[0:m, j, 1]) for j in range(n)])
standard = np.array(
    [np.average(np.absolute(data_interpolated[i, :, 1]-average_profile)) for i in range(m)])

# Clustering time
points = data_interpolated[:, :, 1]
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(points)
y_km = kmeans.fit_predict(points)
centers = kmeans.cluster_centers_
centers = [np.average(centers[i]) for i in range(n_clusters)]
print(centers)
print(np.argsort(centers))
indexes = np.arange(n_clusters)[np.argsort(centers)]
print(indexes)
print(np.array(colors)[indexes])


plt.figure()
s = plt.scatter(average, location_of_maximum, c=data['noise'], cmap='gray')
plt.colorbar(s)

plt.figure()
for ii in indexes:
    plt.scatter(average[y_km == ii], np.array(data['noise'])[y_km == ii],
                c=colors[ii], label=ii)
# plt.legend()

plt.figure()
for ii in indexes:
    plt.scatter(average[y_km == ii], maximum[y_km == ii],
                c=colors[ii], label=ii)

plt.figure()
for ii in indexes:
    plt.scatter(average[y_km == ii], location_of_maximum[y_km == ii],
                c=colors[ii], label=ii)
                
plt.figure()
for ii in indexes:
    plt.scatter(location_of_maximum[y_km == ii], np.array(data['noise'])[y_km == ii],
                c=colors[ii], label=ii)

plt.figure()
for ii in indexes:
    plt.scatter(average[y_km == ii], np.array(data['noise'])[y_km == ii],
                c=colors[ii], label=ii)

plt.figure()
for ii in indexes:
    plt.scatter(maximum[y_km == ii], np.array(data['noise'])[y_km == ii],
                c=colors[ii], label=ii)
# plt.legend()

plt.figure()
data = [data_interpolated[y_km == i] for i in range(n_clusters)]
average_plot = np.array([[np.average(data[i][:, :, 1], axis=0)] for i in range(n_clusters)])

for jj in indexes:
    plt.subplot(2, 2, jj+1)
    data_i = data_interpolated[y_km == jj]
    for i in range(n):
        x, y = data_i[i].T
        plt.plot(y, x, jj, color='k', alpha=0.05)
    plt.plot(average_plot[jj][0, :], data[jj][0, :, 0], color=colors[jj])
plt.show()
