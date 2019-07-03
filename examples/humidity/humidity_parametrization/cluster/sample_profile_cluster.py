import pickle
import numpy as np
import matplotlib.pyplot as plt
from misc_cluster import truncate_at_altitude
from misc_humidity import package_data

# Load Gaussian Mixture Model and reference profile used
data = pickle.load(open('GMM_balloon_profiles_4.p','rb'))

# Prepare altitudes for plotting
profile = truncate_at_altitude(data['reference_profile'], cruise_altitude =
                               data['cruise_altitude'])
alts, rh = package_data(profile)

# Sample model
n_samples = 10000
sample = data['model'].sample(n_samples = n_samples)
cluster_sum = {i:0 for i in range(len(sample[1]))}
cluster_count = {i:0 for i in range(len(sample[1]))}
for i in range(len(sample[1])):
    key = sample[1][i]
    cluster_sum[key] += sample[0][i]
    cluster_count[key] += 1

cluster_means = [cluster_sum[i]/cluster_count[i] for i in range(n_samples)
                 if cluster_count[i] != 0]

fig = plt.figure()
for i in range(len(cluster_means)):
    label = 'Cluster %i' % list(cluster_count.keys())[i]
    plt.plot(cluster_means[i], alts, label = label)
plt.legend()
plt.xlabel('Relative Humidity [%]')
plt.ylabel('Altitude [m]')
plt.show()
