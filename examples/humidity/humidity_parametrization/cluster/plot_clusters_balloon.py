import pickle
import numpy as np
import matplotlib.pyplot as plt

from misc_humidity import package_data
from misc_cluster import interpolate_profiles, profile_mixture_vector,\
                         truncate_at_altitude, predict_clusters,\
                         calculate_average_profile

# Load atmospheric data
data = pickle.load(open('./../../../72469_profiles.p','rb'))

# Load model
model = pickle.load(open('GMM_balloon_profiles_4.p','rb'))

# Prepare data for cluster prediction
interpolated_profiles = interpolate_profiles(model['reference_profile'],
                                             data['humidity'],
                                             include_reference = False)

initial_profile = truncate_at_altitude(interpolated_profiles[0], cruise_altitude\
                                       = model['cruise_altitude'])
truncated_profiles = np.array([initial_profile])

for profile in interpolated_profiles[1:]:
    truncated_profile = truncate_at_altitude(profile, cruise_altitude = \
                                             model['cruise_altitude'])
    truncated_profiles = np.append(truncated_profiles, [truncated_profile], axis = 0)

# Predict cluster for each profile in data (profiles signified by index in array)
cluster_assignments = predict_clusters(truncated_profiles, model['model'])

# Calculate the average profile for each cluster
average_profile = calculate_average_profile(truncated_profiles,
                                            cluster_assignments, model['model'])

# Plot profiles separated by cluster
row_col = {0:[0,0], 1:[0,1], 2:[1,0], 3:[1,1]}

keys = list(cluster_assignments.keys())
fig, ax = plt.subplots(nrows = 2, ncols = 2)
for key in keys[:]:
    alts, rh = package_data(truncated_profiles[key])
    row, col = row_col[cluster_assignments[key][0]]
    ax[row][col].plot(rh, alts, alpha = 0.2)#, color = 'k')

# Set x and y axis titles for each subplot
j = list(range(4))
i = 0
for row in range(ax.shape[0]):
    for col in range(ax.shape[1]):
        ax[row][col].set_xlabel('Relative Humidity [%]')
        ax[row][col].set_ylabel('Altitdue [m]')
        ax[row][col].set_title('Cluster %i' % j[i])
        i += 1

average_color = 'dodgerblue'
ax[0][0].plot(average_profile[0], alts, color = average_color)
ax[0][1].plot(average_profile[1], alts, color = average_color)
ax[1][0].plot(average_profile[2], alts, color = average_color)
ax[1][1].plot(average_profile[3], alts, color = average_color)

plt.show()
