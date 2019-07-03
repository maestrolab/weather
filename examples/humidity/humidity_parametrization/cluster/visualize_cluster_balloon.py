import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from misc_humidity import package_data, prepare_standard_profiles
from misc_cluster import predict_clusters, calculate_average_profile,\
                         prepare_profiles_GMM, vapor_pressures_GMM

# Load atmospheric data
data = pickle.load(open('./../../../72469_profiles.p','rb'))

# Load model
model = pickle.load(open('GMM_balloon_profiles_4.p','rb'))

# Load standard profiles (need standard pressure profile)
path = './../../../../../data/weather/standard_profiles/standard_profiles.p'
standard_profiles = prepare_standard_profiles(standard_profiles_path = path)
data['pressure'] = [standard_profiles['original_pressures'] for i in range(
                    len(data['humidity']))]

# Prepare data
humidity_profiles = prepare_profiles_GMM(data['humidity'],
                                         model['reference_profile'],
                                         model['cruise_altitude'],
                                         include_reference = False)
temperature_profiles = prepare_profiles_GMM(data['temperature'],
                                            model['reference_temperature'],
                                            model['cruise_altitude'],
                                            include_reference = False)
pressure_profiles = prepare_profiles_GMM(data['pressure'],
                                         model['reference_profile'],
                                         model['cruise_altitude'],
                                         include_reference = False)

# Calculate vapor pressure profiles
vps_profiles = vapor_pressures_GMM(humidity_profiles, temperature_profiles,
                                   pressure_profiles)

# Predict cluster for each profile in data (profiles signified by index in array)
cluster_assignments = predict_clusters(humidity_profiles, model['model'])

# Calculate the average profile for each cluster
average_profile_per_cluster = calculate_average_profile(humidity_profiles,
                                            cluster_assignments, model['model'])

# Compute the average profile considering all profiles
profiles_to_average = np.array(humidity_profiles)
average_humidity_profile = np.mean(profiles_to_average, axis = 0)
average_alts, average_rh = package_data(average_humidity_profile)

profiles_to_average = np.array(vps_profiles)
average_vps_profile = np.mean(profiles_to_average, axis = 0)
average_alts, average_vps = package_data(average_vps_profile)

# Computing the mean and standard deviation for each profile
database = {i:{'mean':0, 'std':0, 'ave_std':0, 'color':''} for i in \
                                                range(len(humidity_profiles))}
colors = {0:'b', 1:'orange', 2:'g', 3:'r'}

for i in range(len(humidity_profiles)):
    alts, rh = package_data(humidity_profiles[i])
    cluster = cluster_assignments[i]

    df = pd.DataFrame({'rh':rh, 'ave_rh_cluster':\
                      average_profile_per_cluster[cluster[0]]})
    df_ave = pd.DataFrame({'rh':rh, 'ave_rh':average_rh})

    database[i]['rh_mean'] = df.rh.mean()
    database[i]['rh_std'] = df.rh.std()
    database[i]['rh_ave_std'] = (df_ave.std(axis = 1)).mean()
    database[i]['rh_ave_std_cluster'] = (df.std(axis = 1)).mean()

    # Compute the mean and standard deviation for the vapor pressure profiles
    alts, vps = package_data(vps_profiles[i])
    df_vps = pd.DataFrame({'vps':vps, 'ave_vps':average_vps})

    database[i]['vps_mean'] = df_vps.vps.mean()
    database[i]['vps_ave_std'] = (df_vps.std(axis = 1)).mean()

    database[i]['color'] = colors[cluster[0]]

# Plot average vs standard deviation
plots = ['rh_std', 'rh_ave_std', 'rh_ave_std_cluster']
plots_mean = ['vps_mean']
plots_std = ['vps_ave_std']
for mean_type in plots_mean:
    for std_type in plots_std:
        cluster_counts = {i:0 for i in range(4)}
        keys = database.keys()
        fig = plt.figure()
        for key in keys:
            label = None
            if cluster_counts[cluster_assignments[key][0]] == 0:
                label = cluster_assignments[key][0]
                cluster_counts[cluster_assignments[key][0]] += 1
            plt.scatter(database[key][mean_type], database[key][std_type],
                        color = database[key]['color'], alpha = 0.5,
                        label = label)
        plt.xlabel('$\mu_{vps}$')
        plt.ylabel('$\sigma_{vps}$')
        plt.legend(title = 'Cluster')

plt.show()
