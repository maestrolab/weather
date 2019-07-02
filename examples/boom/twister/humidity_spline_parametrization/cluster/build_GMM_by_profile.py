import pickle
import numpy as np
from misc_cluster import interpolate_profiles, profile_mixture_vector,\
                         truncate_at_altitude
from sklearn.mixture import GaussianMixture

data = pickle.load(open('./../../../72469_profiles.p','rb'))
humidity_profiles = data['humidity']
temperature_profiles = data['temperature']
cruise_altitude = 13500
n_components = 4

interpolated_humidity_profiles = interpolate_profiles(humidity_profiles[0],
                                                      humidity_profiles[1:],
                                                      include_reference = True)

for i in range(len(interpolated_humidity_profiles)):
    interpolated_humidity_profiles[i] = truncate_at_altitude(
                                              interpolated_humidity_profiles[i],
                                              cruise_altitude = cruise_altitude)


mixture_vector = profile_mixture_vector(interpolated_humidity_profiles)

GMM = GaussianMixture(n_components = n_components).fit(mixture_vector)

handle = open('GMM_balloon_profiles_4.p','wb')
pickle.dump({'model':GMM, 'reference_profile':humidity_profiles[0],
             'reference_temperature':temperature_profiles[0],
             'cruise_altitude':cruise_altitude,
             'feature_profile':interpolated_humidity_profiles[0]}, handle)
handle.close()
