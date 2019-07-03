import pickle
import numpy as np
from misc_cluster import profile_mixture_vector, prepare_profiles_GMM
from sklearn.mixture import GaussianMixture

data = pickle.load(open('./../../../72469_profiles.p','rb'))
humidity_profiles = data['humidity']
temperature_profiles = data['temperature']
cruise_altitude = 13500
n_components = 4

humidity_profiles_GMM = prepare_profiles_GMM(humidity_profiles[1:],
                                             humidity_profiles[0],
                                             cruise_altitude,
                                             include_reference = True)

mixture_vector = profile_mixture_vector(humidity_profiles_GMM)

GMM = GaussianMixture(n_components = n_components).fit(mixture_vector)

handle = open('GMM_balloon_profiles_4.p','wb')
pickle.dump({'model':GMM, 'reference_profile':humidity_profiles[0],
             'reference_temperature':temperature_profiles[0],
             'cruise_altitude':cruise_altitude,
             'feature_profile':humidity_profiles_GMM[0]}, handle)
handle.close()
