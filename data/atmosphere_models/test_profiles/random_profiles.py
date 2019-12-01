import pickle
import random
import numpy as np

from weather.parameterize_atmosphere.autoencoder import interpolate_profiles, outliers

################################################################################
#                           Load profile data
################################################################################
cruise_alt = 13500
n = 75
balloon_data_path = '../test_data/US_2017_2018.p'

data = pickle.load(open(balloon_data_path,'rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

# Remove outliers from dataset
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated,
                                                          temp_interpolated,
                                                          elevations)

################################################################################
#                         Randomly choose profiles
################################################################################
number_of_profiles = 10
set = 0
profiles_name = 'profiles_%i_%i' % (number_of_profiles, set)

# Randomly select profiles to compute the noise calculations
random_list = []
i = 0
while i < number_of_profiles:
    r = random.randint(0,len(rh_interpolated)-1)
    if r not in random_list:
        random_list.append(r)
        i += 1
profiles = random_list[:]
profiles = {'profiles':profiles}

handle = open(profiles_name + '.p','wb')
pickle.dump(profiles, handle)
handle.close()
