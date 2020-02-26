import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras import backend as K

from weather.parameterize_atmosphere.autoencoder import *
from weather.boom import boom_runner

################################################################################
#                                 Parameters
################################################################################
# Balloon data
balloon_data  = 'G:/Shared drives/Maestro Team Drive/Misc File Sharing/' +\
                 'Atmospheric Profiles Machine Learning/balloon_data/2017+2018/'+\
                 'US_2017_2018.p'

# Variable bounds
feature_bounds = '../../data/atmosphere_models/feature_bounds.p'

# Interpolating parameters
cruise_alt = 13500 # meters
n = 75 # number of data points in interpolated profiles (model parameter)

# Profiles for noise calculations
profile_path = '../../data/atmosphere_models/test_profiles/profiles_1000_0.p'

# Specify the model to use based on the number of parameters used
n_params = 7

# File to write noise calculatiosn
noise_file = '../../data/atmosphere_models/noise/%i_parameters.txt' % (n_params)

################################################################################
#                               Load models
################################################################################
path = '../../data/atmosphere_models/trained_models/%i_parameters' % n_params
encoder = tf.keras.models.load_model(path + '_E.h5')
decoder = tf.keras.models.load_model(path + '_D.h5')

variable_bounds = pickle.load(open(feature_bounds, 'rb'))
################################################################################
#                           Load profile data
################################################################################
data = pickle.load(open(balloon_data,'rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh,
                                                          temp)

# Remove outliers from dataset
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated,
                                                          temp_interpolated,
                                                          elevations)

################################################################################
#                           Noise calculations
################################################################################
f = open(noise_file, 'w')

profiles = pickle.load(open(profile_path,'rb'))['profiles']
n_profiles = 0

for profile in profiles:
    n_profiles += 1
    print(n_profiles)
    # Normalize data to predict from model
    predict_data = np.hstack((rh_interpolated[profile][:,1],
                              temp_interpolated[profile][:,1]))
    elevation = np.array([data['height'][profile][0]])
    y = np.array([normalize_variable_bounds(predict_data, n, variable_bounds, elevation)])
    latent_rep = encoder.predict(y)
    y_pred = decoder.predict(latent_rep)
    y_pred_normalized = normalize_variable_bounds(y_pred[0][:-1], n, variable_bounds,
                                         np.array(y_pred[0][-1]),
                                         inverse = True)

    # Structure data for sBoom
    rh_profile = rh_interpolated[profile,:,:]
    temp_profile = temp_interpolated[profile,:,:]
    p_rh = np.array([alt_interpolated, y_pred_normalized[:n]]).T
    p_temp = np.array([alt_interpolated, y_pred_normalized[n:-1]]).T
    sBoom_data = [list(temp_profile), 0, list(rh_profile)]
    sBoom_data_parametrized = [list(p_temp), 0, list(p_rh)]

    # Noise parameters
    elevation = data['height'][profile][0] / 0.3048
    nearfield_file = './../../data/nearfield/25D_M16_RL5.p'

    # Noise calculations
    noise = {'original': 0, 'parameterized': 0, 'difference': 0}
    noise['original'] = boom_runner(sBoom_data, cruise_alt / 0.3048, elevation
                                    nearfield_file=nearfield_file)

    noise['parameterized'] = boom_runner(sBoom_data_parametrized, cruise_alt / 0.3048, elevation,
                                        nearfield_file=nearfield_file)
    noise['difference'] = noise['original'] - noise['parameterized']
    print(noise)

    height_to_ground_m = height_to_ground * 0.3048

    # Write to noise file
    write_line = [str(profile), noise['original'], noise['parameterized'],
                  noise['difference']]
    write_line.extend(latent_rep[0])
    write_line.append(eleva)

    f.write('\t'.join(map(str,write_line)) + '\n')

f.close()
