import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras import backend as K
from autoencoder import *
from weather.boom import boom_runner

################################################################################
#                               Load model
################################################################################
n_params = 4

min_RH = 'constant_min'
max_RH = 'max'
min_temp = 'min'
max_temp = 'max'

# RUN WITH CONSTANT_TYPE = '5_params_varaible_rh_temp_profiles_2'
# ASDFA
constant_type = 'test' # 'variable_rh_temp_2'
variable_type = 'both' # overwrites the min and max choices above
path = 'trained_models/%i_params_%s_' % (n_params, constant_type)

autoencoder = tf.keras.models.load_model(path + 'AE.h5')
encoder = tf.keras.models.load_model(path + 'E.h5')

variable_bounds = pickle.load(open('feature_bounds.p','rb'))

run = '_1000'
profiles_name = 'profiles' + run
constant_type += run + 'test'
################################################################################
#                           Load profile data
################################################################################
balloon_data  ='balloon_data/2017+2018/US_2017_2018'

data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
cruise_alt = 13500
n = int((autoencoder.layers[0].input_shape[1]-1)/2)
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

# Remove outliers from dataset
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated, temp_interpolated, elevations)

################################################################################
#                   Open file to write noise and parameters
################################################################################
filepath = 'noise/' + '%i_params_%s' % (n_params, constant_type)
f = open(filepath + '.txt','w')

################################################################################
#      Load profiles for noise calculations (previously randomly selected)
################################################################################
profiles = pickle.load(open(profiles_name + '.p','rb'))['profiles']

# Randomly select profiles to compute the noise calculations
# random_list = []
# i = 0
# count = 1000
# while i < count:
#     r = random.randint(0,len(rh_interpolated)-1)
#     if r not in random_list:
#         random_list.append(r)
#         i += 1
# profiles = random_list[:]
# profiles = {'profiles':profiles}
#
# handle = open(profiles_name + '.p','wb')
# pickle.dump(profiles, handle)
# handle.close()
#
# profiles = profiles['profiles']

################################################################################
#                           Noise calculations
################################################################################
n_profiles = 0
for profile in profiles:
    n_profiles += 1
    print(n_profiles)
    # Normalize data to predict from model
    predict_data = np.hstack((rh_interpolated[profile][:,1],
                              temp_interpolated[profile][:,1]))
    bounds = define_bounds([predict_data], n,
                            type = [[min_RH,max_RH],
                            [min_temp,max_temp]])[0]
    elevation = np.array([data['height'][profile][0]])
    # y = np.array([normalize_inputs(predict_data, n, bounds, elevation)])
    y = np.array([normalize_variable_bounds(predict_data, n, bounds, variable_bounds, variable_type, elevation)])
    y_pred = autoencoder.predict(y)
    # y_pred_normalized = normalize_inputs(y_pred[0][:-1], n, bounds,
    #                                      np.array(y_pred[0][-1]),
    #                                      inverse = True)
    y_pred_normalized = normalize_variable_bounds(y_pred[0][:-1], n, bounds, variable_bounds, variable_type,
                                         np.array(y_pred[0][-1]),
                                         inverse = True)

    print('Elevation:               %.2f' % elevation)
    print('Reconstructed elevation: %.2f' % y_pred_normalized[-1])

    # Generate hidden layer representation
    encoded_rep = encoder.predict(y)
    x = encoded_rep[0]
    print(x)

    # Structure data for sBoom
    rh_profile = rh_interpolated[profile,:,:]
    temp_profile = temp_interpolated[profile,:,:]
    p_rh = np.array([alt_interpolated, y_pred_normalized[:n]]).T
    p_temp = np.array([alt_interpolated, y_pred_normalized[n:-1]]).T
    sBoom_data = [list(temp_profile), 0, list(rh_profile)]
    sBoom_data_parametrized = [list(p_temp), 0, list(p_rh)]

    height_to_ground = (cruise_alt - data['height'][profile][0]) / 0.3048
    print('Height to ground: %0.2f m' % (height_to_ground*0.3048))
    # print('Ground level: %0.2f m' % data['height'][profile][0])
    nearfield_file = './../../../data/nearfield/25D_M16_RL5.p'

    # Noise calculations
    noise = {'original': 0, 'parametrized': 0, 'difference': 0}
    noise['original'] = boom_runner(sBoom_data, height_to_ground,
                                    nearfield_file=nearfield_file)

    noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground,
                                        nearfield_file=nearfield_file)
    noise['difference'] = noise['original'] - noise['parametrized']
    print(noise)

    height_to_ground_m = height_to_ground * 0.3048

    f.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (str(profile),
                                                      noise['original'],
                                                      noise['parametrized'],
                                                      noise['difference'],
                                                      x[0], x[1], x[2], x[3],
                                                      height_to_ground_m))
    # f.close()
    # print(x)
    # asdf

f.close()
