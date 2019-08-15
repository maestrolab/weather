import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from autoencoder import *
from weather.boom import boom_runner

# locations = ['72786','72558','72261','72363','74646']

# locations_path = 'balloon_data/2017+2018/US_2017_2018_' + '_'.join(locations) + '_only'

locations_path = 'balloon_data/2017+2018/US_2017_2018'

# Load profile data
data = pickle.load(open(locations_path + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

# Load model
n_params = 9
first_layer_dim = 2500
# path = 'validation_data/'
# path = 'multi-year_vs_single_year/'
path = 'batch_normalization/'
model_path = path + 'trained_models/'
# type = '%i_params_' % (n_params) + '_'.join(locations) + '_'
type = '%i_params_' % n_params
autoencoder_name = type + 'AE'
encoder_name = type + 'E'
autoencoder = tf.keras.models.load_model(model_path + autoencoder_name + '.h5')
encoder = tf.keras.models.load_model(model_path + encoder_name + '.h5')

# Interpolate profiles
cruise_alt = 13500 # [m]
n = int((autoencoder.layers[0].input_shape[1]-1)/2)
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

# Open file to write noise and parameters
filepath = path + 'noise/' + type[:-1] + '_0' + '_trained_data'
# filepath = path + 'noise/' + type[:-1]
f = open(filepath + '.txt','w')

# Randomly select profiles to compute the noise calculations
# random_list = []
# i = 0
# count = 100
# while i < count:
#     r = random.randint(0,len(rh)-1)
#     if r not in random_list:
#         random_list.append(r)
#         i += 1
# profiles = random_list[:]

profiles = pickle.load(open('profiles.p','rb'))['profiles']

n_profiles = 0

for profile in profiles:
    n_profiles += 1
    print(n_profiles)
    print(profile)
    # Normalize data to predict from model
    predict_data = np.hstack((rh_interpolated[profile][:,1], temp_interpolated[profile][:,1]))
    bounds = define_bounds([predict_data], n, type = [['min','max'],['min','max']])[0]
    elevation = np.array([data['height'][profile][0]])
    y = np.array([normalize_inputs(predict_data, n, bounds, elevation)])
    y_pred = autoencoder.predict(y)
    y_pred_normalized = normalize_inputs(y_pred[0][:-1], n, bounds, np.array(y_pred[0][-1]), inverse = True)

    print('Elevation:               %.2f' % elevation)
    print('Reconstructed elevation: %.2f' % y_pred_normalized[-1])

    # Generate hidden layer representation
    encoded_rep = encoder.predict(y)
    x = encoded_rep[0]

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

    min_rh = np.min(predict_data[:n])
    min_temp = np.min(predict_data[n:-1])
    max_rh = np.max(predict_data[:n])
    max_temp = np.max(predict_data[n:-1])
    height_to_ground_m = height_to_ground * 0.3048

    f.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (str(profile),
                                                         noise['original'],
                                                         noise['parametrized'],
                                                         noise['difference'],
                                                         x[0], x[1], x[2], x[3], #x[4],
                                                         min_rh, max_rh,
                                                         min_temp, max_temp,
                                                         height_to_ground_m))
    # f.close()
    # print(x)
    # asdf

f.close()
