import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from autoencoder import *
from weather.boom import boom_runner

# Load profile data
data = pickle.load(open('./balloon_data/72469_2000-2018.p','rb'))
# data = pickle.load(open('./balloon_data/2018/03953-70350.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

# Load model
n_params = 9
first_layer_dim = 2500
# path = 'trained_models/temp_append_encoding_dim_varied/'
# type = '%i_params_%i_dim_temp_append_constant_mins_' % (n_params, first_layer_dim)
path = 'trained_models/multiple_locations/min_max_used/'
type = 'locations_03953-72747_%i_params_' % n_params
autoencoder_name = type + 'AE'
encoder_name = type + 'E'
autoencoder = tf.keras.models.load_model(path + autoencoder_name + '.h5')
encoder = tf.keras.models.load_model(path + encoder_name + '.h5')

# Interpolate profiles
cruise_alt = 13500 # [m]
n = int(autoencoder.layers[0].input_shape[1]/2)
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

# CHANGE_NAME
###
# filename = '72469_0-150'
###

# filepath = 'noise_txt_files/' + date[:-1]
# filepath = 'noise_txt_files/multiple_locations/' + filename #temp_append_encoding_dim_varied/different_locations/' + filename
filepath = 'noise_txt_files/multiple_locations/min_max_used/' + type[:-1]
f = open(filepath + '.txt','w')
for profile in range(0,100):#len(rh_interpolated)):
    print(profile)
    # Normalize data to predict from model
    predict_data = np.hstack((rh_interpolated[profile][:,1], temp_interpolated[profile][:,1]))
    bounds = define_bounds_temp_append([predict_data], n, type = [['constant_min','max'],
                                                                  ['constant_min','max']])[0]
    y = np.array([normalize_inputs_temp_append(predict_data, n, bounds)])
    y_pred = autoencoder.predict(y)
    y_pred_normalized = normalize_inputs_temp_append(y_pred[0], n, bounds, inverse = True)

    # Generate hidden layer representation
    encoded_rep = encoder.predict(y)
    x = encoded_rep[0]

    # Structure data for sBoom
    rh_profile = rh_interpolated[profile,:,:]
    temp_profile = temp_interpolated[profile,:,:]
    p_rh = np.array([alt_interpolated, y_pred_normalized[:n]]).T
    p_temp = np.array([alt_interpolated, y_pred_normalized[n:]]).T
    sBoom_data = [list(temp_profile), 0, list(rh_profile)]
    sBoom_data_parametrized = [list(p_temp), 0, list(p_rh)]

    height_to_ground = (cruise_alt - data['height'][profile][0]) / 0.3048
    print('Height to ground: %0.2f m' % (height_to_ground*0.3048))
    print('Ground level: %0.2f m' % data['height'][profile][0])
    nearfield_file = './../../../data/nearfield/25D_M16_RL5.p'

    # Noise calculations
    noise = {'original': 0, 'parametrized': 0, 'difference': 0}
    noise['original'] = boom_runner(sBoom_data, height_to_ground,
                                    nearfield_file=nearfield_file)
    noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground,
                                        nearfield_file=nearfield_file)
    noise['difference'] = noise['original'] - noise['parametrized']
    print(noise)

    if profile == 0:
        print('Should have indices 0 through 3')
        for i in range(len(x)):
            print(i)

    min_rh = np.min(predict_data[:n])
    min_temp = np.min(predict_data[n:])
    max_rh = np.max(predict_data[:n])
    max_temp = np.max(predict_data[n:])
    height_to_ground_m = height_to_ground * 0.3048

    f.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (str(profile),
                                                         noise['original'],
                                                         noise['parametrized'],
                                                         noise['difference'],
                                                         x[0], x[1], x[2], x[3],
                                                         min_rh, max_rh,
                                                         min_temp, max_temp,
                                                         height_to_ground_m))

f.close()
