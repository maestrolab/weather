import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from autoencoder import *
from weather.boom import boom_runner

# Load profile data
data = pickle.load(open('./balloon_data/72469_2000-2018.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

# Load model
# path = 'trained_models/'
# date = '7_12_19_'
# autoencoder_name = date + 'AE'
path = 'trained_models/encoding_dim_and_layer_dim_varied/'
n_params = 7
dim = 3750
definition = '%i_params_%i_dim_' % (n_params, dim)
autoencoder_name = definition + 'AE'
encoder_name = definition + 'E'
autoencoder = tf.keras.models.load_model(path + autoencoder_name + '.h5')
# encoder_name = date + 'E'
encoder = tf.keras.models.load_model(path + encoder_name + '.h5')

# Interpolate profiles
n = autoencoder.layers[0].input_shape[1]
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

# filepath = 'noise_txt_files/' + date[:-1]
filepath = 'noise_txt_files/encoding_dim_and_layer_dim_varied/' + definition[:-1]
f = open(filepath + '.txt','w')
for profile in range(6001,len(rh_interpolated)):
    print(profile)
    # Normalize data to predict from model
    predict_data = np.array(rh_interpolated[profile][:,1])
    bounds = define_bounds([predict_data])[0]
    y = np.array([normalize_inputs(predict_data, bounds)])
    y_pred = autoencoder.predict(y)
    y_pred_normalized = normalize_inputs(y_pred[0], bounds, inverse = True)

    # Generate hidden layer representation
    encoded_rep = encoder.predict(y)
    x = encoded_rep[0]

    # Structure data for sBoom
    rh_profile = rh_interpolated[profile,:,:]
    temp_profile = temp_interpolated[profile,:,:]
    p_rh = np.array([alt_interpolated, y_pred_normalized]).T
    sBoom_data = [list(temp_profile), 0, list(rh_profile)]
    sBoom_data_parametrized = [list(temp_profile), 0, list(p_rh)]

    height_to_ground = 13500 / 0.3048
    nearfield_file = './../../../data/nearfield/25D_M16_RL5.p'

    # Noise calculations
    noise = {'original': 0, 'parametrized': 0, 'difference': 0}
    noise['original'] = boom_runner(sBoom_data, height_to_ground,
                                    nearfield_file=nearfield_file)
    noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground,
                                        nearfield_file=nearfield_file)
    noise['difference'] = noise['original'] - noise['parametrized']
    print(noise)

    if profile == 6001:
        print('Should have indices 0 through 4')
        for i in range(len(x)):
            print(i)

    f.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (str(profile),
                                                            noise['original'],
                                                            noise['parametrized'],
                                                            noise['difference'],
                                                            x[0], x[1], x[2], x[3],
                                                            x[4]))#, x[5], x[6], x[7]))

f.close()
