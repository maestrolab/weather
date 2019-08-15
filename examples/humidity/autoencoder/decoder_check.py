import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from autoencoder import *
from weather.boom import boom_runner

# Load profile data
balloon_data = 'balloon_data/2017+2018/US_2017_2018'
data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array(data['height'])
elevations = np.array([n[0] for n in elevations])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

# Load encoder and decoder
n_params = 9
model_path = 'multi-year_vs_single_year/trained_models/'
encoder_name = '%i_params_E' % n_params
encoder = tf.keras.models.load_model(model_path + encoder_name + '.h5')
decoder_name = '%i_params_D' % n_params
decoder = tf.keras.models.load_model(model_path + decoder_name + '.h5')

# Prepare data for encoder prediction
predict_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))
bounds = define_bounds(predict_data, n, type = [['min','max'],['min','max']])
print(len(predict_data))
predict_data, bounds, index = bounds_check(predict_data, bounds, index = True)
print(len(predict_data))
y = np.array([normalize_inputs(predict_data[i],n,bounds[i],elevations[i]) for i in
             range(predict_data.shape[0])])

# Compute encoded representation
encoded_rep = encoder.predict(y)
x = encoded_rep

l3_not_0 = np.where(x[:,3] != 0)

# Reconstruct profiles using decoder only
decoded_rep = decoder.predict(x[l3_not_0])
profiles = np.array([i for i in range(len(decoded_rep))])
for profile in profiles:
    predicted = normalize_inputs(decoded_rep[profile][:-1], n, bounds[l3_not_0][profile],
                                 np.array(decoded_rep[profile][-1]), inverse = True)

    # Set latent variable 3 to be 0
    not_0 = x[l3_not_0]
    not_0[:,3] = np.zeros(not_0.shape[0],)
    decoded_0 = decoder.predict(not_0)
    predicted_0 = normalize_inputs(decoded_0[profile][:-1], n, bounds[l3_not_0][profile],
                                 np.array(decoded_0[profile][-1]), inverse = True)

    # Plot profiles
    # plt.figure()
    # plt.plot(rh_interpolated[l3_not_0][profile][:,1], alt_interpolated, label = 'original')
    # plt.plot(predicted[:n], alt_interpolated, '-o', label = 'reconstructed')
    # plt.plot(predicted_0[:n], alt_interpolated, label = 'reconstructed_0')
    # plt.xlabel('Relative Humidity [%]')
    # plt.ylabel('Altitude [m]')
    # plt.legend()
    # plt.show()

    # Structure data for sBoom
    rh_profile = rh_interpolated[l3_not_0][profile,:,:]
    temp_profile = temp_interpolated[l3_not_0][profile,:,:]
    p_rh = np.array([alt_interpolated, predicted[:n]]).T
    p_temp = np.array([alt_interpolated, predicted[n:-1]]).T
    p_rh_0 = np.array([alt_interpolated, predicted_0[:n]]).T
    p_temp_0 = np.array([alt_interpolated, predicted_0[n:-1]]).T
    sBoom_data = [list(temp_profile), 0, list(rh_profile)]
    sBoom_data_parametrized = [list(p_temp), 0, list(p_rh)]
    sBoom_data_parametrized_0 = [list(p_temp_0), 0, list(p_rh_0)]

    height_to_ground = (13500 - data['height'][profile][0]) / 0.3048
    print('Height to ground: %0.2f m' % (height_to_ground*0.3048))
    # print('Ground level: %0.2f m' % data['height'][profile][0])
    nearfield_file = './../../../data/nearfield/25D_M16_RL5.p'

    # Noise calculations
    noise = {'original': 0, 'parametrized': 0, 'p_0': 0, 'difference': 0,
             'd_0': 0, 'p_d': 0}
    noise['original'] = boom_runner(sBoom_data, height_to_ground,
                                    nearfield_file=nearfield_file)

    noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground,
                                        nearfield_file=nearfield_file)

    noise['p_0'] = boom_runner(sBoom_data_parametrized_0, height_to_ground,
                                        nearfield_file=nearfield_file)

    noise['difference'] = noise['original'] - noise['parametrized']
    noise['d_0'] = noise['original'] - noise['p_0']

    noise['p_d'] = noise['parametrized'] - noise['p_0']
    print(noise)
