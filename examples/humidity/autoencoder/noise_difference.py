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
date = '7_12_19_'
autoencoder_name = date + 'AE'
autoencoder = tf.keras.models.load_model('trained_models/' + autoencoder_name + '.h5')

# Interpolate profiles
n = autoencoder.layers[0].input_shape[1]
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

# Normalize data to predict from model
profile = 6100
predict_data = np.array(rh_interpolated[profile][:,1])
bounds = define_bounds([predict_data])[0]
y = np.array([normalize_inputs(predict_data, bounds)])
y_pred = autoencoder.predict(y)
y_pred_normalized = normalize_inputs(y_pred[0], bounds, inverse = True)

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
