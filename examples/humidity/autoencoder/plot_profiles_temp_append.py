import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from autoencoder import *

# Load profile data
data = pickle.load(open('./balloon_data/2018/72469_2000-2018.p','rb'))
# data = pickle.load(open('./balloon_data/all_locations_2018.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

# Load model
n_params = 7
first_layer_dim = 2500
# path = 'trained_models/temp_append_encoding_dim_varied/'
path = 'trained_models/encoding_dim_and_layer_dim_varied/'
type = '%i_params_%i_dim_' % (n_params, first_layer_dim)
# type = '%i_params_%i_dim_temp_append_constant_mins_' % (n_params, first_layer_dim)
# path = 'trained_models/multiple_locations/min_max_used/'
# type = 'locations_03953-72747_%i_params_' % n_params
autoencoder_name = type + 'AE'
autoencoder = tf.keras.models.load_model(path + autoencoder_name + '.h5')

# Interpolate profiles
n = int(autoencoder.layers[0].input_shape[1]/2)
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh,
                                                          temp)

# Normalize data to predict from model
profile = 100
predict_data = np.hstack((rh_interpolated[profile][:,1], temp_interpolated[profile][:,1]))
bounds = define_bounds_temp_append([predict_data], n, type = [['min','max'],
                                                              ['min','max']])[0]
y = np.array([normalize_inputs_temp_append(predict_data, n, bounds)])
y_pred = autoencoder.predict(y)
y_pred_normalized = normalize_inputs_temp_append(y_pred[0], n, bounds, inverse = True)

# Plot original and reconstructed profiles
plt.figure()
plt.plot(predict_data[:n], alt_interpolated, label = 'original')
plt.plot(y_pred_normalized[:n], alt_interpolated, label = 'predicted')
plt.xlabel('Relative Humidity [%]')
plt.ylabel('Altitude [m]')
plt.legend()

plt.figure()
plt.plot(predict_data[n:], alt_interpolated, label = 'original')
plt.plot(y_pred_normalized[n:], alt_interpolated, label = 'predicted')
plt.xlabel('Temperature [F]')
plt.ylabel('Altitude [m]')
plt.legend()

plt.show()
