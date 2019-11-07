import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from autoencoder import *

balloon_data = 'balloon_data/2017+2018/US_2017_2018'

# Load profile data
data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array(data['height'])
elevations = np.array([n[0] for n in elevations])

# Load model
n_params = 5
first_layer_dim = 2500
path = 'constant_min_max/trained_models/'
type = '%i_params_' % (n_params)
autoencoder_name = type + 'AE'
autoencoder = tf.keras.models.load_model(path + autoencoder_name + '.h5')

# Interpolate profiles
n = int((autoencoder.layers[0].input_shape[1]-1)/2)
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh,
                                                          temp)

# Normalize data to predict from model
profile = random.randint(0,len(rh_interpolated)-1)
print(profile)
predict_data = np.hstack((rh_interpolated[profile][:,1], temp_interpolated[profile][:,1]))
bounds =  define_bounds([predict_data], n, type = [['constant_min','constant_max'],
                                                   ['constant_min','constant_max']])[0]
elevation = np.array([data['height'][profile][0]])
y = np.array([normalize_inputs(predict_data, n, bounds, elevation)])
y_pred = autoencoder.predict(y)
y_pred_normalized = normalize_inputs(y_pred[0][:-1], n, bounds, np.array(y_pred[0][-1]), inverse = True)

# Plot original and reconstructed profiles
plt.figure()
plt.plot(predict_data[:n], alt_interpolated, label = 'original')
plt.plot(y_pred_normalized[:n], alt_interpolated, label = 'predicted', color = 'orange')
plt.xlabel('Relative Humidity [%]')
plt.ylabel('Altitude [m]')
plt.legend()

# plt.figure()
# plt.plot(predict_data[n:], alt_interpolated, label = 'original')
# plt.plot(y_pred_normalized[n:], alt_interpolated, label = 'predicted')
# plt.xlabel('Temperature [F]')
# plt.ylabel('Altitude [m]')
# plt.legend()

plt.show()
