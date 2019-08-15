import pickle
import numpy as np
import tensorflow as tf
import random

from autoencoder import *

# Load model
n_params = 9
path = 'multi-year_vs_single_year/'
model_path = path + 'trained_models/'
type = '%i_params_' % n_params
autoencoder_name = type + 'AE'
encoder_name = type + 'E'
decoder_name = type + 'D'
autoencoder = tf.keras.models.load_model(model_path + autoencoder_name + '.h5')
encoder = tf.keras.models.load_model(model_path + encoder_name + '.h5')
decoder = tf.keras.models.load_model(model_path + decoder_name + '.h5')

# Print layer dimensions
for layer in autoencoder.layers:
    print(layer.output_shape)

# Load profile data
locations_path = 'balloon_data/2017+2018/US_2017_2018'
data = pickle.load(open(locations_path + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

# Interpolate profiles
cruise_alt = 13500 # [m]
n = int((autoencoder.layers[0].input_shape[1]-1)/2)
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

# Read encoded representations of profiles to decode
n_params = 9
database = {}
path = 'multi-year_vs_single_year/noise/'
noise_file = '%i_params_0_trained_data' % n_params
f = open(path + noise_file + '.txt','r')
lines = f.readlines()
for line in lines:
    key = int(line.split()[0])
    database[key] = np.array([float(line.split()[i]) for i in range(-5,-1)])
f.close()

# Reconstruct profiles using the decoder
profile = random.randint(0,len(rh_interpolated))
predict_data = np.array(rh_interpolated[profile][:,1])
bounds = define_bounds([predict_data], n, type = [['min','max'],['min','max']])[0]
y_pred = decoder.predict(np.array([database[profile]]))
y_pred_normalized = normalize_inputs(y_pred[0], n, bounds[0], elevation, inverse = True)

# Plot original and reconstructed profiles
plt.figure()
plt.plot(predict_data, alt_interpolated, label = 'original')
plt.plot(y_pred_normalized, alt_interpolated, label = 'predicted')
plt.xlabel('Relative Humidity [%]')
plt.ylabel('Altitude [m]')
plt.legend()
plt.show()
