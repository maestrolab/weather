import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from autoencoder import *

profile = 6050

# Load profile data
data = pickle.load(open('./balloon_data/72469_2000-2018.p','rb'))
rh = np.array(data['humidity'])

# Load model
date = '7_12_19_'
decoder_name = date + 'D'
decoder = tf.keras.models.load_model('trained_models/' + decoder_name + '.h5')

# Interpolate profiles
n = decoder.layers[-1].output_shape[1]
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated = np.zeros((len(rh), len(alt_interpolated), 2))
for i in range(len(rh)):
    alts, values = np.array(rh[i]).T
    fun = interp1d(alts, values)
    rh_interp = fun(alt_interpolated)
    rh_interpolated[i] = np.array([alt_interpolated, rh_interp]).T

# Read encoded representations of profiles to decode
database = {}
path = 'noise_txt_files/'
noise_file = date[:-1]
f = open(path + noise_file + '.txt','r')
lines = f.readlines()
for line in lines:
    key = int(line.split()[0])
    database[key] = np.array([float(line.split()[i]) for i in range(-8,0)])
f.close()

# Reconstruct profiles using the decoder
predict_data = np.array(rh_interpolated[profile][:,1])
bounds = define_bounds([predict_data])[0]
y_pred = decoder.predict(np.array([database[profile]]))
y_pred_normalized = normalize_inputs(y_pred[0], bounds, inverse = True)

# Plot original and reconstructed profiles
plt.figure()
plt.plot(predict_data, alt_interpolated, label = 'original')
plt.plot(y_pred_normalized, alt_interpolated, label = 'predicted')
plt.xlabel('Relative Humidity [%]')
plt.ylabel('Altitude [m]')
plt.legend()
plt.show()
