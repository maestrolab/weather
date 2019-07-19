import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d

from autoencoder import *

# Load profile data
data = pickle.load(open('./balloon_data/72469_2000-2018.p','rb'))
rh = np.array(data['humidity'])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated = np.zeros((len(rh), len(alt_interpolated), 2))
for i in range(len(rh)):
    alt, values = np.array(rh[i]).T
    fun = interp1d(alt, values)
    rh_interp = fun(alt_interpolated)
    rh_interpolated[i] = np.array([alt_interpolated, rh_interp]).T

# Set and normalize training data
test_data = rh_interpolated[:,:,1][:]
bounds = define_bounds(test_data)
y = np.array([normalize_inputs(test_data[i],bounds[i]) for i in
             range(test_data.shape[0])])

# Build autoencoder
first_layer_dim = 3750
encoding_dim = 5
input = tf.keras.Input(shape=(test_data.shape[1],))
x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(input)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(test_data.shape[1], activation='sigmoid')(x)
autoencoder = tf.keras.Model(input,decoded)

# Build encoder and decoder
decoder_input = tf.keras.Input(shape=(encoding_dim,))
x = autoencoder.get_layer(index=-2)(decoder_input)
decoded = autoencoder.get_layer(index=-1)(x)

encoder = tf.keras.Model(input, encoded)
decoder = tf.keras.Model(decoder_input, decoded)

# Compile autoencoder, encoder, and decoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
decoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train autoencoder
epochs = 100
batch_size = 5
break_points = {1:6000}
autoencoder.fit(y[:break_points[1]], y[:break_points[1]],
                epochs=epochs,
                batch_size=batch_size)

# Save autoencoder, encoder, and decoder
path = 'trained_models/encoding_dim_and_layer_dim_varied/'
# date = '7_15_19_'
n_params = '%i_params_%i_dim_' % ((encoding_dim + 2), first_layer_dim)
# autoencoder_name = date + 'AE'
autoencoder_name = n_params + 'AE'
autoencoder.save(path + autoencoder_name + '.h5')

# encoder_name = date + 'E'
encoder_name = n_params + 'E'
encoder.save(path + encoder_name + '.h5')

# decoder_name = date + 'D'
decoder_name = n_params + 'D'
decoder.save(path + decoder_name + '.h5')
