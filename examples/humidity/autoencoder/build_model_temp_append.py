import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d

from autoencoder import *

# Load profile data
# CHANGE_FILE_TO_INCLUDE_ALL_LOCATIONS
# data = pickle.load(open('./balloon_data/72469_2000-2018.p','rb'))
data = pickle.load(open('./balloon_data/2018/03953-72747.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

# Set and normalize training data
test_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))
bounds = define_bounds_temp_append(test_data, n, type = [['min','max'],
                                                         ['min','max']])
test_data, bounds = bounds_check(test_data, bounds)
y = np.array([normalize_inputs_temp_append(test_data[i],n,bounds[i]) for i in
             range(test_data.shape[0])])

# Build autoencoder
first_layer_dim = 2500
encoding_dim = 4
input_tensor = tf.keras.Input(shape=(test_data.shape[1],))
x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(input_tensor)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(test_data.shape[1], activation='sigmoid')(x)
autoencoder = tf.keras.Model(input_tensor,decoded)

# Build encoder and decoder
decoder_input = tf.keras.Input(shape=(encoding_dim,))
x = autoencoder.get_layer(index=-2)(decoder_input)
decoded = autoencoder.get_layer(index=-1)(x)

encoder = tf.keras.Model(input_tensor, encoded)
decoder = tf.keras.Model(decoder_input, decoded)

# Compile autoencoder, encoder, and decoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
decoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train autoencoder
epochs = 100
batch_size = 25
break_points = {0:6000, 1:len(y)-1}
autoencoder.fit(y[:break_points[1]], y[:break_points[1]],
                epochs=epochs,
                batch_size=batch_size)

# Save autoencoder, encoder, and decoder

# CHANGE_NAME

# path = 'trained_models/temp_append_encoding_dim_varied/'
# type = '%i_params_%i_dim_temp_append_constant_mins_' % ((encoding_dim + 3), first_layer_dim) # Plus 3 is for max of rh and temp profiles and ground elevation
path = 'trained_models/multiple_locations/min_max_used/'
type = 'locations_03953-72747_%i_params_' % (encoding_dim + 5)

print('Filepath: %s' % path)
print('Filename: %s' % (type + '(model type).h5'))
save = input('Save: (Y/N)\n')
if save == 'Y':
    change_name = input('Change name: (Y/N)\n')
    if change_name == 'N':
        autoencoder_name = type + 'AE'
        autoencoder.save(path + autoencoder_name + '.h5')

        encoder_name = type + 'E'
        encoder.save(path + encoder_name + '.h5')

        decoder_name = type + 'D'
        decoder.save(path + decoder_name + '.h5')
    elif change_name == 'Y':
        new_name = input('Enter new name: (include path)\n')
        autoencoder_name = new_name
        autoencoder.save(autoencoder_name + '.h5')

        encoder_name = new_name
        encoder.save(new_name + '.h5')

        decoder_name = new_name
        decoder.save(new_name + '.h5')
