import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d

from autoencoder import *

################################################################################
# Lines that are changed depending on using all data or using data w/ 5 locations
#   removed from the training set.

# locations_not_included = ['72786','72558','72261','72363','74646']
# balloon_data = 'balloon_data/2017+2018/US_2017_2018_' + '_'.join(locations_not_included)
balloon_data  ='balloon_data/2017+2018/US_2017_2018'

epochs = 10
batch_size = 5
encoding_dim = 4
type = '%i_params_' % ((encoding_dim + 5))
# type = '%i_params_' % (encoding_dim + 5) + '_'.join(locations_not_included) + '_'

################################################################################
# Load profile data
data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

# Remove outliers
print(len(rh_interpolated))
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated,
                                                  temp_interpolated, elevations)
print(len(rh_interpolated))

# Set and normalize training data
test_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))
bounds = define_bounds(test_data, n, type = [['min','max'],['min','max']])
print(len(test_data))
test_data, bounds = bounds_check(test_data, bounds)
print(len(test_data))
y = np.array([normalize_inputs(test_data[i],n,bounds[i],elevations[i]) for i in
             range(test_data.shape[0])])

# Build autoencoder
first_layer_dim = 2500
input_tensor = tf.keras.Input(shape=(y.shape[1],))
x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(input_tensor)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(x)
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
autoencoder.fit(y, y,
                epochs=epochs,
                batch_size=batch_size)

# Save autoencoder, encoder, and decoder
path = 'outliers_removed/trained_models/'

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
