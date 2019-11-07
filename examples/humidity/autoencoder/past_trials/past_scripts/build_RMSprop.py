import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d

from autoencoder import *

locations_not_included = ['72562','72214','72645','72582','72672']

# Load profile data
balloon_data = 'balloon_data/2017+2018/US_2017_2018_' + '_'.join(locations_not_included)
# balloon_data  ='balloon_data/2017+2018/US_2017_2018'

data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

# Remove outliers from dataset
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated, temp_interpolated, elevations)

# Set and normalize training data
test_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))
y = test_data[:]

# Build autoencoder
first_layer_dim = 2500
second_layer_dim = 1250
third_layer_dim = 650
fourth_layer_dim = 125
encoding_dim = 4
input_tensor = tf.keras.Input(shape=(y.shape[1],))
x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(input_tensor)
x = tf.keras.layers.Dense(second_layer_dim, activation='relu')(x)
x = tf.keras.layers.Dense(third_layer_dim, activation='relu')(x)
x = tf.keras.layers.Dense(fourth_layer_dim, activation='relu')(x)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
x = tf.keras.layers.Dense(fourth_layer_dim, activation='relu')(encoded)
x = tf.keras.layers.Dense(third_layer_dim, activation='relu')(x)
x = tf.keras.layers.Dense(second_layer_dim, activation='relu')(x)
x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(x)
decoded = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(x)
autoencoder = tf.keras.Model(input_tensor,decoded)

# Build encoder and decoder
decoder_input = tf.keras.Input(shape=(encoding_dim,))
x1 = autoencoder.get_layer(index=-5)(decoder_input)
x2 = autoencoder.get_layer(index=-4)(x1)
x3 = autoencoder.get_layer(index=-3)(x2)
x4 = autoencoder.get_layer(index=-2)(x3)
decoded = autoencoder.get_layer(index=-1)(x4)

encoder = tf.keras.Model(input_tensor, encoded)
decoder = tf.keras.Model(decoder_input, decoded)

# Compile autoencoder, encoder, and decoder
optimizer = 'RMSprop'
loss = 'binary_crossentropy'
autoencoder.compile(optimizer=optimizer, loss=loss)
encoder.compile(optimizer=optimizer, loss=loss)
decoder.compile(optimizer=optimizer, loss=loss)

# Train autoencoder
epochs = 1
batch_size = 5000
autoencoder.fit(y, y,
                epochs=epochs,
                batch_size=batch_size)

# Save autoencoder, encoder, and decoder
type = '%i_params_' % ((encoding_dim + 1))
path = 'RMSprop/trained_models/'

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
