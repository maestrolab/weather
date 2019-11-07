import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d

from autoencoder import *

################################################################################
#                                 Parameters
################################################################################
# Balloon data
balloon_data  ='balloon_data/2017+2018/US_2017_2018'

# Normalizing (bounds specification)
min_RH = 'constant_min'
max_RH = 'max'
min_temp = 'min'
max_temp = 'max'
constant_type = 'variable_rh_temp_3_latent_0'

variable_bounds_handle = 'feature_bounds'
variable_type = 'both'

# Saving models
encoding_dim = 3
type = '%i_params_%s_' % ((encoding_dim + 1), constant_type)
path = 'trained_models/'

################################################################################
#                              Load profile data
################################################################################
data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

################################################################################
#                            Interpolate profiles
################################################################################
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

################################################################################
#                              Remove outliers
################################################################################
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated, temp_interpolated, elevations)

################################################################################
#                        Prepare data used for training
################################################################################
test_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))
bounds = define_bounds(test_data, n, type = [[min_RH,max_RH],
                                            [min_temp,max_temp]])

# print(len(test_data))
# test_data, bounds = bounds_check(test_data, bounds)
# print(len(test_data))

variable_bounds = pickle.load(open(variable_bounds_handle + '.p', 'rb'))

y = np.array([normalize_variable_bounds(test_data[i],n,bounds[i],variable_bounds,variable_type,elevations[i]) for i in
             range(test_data.shape[0])])

################################################################################
#                              Build autoencoder
################################################################################
intermediate_dim = [2500, 1250, 625, 300, 150, 75, 30, 15]
input_tensor = tf.keras.Input(shape=(y.shape[1],))
x = tf.keras.layers.Dense(intermediate_dim[0], activation='relu')(input_tensor)
x = tf.keras.layers.Dense(intermediate_dim[1], activation='relu')(x)
x = tf.keras.layers.Dense(intermediate_dim[2], activation='relu')(x)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
x = tf.keras.layers.Dense(intermediate_dim[2], activation='relu')(encoded)
x = tf.keras.layers.Dense(intermediate_dim[1], activation='relu')(x)
x = tf.keras.layers.Dense(intermediate_dim[0], activation='relu')(x)
decoded = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(x)
autoencoder = tf.keras.Model(input_tensor,decoded)

################################################################################
#                             Build encoder and decoder
################################################################################
decoder_input = tf.keras.Input(shape=(encoding_dim,))
x1 = autoencoder.get_layer(index=-4)(decoder_input)
x2 = autoencoder.get_layer(index=-3)(x1)
x3 = autoencoder.get_layer(index=-2)(x2)
decoded = autoencoder.get_layer(index=-1)(x3)

encoder = tf.keras.Model(input_tensor, encoded)
decoder = tf.keras.Model(decoder_input, decoded)

################################################################################
#                      Compile autoencoder, encoder, and decoder
################################################################################
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
decoder.compile(optimizer='adadelta', loss='binary_crossentropy')

################################################################################
#                               Train autoencoder
################################################################################
epochs = 10
batch_size = 500
autoencoder.fit(y, y,
                epochs=epochs,
                batch_size=batch_size)

################################################################################
#                     Save autoencoder, encoder, and decoder
################################################################################
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
