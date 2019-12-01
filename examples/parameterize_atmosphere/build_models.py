import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d

from weather.parameterize_atmosphere.autoencoder import *

################################################################################
#                                 Parameters
################################################################################
# Balloon data
balloon_data  = 'G:/Shared drives/Maestro Team Drive/Misc File Sharing/' +\
                 'Atmospheric Profiles Machine Learning/balloon_data/2017+2018/'+\
                 'US_2017_2018.p'

# Interpolating parameters
cruise_alt = 13500 # meters
n = 75 # number of data points in interpolated profiles

# Saving models (number of parameters = encoding_dim + 1 for elevation at ground)
encoding_dim = 6
path = '../../data/atmosphere_models/trained_models/%i_parameters' % (encoding_dim+1)

confirmed = input('Please confirm path:\n\n%s\n\n(True/False): ' % path)

if not confirmed:
    raise RuntimeError('Model save path entered incorrectly.')

# Variable bounds
feature_bounds = '../../data/atmosphere_models/feature_bounds.p'

################################################################################
#                              Load profile data
################################################################################
data = pickle.load(open(balloon_data,'rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

variable_bounds = pickle.load(open(feature_bounds, 'rb'))
################################################################################
#                          Prepare training data
################################################################################
alt_interpolated = np.linspace(0, cruise_alt, n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated,
                                                          temp_interpolated,
                                                          elevations)

################################################################################
#                        Prepare data used for training
################################################################################
test_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))
y = np.array([normalize_variable_bounds(test_data[i],n,
              variable_bounds,elevations[i]) for i in
              range(test_data.shape[0])])

################################################################################
#                              Build autoencoder
################################################################################
intermediate_dim = [2500, 1250, 625] #, 300, 150, 75, 30, 15]
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
#                     Save encoder and decoder
################################################################################
encoder.save(path + '_E.h5')
decoder.save(path + '_D.h5')
