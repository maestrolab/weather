import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from scipy.interpolate import interp1d

from autoencoder import *

################################################################################
# Load profile data
# locations_not_included = ['72562','72214','72645','72582','72672']
# balloon_data = 'balloon_data/2017+2018/US_2017_2018_' + '_'.join(locations_not_included)
balloon_data  ='balloon_data/2017+2018/US_2017_2018'

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

################################################################################
# Organize data into the input shape of the VAE
test_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))
# y = test_data[:]

################################################################################
# Normalize data
bounds = define_bounds(test_data, n, type = [['constant_min','constant_max'],
                                            ['constant_min','constant_max']])
print(len(test_data))
test_data, bounds = bounds_check(test_data, bounds)
print(len(test_data))
y = np.array([normalize_inputs(test_data[i],n,bounds[i],elevations[i]) for i in
             range(test_data.shape[0])])
################################################################################
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

################################################################################
# Model design variables
input_shape = (y.shape[1],)
intermediate_dim = 300
latent_dim = 4
batch_size = 50
epochs = 10

################################################################################
# Define encoder model
inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
x = tf.keras.layers.Dense(intermediate_dim, activation='relu', kernel_initializer='random_normal')(inputs)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean', kernel_initializer='ones')(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var', kernel_initializer='zeros')(x)

z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = tf.keras.Model(inputs, [z_mean,z_log_var], name='encoder')
################################################################################
# Define decoder
latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
x = tf.keras.layers.Dense(intermediate_dim, activation='relu', kernel_initializer='random_normal')(latent_inputs)
outputs = tf.keras.layers.Dense(input_shape[0], activation='sigmoid')(x)

decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
################################################################################
# Define VAE model
print(decoder(encoder(inputs)[0]))
outputs = decoder(encoder(inputs))
vae = tf.keras.Model(inputs, outputs, name='vae_mlp')

################################################################################
# Train VAE model
def vae_loss(x, x_decoded_mean):
    xent_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
    k1_loss = -0.5*K.mean(1+z_log_var-K.square(z_mean)-K.exp(z_log_var), axis=-1)
    return xent_loss + k1_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.fit(y,y,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

################################################################################
# Save autoencoder, encoder, and decoder
type = '%i_params_' % ((latent_dim + 1))
path = 'VAE/trained_models/'
# type = '%i_params_' % (encoding_dim + 4) + '_'.join(locations_not_included) + '_'

print('Filepath: %s' % path)
print('Filename: %s' % (type + '(model type).h5'))
save = input('Save: (Y/N)\n')
if save == 'Y':
    change_name = input('Change name: (Y/N)\n')
    if change_name == 'N':
        autoencoder_name = type + 'AE'
        vae.save(path + autoencoder_name + '.h5')

        encoder_name = type + 'E'
        encoder.save(path + encoder_name + '.h5')

        decoder_name = type + 'D'
        decoder.save(path + decoder_name + '.h5')
    elif change_name == 'Y':
        new_name = input('Enter new name: (include path)\n')
        autoencoder_name = new_name
        vae.save(autoencoder_name + '.h5')

        encoder_name = new_name
        encoder.save(new_name + '.h5')

        decoder_name = new_name
        decoder.save(new_name + '.h5')
