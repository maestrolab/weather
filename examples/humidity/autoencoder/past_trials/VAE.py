import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras import backend as K
from autoencoder import *
from weather.boom import boom_runner

################################################################################
#                           Load profile data
################################################################################
balloon_data  ='balloon_data/2017+2018/US_2017_2018'

data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

################################################################################
#                           Interpolate profiles
################################################################################
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

################################################################################
#                       Remove outliers from dataset
################################################################################
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated, temp_interpolated, elevations)
test_data = np.hstack((rh_interpolated[:,:,1][:], temp_interpolated[:,:,1][:]))

################################################################################
#                               Normalize data
################################################################################
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
intermediate_dim = [300, 150, 75]
latent_dim = 4
batch_size = 50
epochs = 10

################################################################################
# Define encoder model
inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
x = tf.keras.layers.Dense(intermediate_dim[0], activation='relu')(inputs)
x = tf.keras.layers.Dense(intermediate_dim[1], activation='relu')(x)
x = tf.keras.layers.Dense(intermediate_dim[2], activation='relu')(x)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean', kernel_initializer='ones')(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var', kernel_initializer='zeros')(x)

z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = tf.keras.Model(inputs, [z_mean,z_log_var], name='encoder')
################################################################################
# Define decoder
latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
x = tf.keras.layers.Dense(intermediate_dim[2], activation='relu')(latent_inputs)
x = tf.keras.layers.Dense(intermediate_dim[1], activation='relu')(latent_inputs)
x = tf.keras.layers.Dense(intermediate_dim[0], activation='relu')(latent_inputs)
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
# Open file to write noise and parameters
filepath = 'noise/' + 'VAE_%i_params_%i_%i_%i' % (latent_dim+1, intermediate_dim[0],
                                                                intermediate_dim[1],
                                                                intermediate_dim[2])
f = open(filepath + '.txt','w')

profiles = pickle.load(open('profiles.p','rb'))['profiles']

n_profiles = 0
cruise_alt = 13500
for profile in profiles:
    n_profiles += 1
    print(n_profiles)
    print(profile)
    # Normalize data to predict from model
    predict_data = np.hstack((rh_interpolated[profile][:,1],
                              temp_interpolated[profile][:,1]))
    bounds = define_bounds([predict_data], n,
                            type = [['constant_min','constant_max'],
                            ['constant_min','constant_max']])[0]
    elevation = np.array([data['height'][profile][0]])
    y = np.array([normalize_inputs(predict_data, n, bounds, elevation)])
    y_pred = vae.predict(y)
    y_pred_normalized = normalize_inputs(y_pred[0][:-1], n, bounds,
                                         np.array(y_pred[0][-1]),
                                         inverse = True)

    print('Elevation:               %.2f' % elevation)
    print('Reconstructed elevation: %.2f' % y_pred_normalized[-1])

    # Generate hidden layer representation
    encoded_rep = encoder.predict(y)
    x = encoded_rep[0][0]
    print(x)

    # Structure data for sBoom
    rh_profile = rh_interpolated[profile,:,:]
    temp_profile = temp_interpolated[profile,:,:]
    p_rh = np.array([alt_interpolated, y_pred_normalized[:n]]).T
    p_temp = np.array([alt_interpolated, y_pred_normalized[n:-1]]).T
    sBoom_data = [list(temp_profile), 0, list(rh_profile)]
    sBoom_data_parametrized = [list(p_temp), 0, list(p_rh)]

    height_to_ground = (cruise_alt - data['height'][profile][0]) / 0.3048
    print('Height to ground: %0.2f m' % (height_to_ground*0.3048))
    # print('Ground level: %0.2f m' % data['height'][profile][0])
    nearfield_file = './../../../data/nearfield/25D_M16_RL5.p'

    # Noise calculations
    noise = {'original': 0, 'parametrized': 0, 'difference': 0}
    noise['original'] = boom_runner(sBoom_data, height_to_ground,
                                    nearfield_file=nearfield_file)

    noise['parametrized'] = boom_runner(sBoom_data_parametrized, height_to_ground,
                                        nearfield_file=nearfield_file)
    noise['difference'] = noise['original'] - noise['parametrized']
    print(noise)

    min_rh = np.min(predict_data[:n])
    min_temp = np.min(predict_data[n:-1])
    max_rh = np.max(predict_data[:n])
    max_temp = np.max(predict_data[n:-1])
    height_to_ground_m = height_to_ground * 0.3048

    f.write('%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (str(profile),
                                                      noise['original'],
                                                      noise['parametrized'],
                                                      noise['difference'],
                                                      x[0], x[1], x[2], x[3],
                                                      height_to_ground_m))
    # f.close()
    # print(x)
    # asdf

f.close()
