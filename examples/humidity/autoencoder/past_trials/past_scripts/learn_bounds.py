import pickle
import numpy as np
import tensorflow as tf

from autoencoder import *
from profile_bounds_NN import *

#===============================================================================
balloon_data  ='balloon_data/2017+2018/US_2017_2018'
epochs = 10
batch_size = 50

#===============================================================================

# Load balloon data
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
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated,
                                                temp_interpolated, elevations)

# Determine max and min values for the profiles
test_data = rh_interpolated[:,:,1][:]
rh_bounds = profile_bounds(test_data)
test_data, rh_bounds = bounds_check_profile(test_data, rh_bounds)
y_rh = np.array([normalize_profile(test_data[i], type='relative_humidities') for i in range(test_data.shape[0])])
y_bounds_rh = np.array([normalize_bounds(rh_bounds[i], type='relative_humidities') for i in range(rh_bounds.shape[0])])

test_data = temp_interpolated[:,:,1][:]
temp_bounds = profile_bounds(test_data)
test_data, temp_bounds = bounds_check_profile(test_data, temp_bounds)
y_temp = np.array([normalize_profile(test_data[i], type='temperatures') for i in range(test_data.shape[0])])
y_bounds_temp = np.array([normalize_bounds(temp_bounds[i], type='temperatures') for i in range(temp_bounds.shape[0])])

# Build model
layer_dim = np.array([150, 75, 35, 15, 2])
input_tensor = tf.keras.Input(shape=(y_rh.shape[1],))
x = tf.keras.layers.Dense(layer_dim[0], activation='relu')(input_tensor)
x = tf.keras.layers.Dense(layer_dim[1], activation='relu')(x)
x = tf.keras.layers.Dense(layer_dim[2], activation='relu')(x)
x = tf.keras.layers.Dense(layer_dim[3], activation='relu')(x)
encoded = tf.keras.layers.Dense(layer_dim[4], activation='sigmoid')(x)
rh_encoder = tf.keras.Model(input_tensor, encoded)

input_tensor = tf.keras.Input(shape=(y_rh.shape[1],))
x = tf.keras.layers.Dense(layer_dim[0], activation='relu')(input_tensor)
x = tf.keras.layers.Dense(layer_dim[1], activation='relu')(x)
x = tf.keras.layers.Dense(layer_dim[2], activation='relu')(x)
x = tf.keras.layers.Dense(layer_dim[3], activation='relu')(x)
encoded = tf.keras.layers.Dense(layer_dim[4], activation='sigmoid')(x)
temp_encoder = tf.keras.Model(input_tensor, encoded)

# Train model
rh_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
rh_encoder.fit(y_rh,y_bounds_rh,
               epochs = epochs,
               batch_size = batch_size)

temp_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
temp_encoder.fit(y_temp,y_bounds_temp,
                 epochs = epochs,
                 batch_size = batch_size)

# Test model
j = 3000
test_data = rh_interpolated[:,:,1][j]
y = np.array([normalize_profile(test_data[i], type='relative_humidities') for i in range(test_data.shape[0])])
y_pred = rh_encoder.predict(np.array([y]))
predicted_bounds_rh = normalize_bounds(y_pred, type='relative_humidities', inverse = True)

test_data = temp_interpolated[:,:,1][j]
y = np.array([normalize_profile(test_data[i], type='temperatures') for i in range(test_data.shape[0])])
y_pred = temp_encoder.predict(np.array([y]))
predicted_bounds_temp = normalize_bounds(y_pred, type='temperatures', inverse = True)

print('Relative Humidity:')
print(rh_bounds[j])
print(predicted_bounds_rh)

print('Temperature:')
print(temp_bounds[j])
print(predicted_bounds_temp)

# Save model
rh_encoder_name = 'learn_bounds/rh_encoder'
rh_encoder.save(rh_encoder_name + '.h5')

temp_encoder_name = 'learn_bounds/temp_encoder'
temp_encoder.save(temp_encoder_name + '.h5')
