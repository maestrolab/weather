import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from autoencoder import *
from profile_bounds_NN import *
from weather.boom import boom_runner

# locations = ['72562','72214','72645','72582','72672']

# locations_path = 'balloon_data/2017+2018/US_2017_2018_' + '_'.join(locations) + '_only'

locations_path = 'balloon_data/2017+2018/US_2017_2018'

# Load profile data
data = pickle.load(open(locations_path + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Load model
n_params = 5
path = 'three_NN/trained_models/'
type = '%i_params_' % (n_params)
autoencoder_name = type + 'AE'
encoder_name = type + 'E'
autoencoder = tf.keras.models.load_model(path + autoencoder_name + '.h5')
encoder = tf.keras.models.load_model(path + encoder_name + '.h5')

# Load rh and temp bounds models
rh_encoder = tf.keras.models.load_model('learn_bounds/rh_encoder.h5')
temp_encoder = tf.keras.models.load_model('learn_bounds/temp_encoder.h5')

# Interpolate profiles
cruise_alt = 13500 # [m]
n = int((autoencoder.layers[0].input_shape[1])/2)
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

# Open file to write noise and parameters
# filepath = path + 'noise/' + type[:-2] + '_0' + '_trained_data'
# filepath = path + 'noise/' + type[:-1]
# filepath = 'rh_min_constant/' + 'noise/' + type + '1.25_min_model_not_trained'
# filepath = path + 'noise/' + type[:-1]
filepath = 'three_NN/noise/' + type[:-1]
f = open(filepath + '.txt','w')

# Randomly select profiles to compute the noise calculations
# random_list = []
# i = 0
# count = 100
# while i < count:
#     r = random.randint(0,len(rh)-1)
#     if r not in random_list:
#         random_list.append(r)
#         i += 1
# profiles = random_list[:]

# profiles_path = 'profiles_' + '_'.join(locations)
# g = open(profiles_path + '.p','wb')
# pickle.dump(profiles, g)
# g.close()

profiles = pickle.load(open('profiles.p','rb'))['profiles']
# profiles = pickle.load(open(profiles_path + '.p','rb'))

n_profiles = 0

for profile in profiles:
    n_profiles += 1
    print(n_profiles)
    print(profile)

    # Predict bounds for all profiles
    rh_y = np.array([normalize_profile(rh_interpolated[:,:,1][profile],
                    type='relative_humidities')])
    rh_pred = rh_encoder.predict(rh_y)
    predicted_bounds_rh = normalize_bounds(rh_pred, type='relative_humidities',
                                           inverse = True)
    print(predicted_bounds_rh.shape)
    temp_y = np.array([normalize_profile(temp_interpolated[:,:,1][profile],
                    type='temperatures')])
    temp_pred = temp_encoder.predict(temp_y)
    predicted_bounds_temp = normalize_bounds(temp_pred, type='temperatures',
                                           inverse = True)

    # Set and normalize training data
    rh_normalized = np.array([NN_normalize(rh_interpolated[:,:,1][profile],
                              predicted_bounds_rh[profile])])
    temp_normalized = np.array([NN_normalize(temp_interpolated[:,:,1][profile],
                              predicted_bounds_temp[profile])])

    y = np.hstack((rh_normalized[:],temp_normalized[:]))

    y_pred = autoencoder.predict(y)
    rh_normalized_ = np.array([NN_normalize(rh_interpolated[:,:,1][profile],
                              predicted_bounds_rh[profile], inverse=True)])
    temp_normalized_ = np.array([NN_normalize(temp_interpolated[:,:,1][profile],
                              predicted_bounds_temp[profile], inverse=True)])

    print('Elevation:               %.2f' % elevation)
    print('Reconstructed elevation: %.2f' % y_pred_normalized[-1])

    # Generate hidden layer representation
    encoded_rep = encoder.predict(y)
    x = encoded_rep[0]

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
