import pickle
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from weather.parameterize_atmosphere.autoencoder import *

################################################################################
#                       Load and select desired profile
################################################################################
index = 0 # choose which profile to plot
profile_path = '../../data/atmosphere_models/test_profiles/profiles_1000_0.p'
profiles = pickle.load(open(profile_path,'rb'))['profiles']
profile = profiles[index]

################################################################################
#                               Load models
################################################################################
n_params = 3

path = '../../data/atmosphere_models/trained_models/%i_parameters' % n_params
encoder = tf.keras.models.load_model(path + '_E.h5')
decoder = tf.keras.models.load_model(path + '_D.h5')

################################################################################
#                           Load profile data
################################################################################
balloon_data  = 'G:/Shared drives/Maestro Team Drive/Misc File Sharing/' +\
                 'Atmospheric Profiles Machine Learning/balloon_data/2017+2018/'+\
                 'US_2017_2018.p'

data = pickle.load(open(balloon_data,'rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

feature_bounds = '../../data/atmosphere_models/feature_bounds.p'
variable_bounds = pickle.load(open(feature_bounds, 'rb'))

# Interpolate profiles
cruise_alt = 13500
n = 75
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh,
                                                          temp)

# Remove outliers from dataset
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated,
                                                          temp_interpolated,
                                                          elevations)

################################################################################
#                           Reconsruct profile
################################################################################
predict_data = np.hstack((rh_interpolated[profile][:,1],
                          temp_interpolated[profile][:,1]))
elevation = np.array([data['height'][profile][0]])
y = np.array([normalize_variable_bounds(predict_data, n, variable_bounds, elevation)])
latent_rep = encoder.predict(y)
y_pred = decoder.predict(latent_rep)
y_pred_normalized = normalize_variable_bounds(y_pred[0][:-1], n, variable_bounds,
                                     np.array(y_pred[0][-1]), inverse = True)

################################################################################
#                           Plot profiles
################################################################################
fig = plt.figure()
plt.plot(temp_interpolated[profile,:,1], alt_interpolated)
plt.plot(y_pred_normalized[n:-1], alt_interpolated)
plt.xlabel('Temperature (\N{DEGREE SIGN}F)')
plt.ylabel('Altitude (m)')
plt.xlim([-110,100])

fig = plt.figure()
plt.plot(rh_interpolated[profile,:,1], alt_interpolated)
plt.plot(y_pred_normalized[:n], alt_interpolated)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Altitude (m)')

plt.show()
