import pickle
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from autoencoder import *

################################################################################
#                               Load model
################################################################################
n_params = 5
constant_type = 'variable_rh_temp'
variable_type = 'both' # overwrites the min and max choices above
path = 'trained_models/%i_params_%s_' % (n_params, constant_type)

autoencoder = tf.keras.models.load_model(path + 'AE.h5')

################################################################################
#                           Load profile data
################################################################################
balloon_data  ='balloon_data/2017+2018/US_2017_2018'

data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

variable_bounds = pickle.load(open('feature_bounds.p','rb'))

# Interpolate profiles
cruise_alt = 13500
n = 75
alt_interpolated = np.linspace(0,cruise_alt,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated, rh, temp)

# Remove outliers from dataset
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated, temp_interpolated, elevations)

################################################################################
#                           Load profiles to plot
################################################################################
profiles = pickle.load(open('profiles_1000.p','rb'))['profiles']

################################################################################
#                           Reconsruct profile
################################################################################
profile = profiles[45]
predict_data = np.hstack((rh_interpolated[profile][:,1], temp_interpolated[profile][:,1]))
bounds = define_bounds([predict_data], n, type = [['constant_min','constant_max'], ['constant_min','constant_max']])[0]
elevation = np.array([data['height'][profile][0]])
y = np.array([normalize_variable_bounds(predict_data, n, bounds, variable_bounds, variable_type, elevation)])
y_pred = autoencoder.predict(y)
y_pred_normalized = normalize_variable_bounds(y_pred[0][:-1], n, bounds, variable_bounds, variable_type,
                                     np.array(y_pred[0][-1]), inverse = True)

################################################################################
#                           Plot profiles
################################################################################
path = 'profiles/'

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
# plt.xlim([-5,105])
# plt.savefig(path + '%i_%i' % (i, profiles[i]))

plt.show()
