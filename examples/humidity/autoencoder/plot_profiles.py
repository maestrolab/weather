import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from autoencoder import *

# Load profile data
data = pickle.load(open('./balloon_data/2018/72469_2000-2018.p','rb'))
rh = np.array(data['humidity'])

# Load model
# date = '7_12_19_'
# autoencoder_name = date + 'AE'
n_params = 7
first_layer_dim = 2500
path = 'trained_models/encoding_dim_and_layer_dim_varied/'
type = '%i_params_%i_dim_' % (n_params, first_layer_dim)
autoencoder_name = type + 'AE'
autoencoder = tf.keras.models.load_model(path + autoencoder_name + '.h5')

# Interpolate profiles
n = autoencoder.layers[0].input_shape[1]
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated = np.zeros((len(rh), len(alt_interpolated), 2))
for i in range(len(rh)):
    alts, values = np.array(rh[i]).T
    fun = interp1d(alts, values)
    rh_interp = fun(alt_interpolated)
    rh_interpolated[i] = np.array([alt_interpolated, rh_interp]).T

# Normalize data to predict from model
profile = 6120
predict_data = np.array(rh_interpolated[profile][:,1])
bounds = define_bounds([predict_data])[0]
y = np.array([normalize_inputs(predict_data, bounds)])
y_pred = autoencoder.predict(y)
y_pred_normalized = normalize_inputs(y_pred[0], bounds, inverse = True)

# Plot original and reconstructed profiles
plt.figure()
plt.plot(predict_data, alt_interpolated, label = 'original')
plt.plot(y_pred_normalized, alt_interpolated, label = 'predicted')
plt.xlabel('Relative Humidity [%]')
plt.ylabel('Altitude [m]')
plt.legend()
plt.show()
