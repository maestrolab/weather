import pickle
import numpy as np
import matplotlib.pyplot as plt
from autoencoder_example import build_autoencoder
from examples.boom.twister.humidity_spline_parametrization.misc_humidity import prepare_standard_profiles,\
                                            package_data

data_ = pickle.load(open('2018_06_18_12_all_temperature.p','rb'))

# Prepare standard profiles
# path = './../../../../data/weather/standard_profiles/standard_profiles.p'
# data = prepare_standard_profiles(standard_profiles_path = path)
# rh_alts, rh = package_data(data['relative humidity'])
# temp_alts, temp = package_data(data['temperature'])

# rh = np.array([rh])
# temp = np.array([temp])
print(data_)
encoding_dims = [10]
# input_shape = (rh[0].shape[0],)
# input_shape = (temp[0].shape[0],)

input_shape = (data_.shape[0],)
print(encoding_dims, input_shape)
autoencoder, encoder, decoder = build_autoencoder(encoding_dims, input_shape)

# y = np.linspace(-60,15,len(temp_alts))
# y = np.flip(y)
# y = np.array([y])
# validation_data = (y, y)
print(data_)
# autoencoder.fit(rh, rh, epochs=100)
# autoencoder.fit(temp, temp, epochs = 400, validation_data = validation_data)
autoencoder.fit(data_, data_, epochs = 400)#, validation_data = validation_data)

encoded = encoder.predict(y)
decoded = decoder.predict(encoded)
# Equivalent to encoder.predict() -> decoder.predict() method
autoencoded = autoencoder.predict(y)

plt.plot(decoded[0])
plt.plot(y[0])
plt.show()
