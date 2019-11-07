import pickle
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

from misc_humidity import prepare_standard_profiles
from autoencoder import *
from optimization_tools.DOE import DOE

# Load profile data
data = pickle.load(open('./balloon_data/72469_2009-2018.p','rb'))
rh = np.array(data['humidity'])

def autoencoder_DOE(inputs):
    rh = inputs['data']

    epochs = 10
    batch_size = int(inputs['batch_size'])
    first_layer_dim = int(inputs['first_layer_dim'])
    encoding_dim = 8
    n = int(inputs['n'])

    # Interpolate profiles
    alt_interp = np.linspace(0,13500,n)
    rh_interp = np.zeros((len(rh), len(alt_interp), 2))
    for i in range(len(rh)):
        alt, values = np.array(rh[i]).T
        fun = interp1d(alt, values)
        rh_interp[i] = np.array([alt_interp, fun(alt_interp)]).T

    # Normalize data
    test_data = rh_interp[:,:,1][:]
    bounds = define_bounds(test_data)
    y = np.array([normalize_inputs(test_data[i],bounds[i]) for i in
                 range(test_data.shape[0])])

    input = tf.keras.Input(shape=(y.shape[1],))
    x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(input)
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(first_layer_dim, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(y.shape[1], activation='sigmoid')(x)
    model = tf.keras.Model(input,decoded)

    # Build encoder and decoder
    decoder_input = tf.keras.Input(shape=(encoding_dim,))
    x = model.get_layer(index=-2)(decoder_input)
    decoded = model.get_layer(index=-1)(x)

    encoder = tf.keras.Model(input, encoded)
    decoder = tf.keras.Model(decoder_input, decoded)

    # Compile autoencoder
    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    # Train autoencoder
    break_points = {1:3000}
    history = model.fit(y[:break_points[1]], y[:break_points[1]],
                        epochs=epochs,
                        batch_size=batch_size)

    return {'loss':history.history['loss'][-1]}

# List of variables and corresponding bounds
variables = ['epochs','batch_size','first_layer_dim','encoding_dim','n']
bounds = [[10,100],[2,56],[75,375],[5,17],[75,120]]
i1 = 1
i2 = 2
i3 = 4

# Define points
problem = DOE(levels=4, driver='Full Factorial')
problem.add_variable(variables[i1], lower=bounds[i1][0], upper=bounds[i1][1], type=int)
problem.add_variable(variables[i2], lower=bounds[i2][0], upper=bounds[i2][1], type=int)
problem.add_variable(variables[i3], lower=bounds[i3][0], upper=bounds[i3][1], type=int)
problem.define_points()

# Run for a function with dictionary as inputs
problem.run(autoencoder_DOE, cte_input={'data':rh})
problem.find_influences(not_zero=True)
problem.find_nadir_utopic(not_zero=True)
print('Nadir: ', problem.nadir)
print('Utopic: ', problem.utopic)

# Plot factor effects
problem.plot(xlabel=[variables[i1], variables[i2], variables[i3]],
             ylabel=['loss'], number_y=1)

# Store DOE
variable_and_bounds = '%s_%i-%i_%s_%i-%i_%s_%i-%i' % (variables[i1],
                                                   bounds[i1][0], bounds[i1][1],
                                                   variables[i2],
                                                   bounds[i2][0], bounds[i2][1],
                                                   variables[i3],
                                                   bounds[i3][0], bounds[i3][1])
fileObject = open('saved_DOEs/DOE_FullFactorial_'+variable_and_bounds, 'wb')
pickle.dump(problem, fileObject)
fileObject.close()
