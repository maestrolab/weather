import tensorflow as tf
import numpy as np

test_data = np.random.randint(10,size=(1000,10))

def normalize_inputs(x, bounds):
    normalized_inputs = (x-bounds[0])/(bounds[1]-bounds[0])
    return normalized_inputs

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(len(test_data), activation='relu'))
# model.add(tf.keras.layers.Dense(5, activation='softmax'))
# model.add(tf.keras.layers.Dense(2, activation='softmax'))
# model.add(tf.keras.layers.Dense(5, activation='relu'))
# model.add(tf.keras.layers.Dense(len(test_data), activation='relu'))
#
# model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(np.array([test_data[:500]]), epochs=10, batch_size=32)

bounds = [[np.min(test_data[i]),np.max(test_data[i])] for i in range(test_data.shape[0])]
y = np.array([normalize_inputs(test_data[i],bounds[i]) for i in range(test_data.shape[0])])
print(y)

input = tf.keras.Input(shape=(test_data.shape[1],))
encoded_0 = tf.keras.layers.Dense(7, activation='relu')(input)
encoded_1 = tf.keras.layers.Dense(5, activation='relu')(encoded_0)
decoded_0 = tf.keras.layers.Dense(7, activation='sigmoid')(encoded_1)
decoded_1 = tf.keras.layers.Dense(test_data.shape[1], activation='sigmoid')(decoded_0)

model = tf.keras.Model(input,decoded_1)

model.compile(optimizer='adadelta', loss='binary_crossentropy')

model.fit(y[:500], y[:500], epochs=100, batch_size=25,
          validation_data=(y[500:],y[500:]))

# model.fit(test_data[:5000], test_data[:5000], epochs=10, batch_size=32,
#           validation_data=(test_data[5000:],test_data[5000:]))

predict_data = np.random.randint(10,size=(,))

bounds = [np.min(predict_data),np.max(predict_data)]
y = normalize_inputs(predict_data, bounds)
print(model.predict(y))
