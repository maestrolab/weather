# Example Source:
# https://blog.keras.io/building-autoencoders-in-keras.html

import pickle

from keras.layers import Input, Dense
from keras.models import Model

def build_autoencoder(encoding_dims = [32], input_shape = (784,)):
    input_img = Input(shape=input_shape)
    encoded = Dense(encoding_dims[0], activation='relu')(input_img)

    if len(encoding_dims) > 1:
        for layer_dim in encoding_dims[-1:0:-1]:
            encoded = Dense(layer_dim, activation='relu')(encoded)
        decoded = Dense(encoding_dims[1], activation='relu')(encoded)
        for layer_dim in encoding_dims[2:]:
            decoded = Dense(layer_dim, activation='relu')(decoded)

    decoded = Dense(input_shape[0], activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(min(encoding_dims),))
    decoded_layer = autoencoder.layers[-1]
    print(encoded_input)
    print(decoded_layer(encoded_input))
    decoder = Model(encoded_input, decoded_layer(encoded_input))

    # Training autoencoder
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder, decoder

from keras.datasets import mnist
import numpy as np

autoencoder, encoder, decoder = build_autoencoder()

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_test.shape)

# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
#
# models = {'autoencoder':autoencoder, 'encoder':encoder, 'decoder':decoder}
#
# mnist_model = open('mnist_model.p','wb')
# pickle.dump(models, mnist_model)
# mnist_model.close()

# models = pickle.load(open('mnist_model.p','rb'))
# autoencoder = models['autoencoder']
# encoder = models['encoder']
# decoder = models['decoder']

# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
#
# import matplotlib.pyplot as plt
#
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
