import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import os

class Autoencoder:
    '''Notes: (change later)

        Inputs:
         - data: input data for model to train
                 foramt: np.array([np.array([])])
                 shape: data.shape = (n_sets_of_data, n_data_points)
                        data[i] = (n_data_points,); for i in range(len(n_sets_of_data))
         - encoding_dims: list of encoder layer dimensions in ascending order
    '''

    def __init__(self, data, encoding_dims = [10]):
        self.data = np.array(data)
        self.encoding_dims = encoding_dims
        self._input_shape = self.data[0].shape

    def _prepare_validation_data(self):
        '''Divide data for training and validation'''
        if len(self.data)%2 == 0:
            index = int(len(self.data)/2)
        else:
            index = int(np.around(len(self.data)/2))

        training_data = self.data[:index]
        validation_data = self.data[index:]

        self.training_data = self._normalize_parameters(training_data)
        self.validation_data = self._normalize_parameters(validation_data)

    def _encoder(self, activation):
        self.encoder_input = Input(shape = self._input_shape)
        encoded = Dense(self.encoding_dims[0], activation = activation)(self.encoder_input)
        if len(self.encoding_dims) > 1:
            for layer_dim in self.encoding_dims[-1:0:-1]:
                encoded = Dense(layer_dim, activation = activation)(encoded)
        self.encoder_output = encoded
        self.encoder = Model(self.encoder_input, self.encoder_output)

    def _decoder(self, activation):
        self.decoder_input = Input(shape = (min(self.encoding_dims),))
        if len(self.encoding_dims) > 1:
            decoded = Dense(self.encoding_dims[1], activation = activation)(self.decoder_input)
            for layer_dim in self.encoding_dims[2:-1]:
                decoded = Dense(layer_dim, activation = activation)(decoded)
        else:
            decoded = self.decoder_input
        self.decoder_output = Dense(self._input_shape[0], activation = activation)(decoded)
        self.decoder = Model(self.decoder_input, self.decoder_output)

    def init_model(self, activation = 'relu'):
        self._encoder(activation = activation)
        self._decoder(activation = activation)

        # Need to generalize for muliple hidden layers
        input = Input(shape = self._input_shape)
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        self.autoencoder = Model(input, decoded)

    '''Import normalize_list and normalize_parameters'''
    def _normalize_list(self, inputs, bounds, inverse):
        outputs = np.zeros(len(inputs))
        if inverse:
            for i in range(len(inputs)):
                outputs[i] = inputs[i]*(bounds[i][1]-bounds[i][0]) + bounds[i][0]
        else:
            for i in range(len(inputs)):
                outputs[i] = (inputs[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0])
        return outputs

    def _normalize_parameters(self, data, bounds = None, inverse = False):
        normalized_data = list(np.zeros(len(data)))
        for i in range(len(data)):
            if bounds == None:
                bounds = [[min(data[i]),max(data[i])] for j in range(len(data[i]))]
            normalized_list = self._normalize_list(data[i], bounds = bounds,
                                                   inverse = inverse)
            normalized_data[i] = list(normalized_list)
        return np.array(normalized_data)

    def train(self, batch_size = 10, epochs = 25):
        self.autoencoder.compile(optimizer = 'adadelta',
                                 loss = 'binary_crossentropy')
        self._prepare_validation_data()
        self.autoencoder.fit(self.training_data, self.training_data,
                             epochs = epochs,
                             batch_size = batch_size,
                             validation_data = (self.validation_data,
                                                self.validation_data))

    def __call__(self, parameter):
        # Include an option to save the data
        encoder_input = self._normalize_parameters(parameter)
        encoded = self.encoder.predict(encoder_input)
        decoded = self.decoder.predict(encoded)
        bounds = [[min(parameter[0]),max(parameter[0])] for j in range(len(parameter[0]))]
        decoded = self._normalize_parameters(decoded, bounds = bounds, inverse = True)
        print('Input: %s\nEncoded: %s\nDecoded: %s' % (parameter[0], encoded, decoded))

        return decoded

    def save_model(self):
        # make directory specific to model
        # save autoencoder, encoder, and decoder all separately in directory
        pass

if __name__ == '__main__':
    x = np.random.randint(2,size=(100,4))
    auto = Autoencoder(data = x, encoding_dims = [10])
    auto.init_model()
    auto.train(batch_size=20, epochs=50)
    parameter = np.array([np.array([1,1,1,1])])
    result = auto(parameter = parameter)
