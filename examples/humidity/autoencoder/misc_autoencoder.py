import pickle
import numpy as np
import matplotlib.pyplot as plt
from autoencoder_example import build_autoencoder

def generate_practice_dataset_1(x0 = -1, x1 = 1, m0 = 0.5, m1 = 1.5, y0 = 0, n_lines = 50):
    x = np.linspace(x0,x1,100)
    y = list(np.zeros(n_lines))

    for i in range(n_lines):
        m = ((m1-m0)/n_lines)*i+m0
        yi = m*x
        y[i] = list(yi)

    y = np.array(y)

    return x, y

def normalize_inputs(inputs, bounds = [[0,1]], inverse = False):
    outputs = np.zeros(len(inputs))
    if inverse:
        for i in range(len(inputs)):
            outputs[i] = (inputs[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0])
    else:
        for i in range(len(inputs)):
            outputs[i] = inputs[i]*(bounds[i][1]-bounds[i][0]) + bounds[i][0]
    return outputs

def normalize_list(lists, bounds, inverse = 'False'):
    normalized_lists = list(np.zeros(len(lists)))
    for i in range(len(lists)):
        autoencoder_inputs = normalize_inputs(lists[i], bounds = bounds, inverse = inverse)
        normalized_lists[i] = list(autoencoder_inputs)

    return normalized_lists

if __name__ == "__main__":
    x, y = generate_practice_dataset_1(x0=-5,x1=5,m0=0.9,m1=1.1,n_lines=3000)

    encoding_dims = [5]
    input_shape = (y.shape[1],)

    autoencoder, encoder, decoder = build_autoencoder(encoding_dims, input_shape)

    bounds = [[-5,5] for i in range(len(y[0]))]
    normalized_lists = normalize_list(y, bounds, inverse = True)
    normalized_lists = np.array(normalized_lists)
    print(normalized_lists.shape)
    # Normalize inputs and validation data
    autoencoder.fit(normalized_lists[:1000],normalized_lists[:1000],epochs = 50,validation_data = (normalized_lists[1000:2000],normalized_lists[1000:2000]))

    # Input normalized values for encoder
    encoded = encoder.predict(normalized_lists[2000:])
    decoded_normalized = decoder.predict(encoded)

    # Convert decoded from normalized values to actual values
    decoded = normalize_list(decoded_normalized, bounds = bounds, inverse = False)

    plt.plot(x, decoded[30])
    plt.plot(x, y[30])
    plt.show()
