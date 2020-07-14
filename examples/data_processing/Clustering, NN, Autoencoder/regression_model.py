import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import pandas as pd
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import csv

# loads in the csv file with the x (100,000+) profiles and 404 columns (dimensions)

# first row is all column names
col_names = ['Average Temperature', 'Average Humidity', 'Max Humidity', 'Elevation', 'Noise']
for i in range(400):
    height_name = 'Dimension'
    col_names.append(height_name+str(i))

# this data file includes elevation as a feature and minimum interpolation height
# as the minimum HAG value for that profile (index matching doesn't matter here,
# only matters for clustering)
data = pd.read_csv('main_data_1_w_elevation.csv', names=col_names, skiprows=1)
dataset = data

# split into a training and a test set

train_dataset = dataset.sample(frac=.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# seperate and store the noise value column from the data

train_labels = train_dataset.pop('Noise')
test_labels = test_dataset.pop('Noise')

# print(test_labels)

# use min max scaler here to normalize data, automatically inverse transforms??

scaler = MinMaxScaler()

scaler.fit(train_dataset)
scaler.fit(test_dataset)

norm_train_data = scaler.transform(train_dataset)
norm_test_data = scaler.transform(test_dataset)

### build the model, specify [LAYERS, NEURONS, FUNCTION, OPTIMIZER] here ###

def build_model():
    model = keras.Sequential([
        layers.Dense(128, kernel_initializer='normal', activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, kernel_initializer='normal', activation=tf.nn.relu),
        layers.Dense(64, kernel_initializer='normal', activation=tf.nn.relu),
        layers.Dense(64, kernel_initializer='normal', activation=tf.nn.relu),
        layers.Dense(1, kernel_initializer='normal', activation=tf.nn.linear)
        ])
    
    optimizer = tf.keras.optimizers.RMSprop(.001)

    # mean squared error and mean absolute error

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    
    return model

model = build_model()

model.summary()

# test that model works here; yes it does!
example_batch = norm_train_data[:10]
example_result = model.predict(example_batch)
# print(example_result)

# train the model here, set to high epochs but will stop early (see below)

EPOCHS = 1000

# stops when model stops getting better
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
# history = model.fit(norm_train_data, train_labels, epochs=EPOCHS,
#                     validation_split=.2, verbose=0, callbacks=[early_stop])
history = model.fit(norm_train_data, train_labels, epochs=EPOCHS,
                    validation_split=.2, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch 

# error plots (MAE and MSE)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel("Mean Absolute Error (Noise in pLdB)")
    plt.plot(hist['epoch'], hist['mae'],
              label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
              label='Val Error')
    plt.title('MAE VS. Epoch')
    plt.legend()
    plt.ylim([0, 2])
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel("Mean Squared Error (Noise^2)")
    plt.plot(hist['epoch'], hist['mse'],
              label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
              label='Val Error')
    plt.title('MSE VS. Epoch')
    plt.legend()
    plt.ylim([0, 10])
    
plot_history(history)

# get the mae here
loss, mae, mse = model.evaluate(norm_test_data, test_labels, verbose=0)
print('Testing Set Mean Abs Error: {:5.2f} pLdB'.format(mae))
print('Testing Set Mean Squared Error: {:5.2f} pLdB'.format(mse))

test_predictions= model.predict(norm_test_data).flatten()

# around 23000 points (profiles) in test set 
# shows error for the amount of epochs
plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel('Prediction Error pLdB')
plt.ylabel('Amount of profiles with respective error')
plt.title('Count VS. Prediction Error')

plt.figure()
plt.scatter(test_labels, test_predictions)
# trendline for data
z = np.polyfit(test_labels, test_predictions, 1)
p = np.poly1d(z)
plt.plot(test_labels, p(test_labels),"r--")
# 45 degree line (x=y), where points would lie if predictions were perfect
plt.plot(test_labels, test_labels, "y--")
plt.xlabel('True Values (Noise in pLdB)')
plt.ylabel('Predictions (Noise in pLdB)')
plt.title('Predicted VS. True Noise')













# # stop when loss starts increasing, patience is amount of epochs to check for improvement
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# history = model.fit(norm_train_data, train_labels, epochs=EPOCHS,
#                     validation_split=.2, verbose=0, callbacks=[early_stop])

# # plot the losses

# def plot_history(history):
#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch
    
#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel("Mean Absolute Error (Noise in pLdB)")
#     plt.plot(hist['epoch'], hist['mae'],
#               label='Train Error')
#     plt.plot(hist['epoch'], hist['val_mae'],
#               label='Val Error')
#     plt.legend()
#     plt.ylim([0, 100])
    
#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel("Mean Squared Error")
#     plt.plot(hist['epoch'], hist['mse'],
#               label='Train Error')
#     plt.plot(hist['epoch'], hist['val_mse'],
#               label='Val Error')
#     plt.legend()
#     plt.ylim([0, 100])
    
# plot_history(history)

















