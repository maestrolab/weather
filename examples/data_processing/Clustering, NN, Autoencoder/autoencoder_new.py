import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from tensorflow.keras.models import Model
import seaborn as sns
import pandas as pd
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import csv
from random import seed, randint

# loads in the csv file with the x (100,000+) profiles and 404 or 405 columns (dimensions)
# depending on what features

# first row is all column names
col_names = ['Average Temperature', 'Average Humidity', 'Max Humidity', 'Elevation', 'Min Hag', 'Noise']
for i in range(400):
    dim_name = 'Dimension'
    col_names.append(dim_name+str(i))

# this data file includes elevation as a feature and minimum interpolation height
# as the minimum HAG value for that profile (index matching doesn't matter here,
# only matters for clustering)
data = pd.read_csv('autoencoder_data.csv', names=col_names, skiprows=1)
dataset = data

# split into a training and a test set

train_dataset = dataset.sample(frac=.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# seperate and store the noise value column from the data
# only needed to be store for regression model not autoencoder; however, 
# still needs to be removed

train_labels = train_dataset.pop('Noise')
test_labels = test_dataset.pop('Noise')

# use min max scaler here to normalize data

scaler = MinMaxScaler()

scaler.fit(train_dataset)
scaler.fit(test_dataset)

norm_train_data = scaler.transform(train_dataset)
norm_test_data = scaler.transform(test_dataset)

### define model and layers ###

# things to try: more layers or nodes, differnt layer type, different loss, different optimizer

model = Sequential()
# relu activation doesn't produce negative values so sigmoid is used
input_data = Input(shape=(405,))
encoded = Dense(128, activation='relu')(input_data)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

encoded = Dense(10, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(405, activation='sigmoid')(decoded)

optimizer = tf.keras.optimizers.RMSprop(.001)

autoencoder = Model(input_data, decoded)
# mse vs. cosine similarity really helped
autoencoder.compile(loss='cosine_similarity', optimizer=optimizer, metrics=['mae', 'mse'])

EPOCHS = 1000

# get accuracy; validation data? # monitor='val_loss' is original
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=20)
history = autoencoder.fit(norm_train_data, train_dataset, epochs=EPOCHS,
                    validation_split=.2, verbose=0, callbacks=[early_stop])

### where the errors start ###
autoencoder.fit(norm_train_data, train_dataset, epochs=10)

test_predictions= autoencoder.predict(norm_test_data).flatten()

# compare predictions to actual test data
# reshape since this is just one long array

test_predictions = test_predictions.tolist()

n = 405
# predicted
test = [test_predictions[i:i + n] for i in range(0, len(test_predictions), n)]
# inverse transform to get unnormalized values
test = scaler.inverse_transform(test)
# actual
test_dataset = test_dataset.values.tolist()  


### ONLY EXTRACTING AND FORMATTING RESULTS FROM HERE ON ###

# seed 
seed(1)
# get random profiles to plot
test_indexes = []
for i in range(10):
    value = randint(0, len(test_dataset))
    test_indexes.append(value)

# get correct altitude list for each profile to later plot
# x is the index of the test_index list, values inside that list are random profile numbers
for x in range(len(test_indexes)):
    altitude = []
    for i in range(200):
    # elevation + first measurement hag + 200 equal intervals between elevaton and cruising altitude
        hag = test_dataset[test_indexes[x]][3] + test_dataset[test_indexes[x]][4] + ((50000-test[test_indexes[x]][4])/200)*i
        altitude.append(hag)
     
    # error data for plotting for each profile
    err_temp = []
    err_hum = []
    for i in range(200):
        err_temp.append(abs((test[test_indexes[x]][i+5] - test_dataset[test_indexes[x]][i+5])))
        err_hum.append(abs((test[test_indexes[x]][i+205] - test_dataset[test_indexes[x]][i+205])))
        
    mae_temp = sum(err_temp)/200
    mae_hum = sum(err_hum)/200
        
    print('MAE TEMP for profile', test_indexes[x],':', mae_temp)
    print('MAE HUM for profile', test_indexes[x],':', mae_temp)
    
    ### temperature ###
    plt.figure()
    plt.plot(test[test_indexes[x]][5:205], altitude)
    plt.xlabel('Temperature in f')
    plt.ylabel('Altitude in ft')
    plt.title('Altitude VS. Predicted Temperature of profile %i' % test_indexes[x])
    plt.figure()
    plt.plot(test_dataset[test_indexes[x]][5:205], altitude)
    plt.xlabel('Temperature in f')
    plt.ylabel('Altitude in ft')
    plt.title('Altitude VS. True Temperature of profile %i' % test_indexes[x])
    
    # both true and predicted on same plot
    plt.figure()
    plt.plot(test[test_indexes[x]][5:205], altitude, color='red', label='Predicted')
    plt.plot(test_dataset[test_indexes[x]][5:205], altitude, color='green', label='True')
    plt.title('True and Predicted profiles of profile %i' % test_indexes[x])
    plt.legend()
    plt.show()
    
    # temp error
    plt.figure()
    plt.scatter(test_dataset[test_indexes[x]][5:205], test[test_indexes[x]][5:205])
    # 45 degree line
    plt.plot(test_dataset[test_indexes[x]][5:205], test_dataset[test_indexes[x]][5:205], 'y--')
    plt.xlabel('True Temperature Values')
    plt.ylabel('Predicted Temperature Values')
    plt.title('Predicted VS. True Temp Values of profile %i' % test_indexes[x])
    
    # showing abs error (altitude vs. abs error)
    plt.figure()
    plt.scatter(err_temp, altitude)
    
    ### humidity ###
    plt.figure()
    plt.plot(test[test_indexes[x]][205:], altitude)
    plt.xlabel('Relative Humidity in %')
    plt.ylabel('Altitude in ft')
    plt.title('Altitude VS. Predicted RH of profile %i' % test_indexes[x])
    plt.figure()
    plt.plot(test_dataset[test_indexes[x]][205:], altitude)
    plt.xlabel('Relative Humidity in %')
    plt.ylabel('Altitude in ft')
    plt.title('Altitude VS. True RH of profile %i' % test_indexes[x])
    
    # both true and predicted on same plot
    plt.figure()
    plt.plot(test[test_indexes[x]][205:], altitude, color='red', label='Predicted')
    plt.plot(test_dataset[test_indexes[x]][205:], altitude, color='green', label='True')
    plt.title('True and Predicted profiles of profile %i' % test_indexes[x])
    plt.legend()
    plt.show()
    
    # hum error
    plt.figure()
    plt.scatter(test_dataset[test_indexes[x]][205:], test[test_indexes[x]][205:])
    # 45 degree line
    plt.plot(test_dataset[test_indexes[x]][205:], test_dataset[test_indexes[x]][205:], 'y--')
    plt.xlabel('True Relative Humidity Values')
    plt.ylabel('Predicted Relative Humidity Values')
    plt.title('Predicted VS. True RH Values of profile %i' % test_indexes[x])
    
    # showing abs error (altitude vs. abs error)
    plt.figure()
    plt.scatter(err_hum, altitude)

    





