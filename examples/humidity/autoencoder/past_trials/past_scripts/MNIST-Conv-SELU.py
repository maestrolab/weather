# Adapted KERAS tutorial

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import pickle

from autoencoder import *

batch_size = 128
num_classes = 10
epochs = 30

# Load balloon data
# balloon_data  ='balloon_data/2017+2018/US_2017_2018'
balloon_data  ='balloon_data/2017+2018/US_2017_2018_72456_72214_72403_72265_72582_only'

data = pickle.load(open(balloon_data + '.p','rb'))
rh = np.array(data['humidity'])
temp = np.array(data['temperature'])
elevations = np.array([n[0] for n in np.array(data['height'])])

# Interpolate profiles
n = 75
alt_interpolated = np.linspace(0,13500,n)
rh_interpolated, temp_interpolated = interpolate_profiles(alt_interpolated,
                                                          rh, temp)

# Remove outliers from dataset
rh_interpolated, temp_interpolated, elevations = outliers(rh_interpolated, temp_interpolated, elevations)


x_train = rh_interpolated[:,:,1][:]

img_rows = 1
img_cols = 75

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    # x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
#x_train = (x_train - np.mean(x_train))/np.std(x_train)

y_train = keras.utils.to_categorical(x_train, num_classes)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(1,5),
                 activation='selu',
                 input_shape=input_shape,kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(Conv2D(64, (1,5), activation='selu',kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(AlphaDropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='selu',kernel_initializer='lecun_normal',bias_initializer='zeros'))
model.add(AlphaDropout(0.5))
model.add(Dense(num_classes, activation='softmax',kernel_initializer='lecun_normal',bias_initializer='zeros'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.layers[-1].input_shape)
print(model.layers[-1].output_shape)

# x_train = np.expand_dims(x_train, axis = 2)
model.fit(x_train, x_train,
          batch_size=batch_size,
          epochs=epochs)

score = model.evaluate(x_train[100], y_train[100], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

f = open('MNIST_SELU_results.txt', 'a')
f.write('Test loss:' + str(score[0]) + ' Test accuracy:' + str(score[1]) +  '\n')  # python will convert \n to os.linesep
f.close()
