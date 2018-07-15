import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.optimizers import SGD
import CollectedData_ReadingTest as CDRT

# print(CDRT)

seed = 7
np.random.seed(seed)
X_train, y_train, X_test, y_test = CDRT.prepare_data()
# print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')

# print(type(X_train))
"""
Before reshaping the dimensions of the arrays are (6, 313, 1055) and (6, 4, 11). The extra dimension is added as keras requires input of the form 
(Number of images, image width, image height, number of channels), where number of channels refers to colour channels (RGB etc.), however, out images are 
greyscale so contain only one channel.
"""
X_train = np.reshape(X_train, (6, 313, 1055, 1))
X_test = np.reshape(X_test, (6, 313, 1055, 1))
"""
I'm rounding the coords purely incase they have been mutated again somehow. The img arrays are being divided by 255 and then rounded so that the images 
contain pixels that are either black or white (0 or 1... possibly other way round, not sure).
"""
y_train = abs(np.round(np.reshape(y_train, (6, 4, 11, 1)), 0))
y_test = abs(np.round(np.reshape(y_test, (6, 4, 11, 1)), 0))
X_train = abs(np.round(X_train // 255, 0))
X_test = abs(np.round(X_test // 255, 0))
print('-----------REFORMATTED DATA-----------')
print('X_train shape', X_train.shape, '\n')
print('X_test shape', X_test.shape, '\n')
print('y_train shape', y_train.shape, '\n')
print('y_test shape', y_test.shape, '\n')
# plt.imshow(X_train[0])
# plt.show()
print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')

filter_size = 3
pool_size = 2

"""
I don't fully understand why this needs to be done, but one of the tutorials did something similar, and adding this step stopped the code from crashing at 
the model.compile stage. This is probably the right thing to do, however I am not sure that the arguments of .reshape() are the ones we should be using.
"""
X_train = ((X_train.reshape(6, -1) - np.mean(X_train)) / np.std(X_train)).astype("uint8")
X_test = ((X_test.reshape(6, -1) - np.mean(X_test)) / np.std(X_test)).astype("uint8")
y_train = ((y_train.reshape(6, -1) - np.mean(y_train)) / np.std(y_train)).astype("bool")
y_test = ((y_test.reshape(6, -1) - np.mean(y_test)) / np.std(y_test)).astype("bool")

model = Sequential([
    Dense(X_train.shape[-1]*4, input_dim=X_train.shape[-1]),
    Activation('relu'),
    Dropout(0.2),
    Dense(y_train.shape[-1])
    ])
model.compile('adadelta', 'mse')

print(model.summary())


with tf.device('/gpu:0'):
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

