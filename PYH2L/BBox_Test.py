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
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K
K.set_image_dim_ordering('th')
# from tensorflow.python.client import device_lib
from keras.optimizers import SGD
import CollectedData_ReadingTest as CDRT


plt.switch_backend('QT5Agg')
# device_lib.list_local_devices()
seed = 7
np.random.seed(seed)
X_train, X_test, test_coords2, train_coords2, dimensions = CDRT.prepare_data()
print(X_train.shape)
# plt.imshow(X_train[0])
# plt.imshow(train_coords2[0], alpha=0.1)
# plt.show()
# print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')

# print(type(X_train))
"""
Before reshaping the dimensions of the arrays are (6, 313, 1055) and (6, 4, 11). The extra dimension is added as keras requires input of the form 
(Number of images, image width, image height, number of channels), where number of channels refers to colour channels (RGB etc.), however, out images are 
greyscale so contain only one channel.
"""
# TODO: Remove the hard coding
shape = X_train.shape+(1,)
print(shape)
print('X_train shape', X_train.shape, '\n')
print('X_test shape', X_test.shape, '\n')
X_train = np.round(X_train, 0)
X_test = np.round(X_test, 0)
X_train = np.reshape(X_train, X_train.shape+(1,))
X_test = np.reshape(X_test, X_test.shape+(1,))
test_coords2 = np.reshape(test_coords2, test_coords2.shape+(1,))
train_coords2 = np.reshape(train_coords2, train_coords2.shape+(1,))


model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(shape[1], shape[2], 1)),
    Dropout(0.3),
    Conv2D(1, (15, 15), activation='sigmoid', data_format='channels_last', padding='same'),
    Dropout(0.3),
    ])
model.compile('adadelta', 'mse')

print(model.summary())


with tf.device('/gpu:0'):
    model.fit(X_train, train_coords2, epochs=3, validation_data=(X_test, test_coords2), verbose=3)


shapearoo = X_test.shape
test = model.predict(X_test)
print(test.shape)
shaped_test = test.reshape(shapearoo[:3])
shaped_test_input = X_test.reshape(shapearoo[:3])
# print(shaped_test)
print(shaped_test_input)
for i in range(0, test.shape[0]):

    # plt.imshow(shaped_test_input[i], alpha=1, cmap='binary')
    plt.imshow(shaped_test[i], alpha=0.5)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()
# scores = model.evaluate(X_test, test_coords2, verbose=3)
# print(scores)
# print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

