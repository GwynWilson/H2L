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
from keras.optimizers import SGD
import CollectedData_ReadingTest as CDRT

# print(CDRT)

seed = 7
np.random.seed(seed)
X_train, X_test, test_coords2, train_coords2, dimensions = CDRT.prepare_data()
print(X_train.shape)
plt.imshow(X_train[0])
plt.imshow(train_coords2[0], alpha=0.1)
plt.show()
# print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')

# print(type(X_train))
"""
Before reshaping the dimensions of the arrays are (6, 313, 1055) and (6, 4, 11). The extra dimension is added as keras requires input of the form 
(Number of images, image width, image height, number of channels), where number of channels refers to colour channels (RGB etc.), however, out images are 
greyscale so contain only one channel.
"""
# TODO: Remove the hard coding
print('X_train shape', X_train.shape, '\n')
print('X_test shape', X_test.shape, '\n')
X_train = np.reshape(X_train, X_train.shape+(1,))
X_test = np.reshape(X_test, X_test.shape+(1,))
test_coords2 = np.reshape(test_coords2, test_coords2.shape+(1,))
train_coords2 = np.reshape(train_coords2, train_coords2.shape+(1,))


model = Sequential([
    Conv2D(1, (15, 15), input_shape=(313, 1055, 1), activation='softmax', data_format='channels_last', padding='same'),
    Dropout(0.4),
    # MaxPooling2D(pool_size=(2, 1)),
    ])
model.compile('adadelta', 'mse')

print(model.summary())


with tf.device('/gpu:0'):
    model.fit(X_train, train_coords2, epochs=10, validation_data=(X_test, test_coords2), verbose=3)
scores = model.evaluate(X_test, test_coords2, verbose=3)
print(scores)
# print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

