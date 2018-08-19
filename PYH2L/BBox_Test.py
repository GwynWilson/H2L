import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Lambda
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
from keras import backend as K
K.set_image_dim_ordering('th')
from tensorflow.python.client import device_lib
from keras.optimizers import SGD
import CollectedData_ReadingTest as CDRT


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
plt.switch_backend('QT5Agg')
# device_lib.list_local_devices()
seed = 7
np.random.seed(seed)
X_train, X_test, test_coords2, train_coords2, dimensions, test_coords, train_coords = CDRT.prepare_data()
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
X_train = np.round(X_train-0.01, 0)
X_test = np.round(X_test-0.01, 0)
test_X_train = (X_train.reshape(X_train.shape[0], -1) - np.mean(X_train)) / np.std(X_train)
test_X_test = (X_test.reshape(X_test.shape[0], -1) - np.mean(X_test)) / np.std(X_test)
test_Y_train = train_coords.reshape(train_coords.shape[0], -1)
test_Y_test = test_coords.reshape(test_coords.shape[0], -1)


# shape = X_test.shape+(1,)
# print(shape)
# print('X_train shape', X_train.shape, '\n')
# print('X_test shape', X_test.shape, '\n')
# X_train = np.round(X_train, 0)
# X_test = np.round(X_test, 0)
# X_train = np.reshape(X_train, X_train.shape+(1,))
# X_test = np.reshape(X_test, X_test.shape+(1,))
# test_coords2 = np.reshape(test_coords2, test_coords2.shape+(1,))
# train_coords2 = np.reshape(train_coords2, train_coords2.shape+(1,))


model = Sequential([
    Dense(6, input_dim=test_X_train.shape[-1]),
    Activation('elu'),
    Dropout(0.6),
    Dense(test_Y_train.shape[-1], activation='sigmoid'),
    Dropout(0.8),
    ])
model.compile('adadelta', 'mse')


# model = Sequential([
#     Dense(1, activation='sigmoid', input_shape=(shape[1], shape[2], 1)),
#     Dropout(0.9),
#     Conv2D(1, (15, 15), activation='sigmoid', data_format='channels_last', padding='same')
#     ])
# model.compile('adadelta', 'mse')

print(model.summary())


with tf.device('/gpu:0'):
    model.fit(test_X_train, test_Y_train, epochs=1000, validation_data=(test_X_test, test_Y_test), verbose=3)

# with tf.device('/gpu:0'):
#     model.fit(X_train, train_coords2, epochs=10, validation_data=(X_test, test_coords2), verbose=3)


shapearoo = test_X_test.shape
shapearoo2 = X_test.shape
test = model.predict(test_X_test)
print(test.shape)
empty3 = []
for y in range(0, test.shape[0]):
    empty = []
    for u in range(1, test.shape[1]//4):
        empty.append(test[y][(u-1)*4:u*4])
    empty3.append(empty)
empty2 = np.array(empty3)
print(empty2)
shaped_test = np.transpose(np.transpose(test).reshape(4, -1))
shaped_test_input = X_test.reshape(shapearoo2[:3])
# print(shaped_test)
print(shaped_test_input)

x = []
y = []
w = []
h = []
for c in range(0, shaped_test.shape[0]):
    x.append(shaped_test[c][0])
    y.append(shaped_test[c][1])
    w.append((shaped_test[c][2]-shaped_test[c][0])*dimensions[0])
    h.append((shaped_test[c][3]-shaped_test[c][1])*dimensions[1])


# shapearoo = X_test.shape
# test = model.predict(X_test)
# print(test.shape)
# shaped_test = test.reshape(shapearoo[:3])
# shaped_test_input = X_test.reshape(shapearoo[:3])
# # print(shaped_test)
# print(shaped_test_input)

print(dimensions)
for i in range(0, test.shape[0]):
    plt.imshow(shaped_test_input[i], alpha=1, cmap='binary')
    for f in range(0, 10):
        bot_left_x = min(empty2[i][f][0], empty2[i][f][2])*dimensions[1]
        bot_left_y = min(empty2[i][f][1], empty2[i][f][3])*dimensions[0]
        width = abs(empty2[i][f][0]-empty2[i][f][2])*dimensions[1]
        height = abs(empty2[i][f][1]-empty2[i][f][3])*dimensions[0]
        plt.gca().add_patch(matplotlib.patches.Rectangle((bot_left_x, bot_left_y), width,
                                                         height, ec='r', fc='none', lw=3))
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


# for i in range(0, test.shape[0]):
#     # plt.imshow(shaped_test_input[i], alpha=1, cmap='binary')
#     plt.imshow(shaped_test[i], alpha=0.5)
#     fig_manager = plt.get_current_fig_manager()
#     fig_manager.window.showMaximized()
#     plt.show()
# scores = model.evaluate(X_test, test_coords2, verbose=3)
# print(scores)
# print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

