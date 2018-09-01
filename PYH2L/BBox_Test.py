import CollectedData_ReadingTest as CDRT
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Lambda
from keras.layers import Dropout
from keras import backend as K
K.set_image_dim_ordering('th')


plt.switch_backend('QT5Agg')
# device_lib.list_local_devices()
seed = 7
np.random.seed(seed)
X_train, X_test, test_coords2, train_coords2, dimensions, train_coords, test_coords = CDRT.prepare_data()
# print(X_train.shape)

# for k in range(0, len(X_train)):
#     plt.imshow(X_train[k])
#     print(train_coords[k])
#     for f in range(0, 10):
#         bot_left_x = min(train_coords[k][f][0], train_coords[k][f][2])
#         bot_left_y = min(train_coords[k][f][1], train_coords[k][f][3])
#         width = abs(train_coords[k][f][0]-train_coords[k][f][2])
#         height = abs(train_coords[k][f][1]-train_coords[k][f][3])
#         plt.gca().add_patch(matplotlib.patches.Rectangle((bot_left_x, bot_left_y), width,
#                                                          height, ec='r', fc='none', lw=3))
#     fig_manager = plt.get_current_fig_manager()
#     fig_manager.window.showMaximized()
#     plt.show()


# print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')

# print(type(X_train))
"""
Before reshaping the dimensions of the arrays are (6, 313, 1055) and (6, 4, 11). The extra dimension is added as keras requires input of the form 
(Number of images, image width, image height, number of channels), where number of channels refers to colour channels (RGB etc.), however, out images are 
greyscale so contain only one channel.
"""
X_train = np.round((X_train/255)-0.1, 0)
X_test = np.round((X_test/255)-0.1, 0)
X_train = np.logical_not(X_train).astype(int)
X_test = np.logical_not(X_test).astype(int)
test_X_train = (X_train.reshape(X_train.shape[0], -1) - np.mean(X_train)) / np.std(X_train)
test_X_test = (X_test.reshape(X_test.shape[0], -1) - np.mean(X_test)) / np.std(X_test)
test_Y_train = train_coords.reshape(train_coords.shape[0], -1)
test_Y_test = test_coords.reshape(test_coords.shape[0], -1)
print('Data reshaped')


model = Sequential([
    Dense(64, input_dim=test_X_train.shape[-1]),
    Activation('elu'),
    Dropout(0.9),
    Dense(test_Y_train.shape[-1], activation='sigmoid'),
    Dropout(0.2),
    ])
model.compile(optimizer='adadelta', loss='mse')


print(model.summary())

num_test_images = len(test_X_train)
with tf.device('/gpu:0'):
    model.fit(test_X_train[:num_test_images//2], test_Y_train[:num_test_images//2], epochs=100, validation_data=(test_X_test[num_test_images//2:],
                                                                                                                test_Y_test[num_test_images//2:]), verbose=3)


shapearoo = test_X_test.shape
shapearoo2 = X_test.shape
test = model.predict(test_X_test)
empty3 = []
for y in range(0, test.shape[0]):
    empty = []
    for u in range(1, test.shape[1]//4):
        empty.append(test[y][(u-1)*4:u*4])
    empty3.append(empty)
empty2 = np.array(empty3)
shaped_test = np.transpose(np.transpose(test).reshape(4, -1))
shaped_test_input = X_test.reshape(shapearoo2[:3])


x = []
y = []
w = []
h = []
for c in range(0, shaped_test.shape[0]):
    x.append(shaped_test[c][0])
    y.append(shaped_test[c][1])
    w.append((shaped_test[c][2]-shaped_test[c][0])*dimensions[0])
    h.append((shaped_test[c][3]-shaped_test[c][1])*dimensions[1])


for i in range(0, test.shape[0]):
    plt.imshow(shaped_test_input[i])
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
