import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils

seed = 7
np.random.seed(seed)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')
print('-----------REFORMATTED DATA-----------')
print('TYPES', type(X_train))
print(X_train.shape)
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')
num_classes = y_test.shape[1]
print(y_test.shape[1])


def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


print('Model defined')

model = baseline_model()
print('Session info given')

with tf.device('/gpu:0'):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=1000, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

for i in range(0, len(X_train)):
    image = (X_test[i] * 255).reshape((28, 28)).astype("uint8")
    dig = model.predict_classes(np.atleast_2d(X_test[i]))
    actual = np.where(y_test[i] == 1.)[0][0]
    print('Loading image...')
    print('Most likely digit is {}'.format(dig[0]))
    print('Actual digit is {}'.format(actual, '\n'))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    if dig[0] != actual:
        print('\n', 'Incorrect identification detected.')
        plt.show()

