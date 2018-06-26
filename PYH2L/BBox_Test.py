import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
import CollectedData_ReadingTest as CDRT

# print(CDRT)

seed = 7
np.random.seed(seed)
X_train, y_train, X_test, y_test = CDRT.prepare_data()
# print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')
print('-----------REFORMATTED DATA-----------')
# print('X_train shape', X_train.shape, '\n')
print(type(X_train))
#
# X_train = X_train[np.newaxis]
# X_test = X_test[np.newaxis]
# y_train = y_train[np.newaxis]
# y_test = y_test[np.newaxis]
X_train = X_train // 255
X_test = X_test // 255

# plt.imshow(X_train[0])
# plt.show()
print('X_train:', X_train, '\n\n\n', 'y_train:', y_train, '\n\n\n', 'X_test', X_test, '\n\n\n', 'y_test', y_test, '\n')

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(None, None, 3)))
model.get_weights()
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(GlobalMaxPooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(X_train, y_train)
validation_generator = test_datagen.flow(X_test, y_test)

with tf.device('/gpu:0'):
    model.fit_generator(train_generator, steps_per_epoch=1, epochs=2, validation_data=validation_generator,
                        validation_steps=1)
scores = model.evaluate(X_test, y_test, verbose=0)

