import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.datasets import mnist

# Loading our datasets -- containing images of handwritten digits 0f 0 - 9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Input image dimensions
img_width, img_height = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_width, img_height)
    x_test = x_test.reshape(x_test.shape[0], 1, img_width, img_height)
    input_shape = (1, img_width, img_height)
else:
    x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
    x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing our dataets
x_train = x_train / 255
x_test = x_test / 255

train_y = to_categorical(y_train)
test_y = to_categorical(y_test)

print("x_train shape is: ", x_train.shape)
print("x_test shape is: ", x_test.shape)
print("train_y shape is: ", train_y.shape)
print("test_y shape is: ", test_y.shape)

# Building our model
model = Sequential()
model.add(Convolution2D(filters = 32, kernel_size = (5, 5), input_shape = input_shape, activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(filters = 64, kernel_size = (5, 5), input_shape = input_shape, activation = "relu"))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Flatten()) # reducing the dimension into 1D 
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "softmax"))  # 10---> num of clases,i.e, 0-9

model.summary()

# Compiling our model
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(x_train, train_y, validation_data = (x_test, test_y), epochs = 12, batch_size = 64)
scores = model.evaluate(x_test, test_y)
print("CNN Error : %.2f%%" % (100- scores[1] * 100))
model.save("digit_classifier.mnist")