import os
from os import listdir
from os.path import isfile, join
import csv
import re
import numpy as np
from numpy import array

# for one-hot encoding
import pandas
from pandas.core.series import Series

# for CNN
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json # for model saving
from keras.datasets import mnist # MNIST
from sklearn.model_selection import train_test_split

# use MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocess input: (1) reshape to give a channel dimension (2) scale
# source: https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
# channels last with TensorFlow backend
img_rows = 28
img_cols = img_rows
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_classes = 10 # digits 0 ... 9
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# leaky ReLu probably not necessary here but
relu_leak = 0.3

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=input_shape))
model.add(LeakyReLU(alpha=relu_leak))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (5, 5)))
model.add(LeakyReLU(alpha=relu_leak))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000))
model.add(LeakyReLU(alpha=relu_leak))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.SGD(lr=0.01),
			metrics=['accuracy'])
batch_size = 100
epochs = 1 # digits classify extremely well even with a single epoch

H = model.fit(x_train, y_train,
					batch_size=batch_size,
					epochs=epochs,
					verbose=1,
					validation_data=(x_test, y_test))

# show predictions
y_new = model.predict_classes(x_test)

print(y_test) # one-hot format
print(y_new)
