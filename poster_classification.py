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

# at this point you need to have run file_matching.pl, then discard the movies indicated
# (those without posters)

# get_data will perform the next step of filtering: omitting movies with no genre

def get_data(input_shape, all_genres=True):
  directory = "posters"
  meta_directory = "metadata"
  no_files = len([f for f in listdir(meta_directory) if isfile(join(meta_directory, f))])    
  x = np.empty((no_files, input_shape[0], input_shape[1], input_shape[2]))
  y = pandas.DataFrame()
  idx = 0
  missing_genres = []
  for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
      uuid = filename.split('.')[0]
      filepath = os.path.join(directory, filename)
      img = image.load_img(path=filepath,grayscale=False,target_size=input_shape)
      img = image.img_to_array(img)
      x[idx,:,:,:] = img
      filepath = os.path.join(meta_directory, uuid + ".txt")
      genre_found = False
      with open(filepath) as tsv:
        for line in csv.reader(tsv):
          res = re.search("^genre\t(.*)$", line[0])
          if(res != None):
            genres = res.group(1).split("\t")
            if(len(genres) > 0):
                genre_found = True
            if all_genres:
              for g in genres:
                  y = y.append(Series({'movie':uuid, 'genre':g}), ignore_index=True)
            else:
              y = y.append(Series({'movie':uuid, 'genre':genres[0]}), ignore_index=True)

      if(not genre_found):
        missing_genres.append(idx)
        print("No genre found for " + uuid + "! Omitting it...")        
    idx += 1
    y['count'] = 1
  # remove movies tagged as genre-missing
  x = np.delete(x, missing_genres, axis=0)
  return((x,y))

save_model = False

input_shape = (100, 150, 3)

(x, y) = get_data(input_shape, all_genres=False)
y = y.pivot(index='movie', columns='genre', values='count').fillna(0)
y.to_csv("uuid_genre_mapping.csv", index=True)
exit()

# on this test set:
# 1 : Action
# 2 : Adventure
# 3 : Animation
# 4 : Comedy
# 5 : Crime
# 6 : Documentary
# 7 : Drama
# 8 : Family
# 9 : Fantasy
# 10 : History
# 11 : Horror
# 12 : Music
# 13 : Mystery
# 14 : Romance
# 15 : Science Fiction
# 16 : TV Movie
# 17 : Thriller
# 18 : War
# 19 : Western

y_arr = np.array(y)

idx = 15 # classify genre as above
x_train, x_test, y_train, y_test = train_test_split(x, y_arr[:,idx], test_size=0.25, random_state=1)

# preprocess input
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

num_classes = 2 # 0, 1
# to_categorical shouldn't be strictly necessary
# if not using probably need to use sparse_categorical_crossentropy as loss fn.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

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
batch_size = 20
epochs = 2

# add later: perturb input

# from keras.preprocessing.image import ImageDataGenerator
# aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
#   height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#   horizontal_flip=True, fill_mode="nearest")
# H = model.fit_generator(
#   aug.flow(x_train, y_train, batch_size=batch_size),
#   validation_data=(x_test, y_test),
#   steps_per_epoch=len(x_train),
#   epochs=epochs, verbose=1)

H = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

if save_model:
# serialize model to JSON
  model_json = model.to_json()
  with open("model_action.json", "w") as json_file:
     json_file.write(model_json)
  model.save_weights("model_action.h5") # serialize weights
  print("Saved model to disk")

# add later: load saved model

# json_file = open('model_action.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model_action.h5")
# print("Loaded model from disk")

y_new = model.predict_classes(x_test)

print(y_test)
print(y_new)
