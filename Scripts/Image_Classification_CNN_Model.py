#%% Imports

import os
import random
import cv2
import logging
import time

import tensorflow as tf
import numpy as np

from Scripts import Image_Preprocessing

import matplotlib.pyplot as plt

#%%

RESIZE_SIZE = (50, 50)  # Resize data so that it all is the same


def get_training_data():
    # Function that will append training data to above property

    data_directory = 'DATA/Model_Data'  # Path to the directory that holds the data

    categories = ['a', 'e', 'é', 'i', 'o', 'u']  # Categories in the data

    training_data = []  # property to hold training data

    for category in categories:
        # Iterate through the categories and link them to an index

        path = os.path.join(data_directory, category)  # path to letters directory

        class_as_number = categories.index(category)  # get the category based on index in array

        for image in os.listdir(path):
            # iterate through each image and add them to training_data

            if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                # skip all files that are not images: sometimes mac saves .DS_store

                try:

                    image_as_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)  # Convert to array
                    # in gray-scale -> can try to tweak this
                    resized_image_as_array = cv2.resize(image_as_array, RESIZE_SIZE)  # ensure uniformity
                    training_data.append([resized_image_as_array, class_as_number])  # add data to the list: each
                    # with a class

                    # x = Image_Preprocessing.generate_data([resized_image_as_array, class_as_number])

                except Exception as e:

                    logging.error('Could not load {}'.format(os.path.join(path, image)), exc_info=e)

    random.shuffle(training_data)  # Shuffle the training data to confuse the neural network

    return training_data


print(len(get_training_data()))

#%%

image_data = []
label_date = []

for feature, label in get_training_data():

    # Pass the data into the respective feature and label array
    image_data.append(feature)
    label_date.append(label)

image_data = np.array(image_data).reshape((-1, 50, 50, 1))  # reshape feature to adhere to tensorflow

image_data = image_data / 255  # Normalise feature data -> check out keras.utils.normalise


#%% Create and compile tensorflow model

model = tf.keras.models.Sequential([

  # First layer
  tf.keras.layers.Conv2D(256, (3, 3), input_shape=image_data.shape[1:], activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  # Second layer
  tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  tf.keras.layers.Flatten(),  # this converts our 3D feature maps to 1D feature vectors

  # Third layer
  tf.keras.layers.Dense(64),

  # Output layer
  tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # I think this should be changed if dealing with more than two
              # items
              metrics=['accuracy'])


#%% Fit model to training data

# Save model to directory log to be able to retrieve it from tensorboard
NAME = '{}'.format(int(time.time()))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_logs/{}".format(NAME))

model.fit(image_data, label_date, batch_size=32, validation_split=0.3, epochs=500, callbacks=[tensorboard_callback])
# validation_split automatically splits up data into test and train
# Batch size is how many things you wanna throw at once to the fitter
