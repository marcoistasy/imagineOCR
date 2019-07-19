#%%

import numpy as np
import cv2
import os
import random

# Neural networks
import tensorflow as tf

#%%

# Path to the directory that holds the data
data_directory = '/Users/marcoistasy/Desktop/PetImages'

# Categories in the data
categories = ['Dog', 'Cat']

# Resize data so that it all is the same -> this is variable and can be changed based on hindsight
resize_size = (50, 50)

# Create training data
training_data = []


def create_training_data():
    for category in categories:

        path = os.path.join(data_directory, category)  # path to cats or dogs directory
        class_as_number = categories.index(category)  # get the category based on index in array

        for image in os.listdir(path):

            try:
                image_as_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)  # Convert image to array in
                # gray-scale -> can try to tweak this
                resized_image_as_array = cv2.resize(image_as_array, resize_size)
                training_data.append([resized_image_as_array, class_as_number])  # add the data to the created list: each
                # with a class
            except Exception as e:
                pass


create_training_data()

# Shuffle the training data to confuse the neural network
random.shuffle(training_data)

#%%

feature_data = []
label_date = []

# Pass the data into the respective feature and label array
for feature, label in training_data:
    feature_data.append(feature)
    label_date.append(label)

# Convert feature
feature_data = np.array(feature_data).reshape(-1, 50, 50, 1)

#%%

# Normalise feature data -> check out keras.utils.normalise
feature_data = feature_data / 255

# Create model

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), input_shape=feature_data.shape[1:], activation=tf.nn.relu),  # skip the first part
    # because its -1
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),

    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

])

#%%

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # I think this should be changed if dealing with more than two items
              metrics=['accuracy'])

#%%
model.fit(feature_data, label_date, batch_size=128, validation_split=0.1, epochs=3)  # validation_split automatically
# splits up data into test and train
# Batch size is how many things you wanna throw at once to the fitter


#%%
