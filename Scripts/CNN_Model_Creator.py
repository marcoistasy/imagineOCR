#%% Imports

import time

import tensorflow as tf
import numpy as np

from Scripts.Data_Preprocessing import get_training_data

#%% ----- SET UP TRAINING DATA INTO TENSORFLOW FORMAT -------

image_data = []
label_date = []

for feature, label in get_training_data():

    # Pass the data into the respective feature and label array
    image_data.append(feature)
    label_date.append(label)

image_data = np.array(image_data).reshape((-1, 50, 50, 1))  # reshape feature to adhere to tensorflow

image_data = image_data / 255  # Normalise feature data -> check out keras.utils.normalise


#%% ----- CREATE AND COMPILE TENSORFLOW MODEL -------

model = tf.keras.models.Sequential([

  # First layer
  tf.keras.layers.Conv2D(256, (3, 3), input_shape=image_data.shape[1:], activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  # Second layer
  tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  # Third layer
  tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  # Fourth layer
  tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  tf.keras.layers.Flatten(),  # this converts our 3D feature maps to 1D feature vectors

  # Fifth layer
  tf.keras.layers.Dense(64),

  # tf.keras.layers.Dropout(0.5),  # Help reduce over-fitting by randomly turning neurons off during training.

  # Output layer
  tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#%% ----- FIT MODEL TO TRAINING DATA -------

# Save model to directory log to be able to retrieve it from tensorboard
NAME = '{}'.format(int(time.time()))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_logs/{}".format(NAME))

model.fit(image_data,
          label_date,
          batch_size=64,  # Batch size is how many things you wanna throw at once to the fitter
          validation_split=0.3,  # validation_split automatically splits up data into test and train
          epochs=40,
          callbacks=[tensorboard_callback])
