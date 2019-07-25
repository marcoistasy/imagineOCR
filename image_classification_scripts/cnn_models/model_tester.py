#%%

import time
import os

import tensorflow as tf

from image_classification_scripts import data_processing
from image_classification_scripts.cnn_models import three_layer_cnn as cnn_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # This is a mac line that prevents 'initializing libiomp5.dylib, but found libiomp5.dylib already initialized'

# !!!!!!!! BODY !!!!!!!! #

NUMBER_OF_CLASSES = len(data_processing.classes_in_dataset())

image_data = data_processing.get_training_data_in_tensor_flow_format()['image_data']  # reference the images
label_data = data_processing.get_training_data_in_tensor_flow_format()['label_data']  # reference the respective
# label for each image

model = cnn_model.get_model(input_shape=image_data.shape, number_of_classes=NUMBER_OF_CLASSES)  # reference the model; pass in the
# number of classes in the data_processing module

# Save model to directory log to be able to retrieve it from tensorboard
NAME = '{}{}'.format(cnn_model.NAME, int(time.time()))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_logs/{}".format(NAME))

model.fit(image_data, label_data,
          batch_size=64,  # Batch size is how many things you wanna throw at once to the fitter
          validation_split=0.3,  # validation_split automatically splits up data into test and train
          epochs=50,
          callbacks=[tensorboard_callback])


