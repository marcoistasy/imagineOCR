#%% Imports

import os
import random
import cv2
import logging

import numpy as np

from image_classification_scripts import image_preprocessing

DATA_DIRECTORY = 'DATA/Model_Data'  # Path to the directory that holds the data


#%%

# Public Functions

def classes_in_dataset():
    # Function to get all classes (i.e directories) in the Model_Data directory

    paths_to_directories = []  # variable to hold all raw paths
    classes = []  # variable to hold all classes

    # append the directory path to the classes variable
    for directory in os.walk(DATA_DIRECTORY):
        paths_to_directories.append(directory[0])

    # remove reference to self directory in classes variable
    if DATA_DIRECTORY in paths_to_directories:
        paths_to_directories.remove(DATA_DIRECTORY)

    # get the last letter from each path
    for path_to_directory in paths_to_directories:
        x = path_to_directory.split(DATA_DIRECTORY, 1)[1]  # remove path
        y = x.split('/', 1)[1]  # remove path
        classes.append(y)

    return classes


def get_training_data_in_tensor_flow_format():
    # Get the training data into two lists suitable to pass into the neural network

    image_data = []  # Property to hold the image data
    label_data = []  # Property to hold the label data for each respective image

    for feature, label in __retrieve_training_data_from_directory():
        # Pass the data into the respective feature and label array
        image_data.append(feature)
        label_data.append(label)

    image_data = np.array(image_data).reshape((-1, 50, 50, 1))  # reshape feature to adhere to tensorflow

    image_data = image_data / 255  # Normalise feature data -> check out keras.utils.normalise

    dictionary_to_return = {
        'image_data': image_data,
        'label_data': label_data
    }

    return dictionary_to_return  # Return a dictionary with values for the images and labels


# Private Functions

def __retrieve_training_data_from_directory():
    # Function that will get user training data and return a property of it

    training_data = []  # property to hold training data

    resize_size = (50, 50)  # Resize data so that it all is the same

    for category in classes_in_dataset():
        # Iterate through the categories and link them to an index

        path = os.path.join(DATA_DIRECTORY, category)  # path to letters directory

        class_as_number = classes_in_dataset().index(category)  # get the category based on index in array

        for image in os.listdir(path):
            # iterate through each image and add them to training_data

            if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                # skip all files that are not images

                try:

                    image_as_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)  # Convert to array
                    # in gray-scale -> can try to tweak this
                    resized_image_as_array = cv2.resize(image_as_array, resize_size)  # ensure uniformity
                    training_data.append([resized_image_as_array, class_as_number])  # add data to the list: each
                    # with a class

                    training_data.extend(image_preprocessing.generate_artificial_training_data(resized_image_as_array, class_as_number))

                except Exception as e:

                    logging.error('Could not load {}'.format(os.path.join(path, image)), exc_info=e)

    random.shuffle(training_data)  # Shuffle the training data to confuse the neural network

    return training_data
