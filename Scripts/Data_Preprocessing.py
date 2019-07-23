
# Imports

import os
import random
import cv2
import logging

from Scripts import Image_Preprocessing


# Main Function

def get_training_data():
    # Function that will append training data to above property

    data_directory = 'DATA/Model_Data'  # Path to the directory that holds the data

    categories = ['a', 'e', 'é', 'i', 'o', 'u']  # Categories in the data

    training_data = []  # property to hold training data

    resize_size = (50, 50)  # Resize data so that it all is the same

    for category in categories:
        # Iterate through the categories and link them to an index

        path = os.path.join(data_directory, category)  # path to letters directory

        class_as_number = categories.index(category)  # get the category based on index in array

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

                    training_data.extend(Image_Preprocessing.generate_data(resized_image_as_array, class_as_number))

                except Exception as e:

                    logging.error('Could not load {}'.format(os.path.join(path, image)), exc_info=e)

    random.shuffle(training_data)  # Shuffle the training data to confuse the neural network

    return training_data
