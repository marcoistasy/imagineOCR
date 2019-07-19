#%%

# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator


#%%
def correct_colour(image_to_correct):
    return cv2.cvtColor(image_to_correct, cv2.COLOR_BGR2RGB)


#%%
letter_e = cv2.imread('DATA/Model_Data/train/e/1.png')
plt.imshow(correct_colour(letter_e))
plt.show()

image_generator = ImageDataGenerator(brightness_range=[1.0, 1.0])
image = image_generator.random_transform(correct_colour(letter_e))
plt.imshow(image)
plt.show()