#%%

# Imports

import cv2
import matplotlib.pyplot as plt

from keras_preprocessing.image import ImageDataGenerator

#%%

# Generate images from already received images
image_generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, rescale=1/255,
                                     shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

image = cv2.imread('Test_Image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image_generator.random_transform(image)
plt.imshow(image)
