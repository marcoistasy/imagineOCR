# Imports

#%%
import cv2 as cv
from matplotlib import pyplot as plt
import os

# ret, thresholded_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)



#%%

DATA_DIRECTORY = '/Users/marcoistasy/Documents/Coding/imagine-ocr/object_detection/test_images'


for image in os.listdir(DATA_DIRECTORY):
    # iterate through each image and add them to training_data

    if image.lower().endswith(('.png', '.jpg', '.jpeg')):
        # skip all files that are not images

        try:

            image_as_array = cv.imread(os.path.join(DATA_DIRECTORY, image), cv.IMREAD_GRAYSCALE)
            ret, thresholded_image = cv.threshold(image_as_array, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            cv.imwrite('/Users/marcoistasy/Desktop/images/{}'.format(image), thresholded_image)

        except Exception as e:

            pass
