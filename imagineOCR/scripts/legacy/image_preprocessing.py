# Imports

import cv2 as cv
import numpy as np

from keras_preprocessing.image import ImageDataGenerator


# Public Functions

def generate_artificial_training_data(image, class_as_number):
    # Amplify the original sample size by applying several transformations to the original sample

    kernel = np.ones((5, 5), np.float32)

    generated_data = []  # property to hold generated data

    # Get several new iterations of the same image

    ret, thresholded_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)  # Applies adaptive
    # threshold
    median_blurred_image = cv.medianBlur(image, 5)  # Applies median blurring
    bilateral_filtered_image = cv.bilateralFilter(image, 9, 75, 75)  # Applies bilateral filtering
    guassian_blurred_image = cv.GaussianBlur(image, (11, 11), 10)  # Applies Gaussian blurring
    gamma_changed_image = np.power(image, 0.2)  # Applies gamma change
    eroded_image = cv.erode(image, kernel, iterations=2)  # Applies erosion
    dilated_image = cv.dilate(image, kernel, iterations=1)  # Applies dilation
    closed_image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)  # Applies morphological closing
    graded_image = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)  # Applies morphological gradient

    # Apply transformations to the image

    rotated_image = __apply_transformation(image, ImageDataGenerator(rotation_range=45))  # Rotates the image
    x_shifted_image = __apply_transformation(image, ImageDataGenerator(width_shift_range=0.1))  # Shifts the image left
    # and right ward
    y_shifted_image = __apply_transformation(image, ImageDataGenerator(height_shift_range=0.1))  # Shifts the image up
    # and down ward
    zoomed_image = __apply_transformation(image, ImageDataGenerator(zoom_range=0.4))  # Zooms in on the image
    noisy_image = (np.random.randint(low=0, high=2, size=image.shape)*255) + image

    new_images = [thresholded_image, median_blurred_image, bilateral_filtered_image,
                  guassian_blurred_image, gamma_changed_image, eroded_image, dilated_image,
                  closed_image, graded_image, rotated_image, x_shifted_image, y_shifted_image,
                  zoomed_image, noisy_image]  # A reference for all the new images

    for new_image in new_images:
        # Append each image to the generated_data property with its respective label
        generated_data.append([new_image, class_as_number])  # add data to the list: each

    return generated_data


# Private Functions

def __apply_transformation(image, generator: ImageDataGenerator):
    # Turn the read image into colour so that the ImageDataGenerator can apply its transformation. Then return the
    # image gray-scaled

    coloured_image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)  # Transform the gray-scaled image into colour
    transformed_image = generator.random_transform(coloured_image)  # Transform the image with the ImageDataGenerator
    gray_scaled_image = cv.cvtColor(transformed_image, cv.COLOR_RGB2GRAY)  # Revert the coloured image back to grey

    return gray_scaled_image
