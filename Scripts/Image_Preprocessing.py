
# Imports

import cv2

import numpy as np

from keras_preprocessing.image import ImageDataGenerator


# Main Function

def generate_data(image, class_as_number):
    # Amplify the original sample size by applying several transformations to the original sample

    kernel = np.ones((5, 5), np.float32)

    generated_data = []  # property to hold generated data

    # Get several new iterations of the same image

    ret, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Applies adaptive
    # threshold
    median_blurred_image = cv2.medianBlur(image, 5)  # Applies median blurring
    bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)  # Applies bilateral filtering
    guassian_blurred_image = cv2.GaussianBlur(image, (11, 11), 10)  # Applies Gaussian blurring
    gamma_changed_image = np.power(image, 0.2)  # Applies gamma change
    eroded_image = cv2.erode(image, kernel, iterations=2)  # Applies erosion
    dilated_image = cv2.dilate(image, kernel, iterations=1)  # Applies dilation
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Applies morphological closing
    graded_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)  # Applies morphological gradient

    # Apply transformations to the image

    # TODO: Find a way to make these a np.array
    # rotated_image = ImageDataGenerator(rotation_range=45)  # Rotates the image
    # x_shifted_image = ImageDataGenerator(width_shift_range=0.1)  # Shifts the image left and right ward
    # y_shifted_image = ImageDataGenerator(height_shift_range=0.1)  # Shifts the image up and down ward
    # zoomed_image = ImageDataGenerator(zoom_range=0.4)  # Zooms in on the image
    noisy_image = (np.random.randint(low=0, high=2, size=image.shape)*255) + image

    new_images = [thresholded_image, median_blurred_image, bilateral_filtered_image, guassian_blurred_image,
                  gamma_changed_image, eroded_image, dilated_image, closed_image, graded_image, noisy_image]  # A reference for all the new images

    for new_image in new_images:
        # Append each image to the generated_data property with its respective label
        generated_data.append([new_image, class_as_number])  # add data to the list: each

    return generated_data
