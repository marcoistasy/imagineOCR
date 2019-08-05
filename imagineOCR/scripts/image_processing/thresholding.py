# Imports
import cv2 as cv


#%%

def apply_sobel_x(image):
    # applies a sobel x gradient on a passed image with a specific kernel size
    return cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)


def apply_sobel_y(image):
    # applies a sobel y gradient on a passed image with a specific kernel size
    return cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)


def apply_otsu_binarization(image):
    # applies an Otsu's binarization on a passed image
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return image


def write_new_image(from_path):
    image = cv.imread(from_path)
    new_image = apply_otsu_binarization(image)
    cv.imwrite('/Users/marcoistasy/Documents/Coding/Cambridge_2019/imagine-ocr/object_detection/example_data/test/sobelx', new_image)
