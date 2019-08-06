#%%

# -----H:IMPORTS-----

import numpy as np
import cv2 as cv

from scripts.image_processing.utils.im_utils import non_max_suppression
from scripts.image_processing.utils.east_utils import decode_predictions
from scripts.image_processing.utils.east_utils import redundancy

# -----H:IMAGE PROCESSING-----

# load the input image and grab the image dimensions
image = cv.imread('/Users/marcoistasy/Documents/Coding/Cambridge_2019/imagine-ocr/object_detection/example_data/test/image1_original.jpg')
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (1280, 1280)  # todo this can be changed to any multiple of 32
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# -----H:EAST-----

# define the two output layer names for the EAST detector model that
# we are interested in -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
net = cv.dnn.readNet('/Users/marcoistasy/Documents/Coding/Cambridge_2019/EAST/frozen_east_text_detection.pb')
# todo change location of EAST

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probabilities=confidences)

# -----H:BOUNDING BOXES-----

# initialize the list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    dX = int((endX - startX) * 0.25)  # todo change padding
    dY = int((endY - startY) * 0.05)  # todo change padding

    # apply padding to each side of the bounding box, respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # extract the actual padded ROI
    roi = orig[startY:endY, startX:endX]

    # chances that all text will be extracted is low - perform another check with already extracted text erased
    redundancy()

    results.append((startX, startY, endX, endY))

# -----H:DISPLAY ROI ON IMAGE-----

# sort the results bounding box coordinates from top to bottom

# todo organise output by order

output = orig.copy()

# loop over the results
for (startX, startY, endX, endY) in results:
    # display the bounding boxes

    # using OpenCV, then draw the bounding box surrounding
    # the text region of the input image
    cv.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)

# show the output image
cv.imwrite('/Users/marcoistasy/Documents/Coding/Cambridge_2019/imagine-ocr/object_detection/example_data/test/EAST_test_image.jpg', output)
