#%%
# -----IMPORTS-----

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils.im_utils import non_max_suppression


# -----FUNCTION-----

def decode_predictions(scores, geometry):
    # this function decodes the predictions given by EAST

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scores_data[x] < 0.1:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


# -----IMAGE PROCESSING-----

# load the input image and grab the image dimensions
image = cv.imread('/Users/marcoistasy/Documents/Coding/Cambridge_2019/imagine-ocr/object_detection/example_data/test/image1.jpg')
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (2560, 2560)
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# -----EAST-----

# define the two output layer names for the EAST detector model that
# we are interested in -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
net = cv.dnn.readNet('/Users/marcoistasy/Documents/Coding/Cambridge_2019/trained_models/frozen_east_text_detection.pb')

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# -----BOUNDING BOXES-----

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
    dX = int((endX - startX) * 0.1)  # todo change padding
    dY = int((endY - startY) * 0.1)  # todo change padding

    # apply padding to each side of the bounding box, respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # extract the actual padded ROI
    roi = orig[startY:endY, startX:endX]

    results.append((startX, startY, endX, endY))

# -----DISPLAY ROI ON IMAGE-----

# sort the results bounding box coordinates from top to bottom
# results = sorted(results, key=lambda r:r[0][1])

output = orig.copy()

# loop over the results
for (startX, startY, endX, endY) in results:
    # display the bounding boxes

    # using OpenCV, then draw the bounding box surrounding
    # the text region of the input image
    cv.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)

# show the output image
plt.imshow(output)
plt.show()
