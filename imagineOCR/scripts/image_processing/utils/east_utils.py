# -----H:IMPORTS-----

import numpy as np


# -----H:FUNCTIONS-----

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
            if scores_data[x] < 0.1:  # todo change confidence level
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


def redundancy():
    # todo implement redundancy
    pass
