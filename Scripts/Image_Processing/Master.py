#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%

# Opening
kernel = np.ones((5, 5), np.uint8)
image = cv2.imread('Test_Image.jpg')
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


#%%
