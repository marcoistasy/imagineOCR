#%%
import numpy as np 
import cv2
import matplotlib.pyplot as plt

#%%
image = cv2.imread('Other/DATA/rainbow.jpg', 0)
plt.imshow(image, 'gray')

#%%
ret, thres1 = cv2.threshold(src=image, thresh=127, maxval=255, type=cv2.THRESH_TOZERO)

#%%
ret

#%%
plt.imshow(thres1, 'gray')

#%%
image = cv2.imread('Other/DATA/crossword.jpg', 0)
plt.imshow(image, 'gray')

#%%
def show_pic(image):
    figure = plt.figure(figsize= (15,15))
    ax = figure.add_subplot(111)
    ax.imshow(image, 'gray')

#%%
show_pic(image)
ret, thres1 = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
show_pic(thres1)


#%%

# Adaptive Thresholding
threshhold2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(threshhold2)

#%%
blended_image = cv2.addWeighted(thres1, 0.6, threshhold2, 0.4, 0)
show_pic(blended_image)

#%%
