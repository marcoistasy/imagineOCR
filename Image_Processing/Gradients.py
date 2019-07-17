#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')

#%%
def load_img():
    blank_img =np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img,text='ABCDE',org=(50,300), fontFace=font,fontScale= 5,color=(255,255,255),thickness=25,lineType=cv2.LINE_AA)
    return blank_img

#%%
image = cv2.imread('Other/Data/sudoku.jpg', 0)
display_img(image)

#%%
# Sobel X Gradient
sobelx = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
display_img(sobelx)

#%%
# Sobel Y Gradient
sobely = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
display_img(sobely)

#%%
# Laplace Operator
laplacian = cv2.Laplacian(image, cv2.CV_64F)
display_img(laplacian)

#%%
blended = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
display_img(blended)

#%%
ret, th1 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
display_img(th1)


#%%
kernel = np.ones((4,4), np.uint8) 
gradient = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)


#%%
