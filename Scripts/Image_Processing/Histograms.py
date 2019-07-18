#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%

dark_horse = cv2.imread('Other/DATA/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)
rainbow = cv2.imread('Other/DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('Other/DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)


#%%
hist_values = cv2.calcHist([blue_bricks], channels=[0], mask= None, histSize=[256], ranges=[0, 256])
plt.plot(hist_values)

#%%
hist_values = hist_values = cv2.calcHist([dark_horse], channels=[0], mask= None, histSize=[256], ranges=[0, 256])
plt.plot(hist_values)

#%%

image = dark_horse

#%%

color = ('b', 'g', 'r')

#%%
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 50])
    plt.ylim([0, 500000])

plt.title('Histogram')

#%%
####### COLOUR EQUALISATION ###########
rainbow = cv2.imread('Other/DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)
image = rainbow

#%%
mask = np.zeros(image.shape[:2], np.uint8)
mask[300:400, 100:400] = 255
plt.imshow(mask,cmap='gray')


#%%
masked_img = cv2.bitwise_and(image,image,mask = mask)
show_masked_img = cv2.bitwise_and(show_rainbow,show_rainbow,mask = mask)

#%%
plt.imshow(show_masked_img)


#%%
hist_mask_values_red = cv2.calcHist([rainbow], [2], mask, [256], [0, 256]) 
hist_values_red = cv2.calcHist([rainbow], [2], None, [256], [0, 256]) 

#%%
plt.plot(hist_mask_values_red)
plt.title('Masked')

#%%
plt.plot(hist_values_red)
plt.title('Masked')

#%%
gorilla = cv2.imread('Other/DATA/gorilla.jpg', 0)
type(gorilla)

#%%
def display(img, cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)

#%%
display(gorilla, cmap='gray')

#%%
histogram_values = cv2.calcHist([gorilla], [0], None, [256], [0, 256])


#%%
plt.plot(histogram_values)

#%%
equalised_gorrilla = cv2.equalizeHist(gorilla)
display(equalised_gorrilla, 'gray')

#%%
histogram_values = cv2.calcHist([equalised_gorrilla], [0], None, [256], [0, 256])
plt.plot(histogram_values)


#%%
colour_gorilla = cv2.imread('Other/DATA/gorilla.jpg')
show_gorilla = cv2.cvtColor(colour_gorilla, cv2.COLOR_BGR2RGB)


#%%
display(show_gorilla)

#%%
hsv_gorrilla = cv2.cvtColor(colour_gorilla, cv2.COLOR_BGR2HSV)
hsv_gorrilla[:,:,2]

#%%
hsv_gorrilla[:,:,2] = cv2.equalizeHist(hsv_gorrilla[:,:,2])


#%%
equalised_colour_gorilla = cv2.cvtColor(hsv_gorrilla, cv2.COLOR_HSV2RGB)
display(equalised_colour_gorilla)

#%%
