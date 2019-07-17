#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%%

def load_image():
    img = cv2.imread('Other/DATA/bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def display_image(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)

#%%
display_image(load_image())

#%%

# Increase or decrease brightness
gamma = 2 
result = np.power(load_image(), gamma)
display_image(result)

#%%

image = cv2.putText(load_image(), 'Bricks', (10,600), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 0, 0), 4)
display_image(image)

#%%

kernel = np.ones((5,5), np.float32)/25
kernel

#%%
dst = cv2.filter2D(image, -1, kernel)
display_image(dst)


#%%

image = cv2.putText(load_image(), 'Bricks', (10,600), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 0, 0), 4)
display_image(image)


#%%
blurred = cv2.blur(image, (10,5))
display_image(blurred)

#%%
image = cv2.putText(load_image(), 'Bricks', (10,600), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 0, 0), 4)
display_image(image)

#%%
blurred_image = cv2.GaussianBlur(image, (5,5), 10)
display_image(blurred_image)

#%%
image = cv2.putText(load_image(), 'Bricks', (10,600), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 0, 0), 4)
display_image(image)

#%%
# Good for removing noise from image but while keeping details in check
display_image(cv2.medianBlur(image, 5))


#%%
image = cv2.imread('Other/DATA/sammy.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
display_image(image)

#%%
noiseImage = cv2.imread('Other/DATA/sammy_noise.jpg')
display_image(noiseImage)

#%%
median = cv2.medianBlur(noiseImage, 5)
display_image(median)

#%%
image = cv2.putText(load_image(), 'Bricks', (10,600), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 0, 0), 4)
display_image(image)


#%%
blur = cv2.bilateralFilter(image, 9, 75, 75)
display_image(blur)

#%%
