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
image = load_img()
display_img(image)

#%%
# Erosion: erode foreground/background distinction
kernel = np.ones((5,5), np.uint8)
result = cv2.erode(image, kernel, iterations=3)
display_img(result)

#%%
# Opening: Erosion followed by dilation (opposite of erosion) - helpful in removing background noise
white_noise = np.random.randint(0,2, (600,600))
white_noise = white_noise * 255

noisy_image = white_noise + image
display_img(noisy_image)

opening = cv2.morphologyEx(noisy_image, cv2.MORPH_OPEN, kernel)
display_img(opening)

#%%
# Closing: Helpful in removing foreground noise

black_noise = np.random.randint(0, 2, (600,600))
black_noise = black_noise * -255

black_noise_image = image + black_noise
black_noise_image[black_noise_image == -255] = 0
display_img(black_noise_image)

closing = cv2.morphologyEx(black_noise_image, cv2.MORPH_CLOSE, kernel)
display_img(closing)

#%%
# Morphological Gradient: takes the difference between erosion and dilation of image - really helpful in text as it is a reductive method of edge detection

gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)
