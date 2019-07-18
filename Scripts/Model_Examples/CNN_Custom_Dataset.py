#%%
import matplotlib.pyplot as plt 
import numpy as np
import cv2

from keras.preprocessing.image import ImageDataGenerator # allows to create other images from the test
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.preprocessing import image

#%%

# Image tests

cat_four = cv2.imread('CATS_DOGS/train/CAT/4.jpg')
cat_four = cv2.cvtColor(cat_four, cv2.COLOR_BGR2RGB)
plt.imshow(cat_four)

dog_two = cv2.imread('CATS_DOGS/train/DOG/2.jpg')
dog_two = cv2.cvtColor(dog_two, cv2.COLOR_BGR2RGB)
plt.imshow(dog_two)

#%%

# Image generator: will apply transformations to images

image_generator = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1, height_shift_range = 0.1, rescale = 1/255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')

plt.imshow(image_generator.random_transform(dog_two))

#%%

input_shape = (150, 150, 3)

# Build model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size = (3,3), input_shape = input_shape, activation ='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters=64, kernel_size = (3,3), input_shape = input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters=64, kernel_size = (3,3), input_shape = input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))

# Dropout layer - helps reduce overfilling
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#%%

# Create images from training data
batch_size = 16

training_image_generator = image_generator.flow_from_directory('CATS_DOGS/train', target_size = input_shape[:2], batch_size = batch_size, class_mode = 'binary')

test_image_generator = image_generator.flow_from_directory('CATS_DOGS/test', target_size = input_shape[:2], batch_size = batch_size, class_mode = 'binary')

#%%

# ! Took to long to build model
# model.fit_generator(training_image_generator, epochs = 100, steps_per_epoch = 150, validation_data = test_image_generator, validation_steps = 12)

model = load_model('Scripts/Model_Examples/cat_dog_100epochs.h5')

#%%
dog_file = 'CATS_DOGS/train/DOG/2.jpg'
dog_image = image.load_img(dog_file, target_size=(150, 150))
dog_image = image.img_to_array(dog_image)
dog_image = np.expand_dims(dog_image, 0)
#%%
dog_image = dog_image / 255
model.predict_classes(dog_image)


#%%
