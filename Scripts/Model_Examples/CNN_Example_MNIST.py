#%%
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

#%%

# Load training and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Label single image for demonstration
single_image = x_train[0]
plt.imshow(single_image, 'gray_r')

#%%

# Reshape data to adhere to 1-hot encoding
y_categorical_test = to_categorical(y_test, 10)
y_categorical_train = to_categorical(y_train, 10)

#%%
# Normalise X data
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

# Reshape X data to add to colour channels
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#%%

#%%

# Build neural model
model = Sequential()

# Convolution layer
model.add(Conv2D(32, (4,4), input_shape=(28, 28, 1), activation='relu'))
# Pooling layer
model.add(MaxPool2D(pool_size=(2,2)))

# 2d ----> 1d
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation='relu'))

# Output
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

#%%
model.fit(x_train, y_categorical_train, epochs=2)

#%%

# Get name of metrics
model.metrics_names

# Use to appraise the new data
model.evaluate(x_test, y_categorical_test)

#%%

predictions = model.predict_classes(x_test)

#%%
print(classification_report(y_test, predictions))
