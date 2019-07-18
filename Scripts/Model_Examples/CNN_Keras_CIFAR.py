#%%
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import load_model

from sklearn.metrics import classification_report

#%%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#%%

# ! Preprocessing

# Normalise x values
x_train = x_train / 255
x_test = x_test / 255

# 1-hot code y values
y_categorical_train = to_categorical(y_train, 10)
y_categorical_test = to_categorical(y_test, 10)

#%%

# ! Network construction

model = Sequential()

# Create layers

convolutional_layer = Conv2D(filters=32, kernel_size= (4, 4), input_shape = (32, 32, 3), activation = 'relu')

max_pooling_layer = MaxPool2D(pool_size = (2, 2))

second_convolutional_layer = Conv2D(filters=32, kernel_size= (4, 4), input_shape = (32, 32, 3), activation = 'relu')

second_max_pooling_layer = MaxPool2D(pool_size = (2, 2))

flatten_layer = Flatten()

# 128, 256, 512
dense_layer = Dense(256, activation = 'relu')

classifier_layer = Dense(10, activation = 'softmax')

layers = [convolutional_layer, max_pooling_layer, second_convolutional_layer, second_max_pooling_layer, flatten_layer, dense_layer, classifier_layer]

for layer in layers:
    model.add(layer)

model.compile(loss ='categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

#%%
model.summary()

#%%

# ! This took too long - I downloaded the model and loaded it below
model.fit(x_train, y_categorical_train, verbose=1, epochs=10)

#%%

#
model = load_model('Scripts/Model_Examples/cifar10.h5')


#%%
model.metrics_names
model.evaluate(x_test, y_categorical_test)

#%%
predictions = model.predict_classes(x_test)


#%%
print(classification_report(y_test, predictions))

#%%
