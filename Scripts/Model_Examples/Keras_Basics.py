#%%
import numpy as np

from numpy import genfromtxt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

#%%

# Get data and create X, Y labels for convention
data = genfromtxt('/Users/marcoistasy/Documents/Coding/computerVision/Other/DATA/bank_note_data.txt', delimiter=',')

labels = data[:, 4]
features = data[:, 0:4]
X = features
y = labels

#%%

# Generate training data and scale it if necessary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler_object = MinMaxScaler()
scaler_object.fit(X_train)

scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

#%%

# Create the neural network

model = Sequential()

# Add layers to the neural networkd
model.add(Dense(4, input_dim = 4, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# Compile and train network
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
model.fit(scaled_X_train, y_train, epochs=100, verbose=2)

#%%
predicitons = model.predict_classes(scaled_X_test)
confusion_matrix(y_test, predicitons)

#%%
model.save('trainingModel.h5')

#%%
