#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:23:15 2019

@author: waleedzafar
"""

# Handwritten Digit Classification


# Part 1 - Data Preprocessing and Visualization

# Importing the libraries

import pandas as pd # No need in this project as we are importing the dataset directly from the Keras API 
import numpy as np
import matplotlib.pyplot as plt

# Importing and downloading the MNIST dataset from the Keras API
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Visualizing the image of the handwritten digit
index = 10000  # Pick a value between 0 and 59999 to visualize a training image
plt.imshow(x_train[index], cmap = 'Greys')
print(y_train[index])

# Feature Scaling to normalize the RGB codes of the images
x_train = x_train / 255
x_test = x_test / 255

# Reshaping the array to 4 dimensions so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


# Part 2 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Layer 1 - Convolution
classifier.add(Conv2D(16, (3,3), input_shape = (28, 28, 1), activation = 'relu'))

# Layer 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Layer 3 - Flattening
classifier.add(Flatten())

# Layer 4 - Fully Connected Layer (i.e. Hidden Layer)
classifier.add(Dense(units = 128, activation = 'relu'))

# Layer 5 - Output Layer
classifier.add(Dense(units = 10, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# Part 3 - Fitting the CNN to the Training Set
classifier.fit(x_train, y_train, epochs = 5)


# Part 4 - Evaluating the CNN on the Test Set
print(classifier.evaluate(x_test, y_test))

# Part 5 - Predicting the label of a testing image and Comparing with actual label
index = 5555  # Pick a value between 0 and 9999 to visualize a test image and predict the digit on it
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
plt.imshow(x_test[index], cmap = 'Greys')
y_pred = classifier.predict(x_test[index].reshape(1,x_test.shape[1], x_test.shape[2], 1))
print('Predicted Digit = ', y_pred.argmax())
print('Actual Digit = ', y_test[index])

