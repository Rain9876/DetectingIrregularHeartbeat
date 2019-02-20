## Author: Yurun SONG
## Date: 2019/01/30
## Project: Deep Learning about detecting irregular heartbeat
#
# Construct the CNN Neural Network model
# Training, testing and evaluation
#

import pandas as pd
from keras.layers import Convolution1D
from keras.models import Sequential, load_model
from keras.layers import *
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import util, balanced_sampling as bs


## Load training and testing data
def load_train_test_data():

    imbLearn = bs.balanced_Sampling(60)

    X_train, y_train, X_test, y_test = imbLearn.read_TrainingTesting_data()

    y_train = to_categorical(y_train)

    y_test = to_categorical(y_test)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    X_train = np.expand_dims(X_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)

    return X_train, y_train, X_test, y_test


## Construct the Convolution Neural Network Model
def ConvolutionNeuralNetwork(X_train, y_train, X_test, y_test):

    cnn = Sequential()

    cnn.add(Conv1D(30, 5, input_shape=(60,1), padding="same", activation="relu"))

    cnn.add(MaxPooling1D())

    cnn.add(Conv1D(60, 5, padding='same', activation='relu'))

    cnn.add(MaxPooling1D())

    cnn.add(Flatten())

    cnn.add(Dropout(0.5))

    cnn.add(Dense(720, activation='relu'))

    cnn.add(Dense(5, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    cnn.summary()

    result = cnn.fit(X_train, y_train, epochs= 50, batch_size=16, verbose=1, validation_data=(X_test, y_test))

    # cnn.save("./Model/CNN_model_4.h5")

    util.plotAccuracyGraph(result)

    prediction = cnn.predict(X_test)

    y_true,y_pred = util.get_prediction_Truth_value(prediction, y_test)

    util.metricsMeasurement(y_true,y_pred)

