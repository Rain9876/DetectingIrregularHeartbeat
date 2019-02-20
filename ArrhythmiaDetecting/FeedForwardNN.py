## Author: Yurun SONG
## Date: 2019/01/30
## Project: Deep Learning about detecting irregular heartbeat
#
# Construct the FFNN Neural Network model
# Training, testing and evaluation
#

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.optimizers import SGD,adamax
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import util, balanced_sampling as bs
from imblearn.keras import BalancedBatchGenerator
from imblearn.combine import SMOTEENN
from collections import Counter


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

    return X_train, y_train, X_test, y_test


## Construct the Feed Forward Neural Network Model
def FeedForwardNeuralNetwork(X_train, y_train, X_test, y_test):

    md = Sequential()

    md.add(Dense(120, input_shape = (60,), activation = 'relu'))

    md.add(Dense(80, activation = 'relu'))

    md.add(Dropout(0.50))

    md.add(Dense(50, activation = 'relu'))

    md.add(Dense(5, activation = 'softmax'))

    md.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

    md.summary()

    print(X_train.shape)
    print(y_train.shape)

    history = md.fit(X_train, y_train, epochs = 200, validation_data = (X_test, y_test),shuffle=True, verbose=2)

    # md.save("./Model/FFNN_model_4.h5")

    util.plotAccuracyGraph(history)

    prediction = md.predict(X_test)

    y_true,y_pred = util.get_prediction_Truth_value(prediction, y_test)

    util.metricsMeasurement(y_true,y_pred)
