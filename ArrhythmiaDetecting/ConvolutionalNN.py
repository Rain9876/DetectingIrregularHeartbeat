## Author: Yurun SONG
## Date: 2019/01/30
## Project: Deep Learning about detecting irregular heartbeat
#
# Construct the CNN Neural Network model
# Training, testing and evaluation
#

from keras.models import Sequential, load_model
from keras.layers import *
from keras.utils import to_categorical
import numpy as np
import util, balanced_sampling as bs


## Load training and testing data
def load_train_test_data():

    imbLearn = bs.balanced_Sampling(60)

    X_train, y_train, X_test, y_test = imbLearn.read_TrainingTesting_data()

    X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 2)

    return X_train, y_train, X_test, y_test


## Construct the Convolutional Neural Network Model
def ConvolutionalNeuralNetwork(X_train, y_train, X_test, y_test):

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

    # cnn.save("./Model/MLII/CNN_model_1.h5")

    util.plotAccuracyGraph(result)

    prediction = cnn.predict(X_test)

    y_true,y_pred = util.get_prediction_Truth_value(prediction, y_test)

    accuracy = util.metricsMeasurement(y_true,y_pred)

    return accuracy
