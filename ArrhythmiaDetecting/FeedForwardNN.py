import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.optimizers import SGD,adamax
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from ArrhythmiaDetecting import util, balanced_sampling as bs
from imblearn.keras import BalancedBatchGenerator
from imblearn.combine import SMOTEENN
from collections import Counter



def load_train_test_data():

    imbLearn = bs.balanced_Sampling()

    X_train, y_train, X_test, y_test = imbLearn.read_TrainingTesting_data()

    y_train = to_categorical(y_train)

    y_test = to_categorical(y_test)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # print(y_test)

    return X_train, y_train, X_test, y_test


def FeedForwardNeuralNetwork(X_train, y_train, X_test, y_test):

    md = Sequential()
    md.add(Dense(120, input_shape = (60,), activation = 'relu'))
    md.add(Dense(80, activation = 'tanh'))
    md.add(Dropout(0.50))
    md.add(Dense(60, activation = 'tanh'))
    md.add(Dense(5, activation = 'softmax'))

    # opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    # amax = adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    md.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

    md.summary()

    print(X_train.shape)
    print(y_train.shape)

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)

    # training_generator = BalancedBatchGenerator(X_train, y_train, batch_size=32, random_state=42)
    # callback_history = md.fit_generator(generator=training_generator, steps_per_epoch=100, epochs=100, verbose=0)

    history = md.fit(X_train, y_train, epochs = 100, validation_data = (X_test, y_test),shuffle=True, verbose=2)


    # md.save("./Model/FFNN_model_30_1.h5")


    util.plotAccuracyGraph(history)

    prediction = md.predict(X_test)

    y_true,y_pred = util.get_prediction_Truth_value(prediction, y_test)

    # print (y_true)
    # print (y_pred)

    util.metricsMeasurement(y_true,y_pred)




# X_train, y_train, X_test, y_test = load_train_test_data()
#
# FeedForwardNeuralNetwork(X_train, y_train, X_test, y_test)

