import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import *
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from ArrhythmiaDetecting import util, balanced_sampling as bs


def load_train_test_data():

    imbLearn = bs.balanced_Sampling()

    X_train, y_train, X_test, y_test = imbLearn.read_TrainingTesting_data()

    y_train = to_categorical(y_train)

    y_test = to_categorical(y_test)

    # print(y_test)

    X_train = np.expand_dims(X_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)


    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, y_train, X_test, y_test


def LongShortTermMemoryNeuralNetwork(X_train, y_train, X_test, y_test):

    # LSTM_model = Sequential()
    # LSTM_model.add(Embedding(max_features, output_dim=256))
    # LSTM_model.add(LSTM(128))
    # LSTM_model.add(Dropout(0.5))
    # LSTM_model.add(Dense(1, activation='sigmoid'))
    #
    # LSTM_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #
    # LSTM_model.fit(X_train, y_train, batch_size=16, epochs=10)

    LSTM_model = Sequential()
    LSTM_model.add(LSTM(120, input_shape=(60, 1), return_sequences=True))
    # LSTM_model.add(TimeDistributed(Dense(60, activation='tanh')))
    LSTM_model.add(LSTM(64, return_sequences=True))
    # LSTM_model.add(TimeDistributed(Dense(20, activation='tanh')))
    # LSTM_model.add(LSTM(10, return_sequences=True))
    # LSTM_model.add(TimeDistributed(Dense(20, activation='tanh')))
    LSTM_model.add(Flatten())
    LSTM_model.add(Dense(20, activation='relu'))
    LSTM_model.add(Dropout(0.20))
    LSTM_model.add(Dense(5, activation='softmax'))

    LSTM_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    LSTM_model.summary()

    result = LSTM_model.fit(X_train, y_train, batch_size= 32, epochs=10, verbose=1, validation_data=(X_test, y_test))


    # LSTM_model.save("./Model/LSTM_model_1.h5")

    util.plotAccuracyGraph(result)

    prediction = LSTM_model.predict(X_test)

    y_true, y_pred = util.get_prediction_Truth_value(prediction, y_test)

    # print (y_true)
    # print (y_pred)

    util.metricsMeasurement(y_true, y_pred)



X_train, y_train, X_test, y_test = load_train_test_data()
LongShortTermMemoryNeuralNetwork(X_train, y_train, X_test, y_test)

