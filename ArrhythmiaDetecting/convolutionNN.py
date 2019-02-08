import pandas as pd
from keras.layers import Convolution1D
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

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # print(y_test)

    X_train = np.expand_dims(X_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)

    return X_train, y_train, X_test, y_test



def ConvolutionNeuralNetwork(X_train, y_train, X_test, y_test):

    cnn = Sequential()

    cnn.add(Conv1D(38, 10, input_shape=(60,1), padding="same", activation="relu"))

    cnn.add(Conv1D(64, 10, padding='same', activation='relu'))

    cnn.add(MaxPooling1D())

    cnn.add(Conv1D(64, 10, padding='same', activation='relu'))

    cnn.add(Flatten())

    cnn.add(Dropout(0.25))

    cnn.add(Dense(5, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    cnn.summary()

    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, min_lr=0.0001)

    result = cnn.fit(X_train, y_train, epochs= 20, batch_size=16, verbose=1, validation_data=(X_test, y_test))

    # cnn.save("./Model/CNN_model_1.h5")

    util.plotAccuracyGraph(result)

    prediction = cnn.predict(X_test)

    y_true,y_pred = util.get_prediction_Truth_value(prediction, y_test)

    # print (y_true)
    # print (y_pred)

    util.metricsMeasurement(y_true,y_pred)



X_train, y_train, X_test, y_test = load_train_test_data()

ConvolutionNeuralNetwork(X_train, y_train, X_test, y_test)



# model = load_model("FFNN_model.h5")
# score = model.evaluate(X_test,y_test, verbose=0)
# print(score)