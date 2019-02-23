## Author: Yurun SONG
## Date: 2019/01/30
## Project: Deep Learning about detecting irregular heartbeat
#
# Construct the FFNN Neural Network model
# Training, testing and evaluation
#

from keras.models import Sequential,load_model
from keras.layers import *
from keras.utils import to_categorical
from sklearn.model_selection import KFold,StratifiedKFold
import util, balanced_sampling as bs
from sklearn.preprocessing import MinMaxScaler



## Load training and testing data
def load_train_test_data():

    imbLearn = bs.balanced_Sampling(60)

    X_train, y_train, X_test, y_test = imbLearn.read_TrainingTesting_data()

    X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 1)

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

    # md.save("./Model/MLII/FFNN_model_1.h5")

    util.plotAccuracyGraph(history)

    prediction = md.predict(X_test)

    y_true,y_pred = util.get_prediction_Truth_value(prediction, y_test)

    accuracy = util.metricsMeasurement(y_true,y_pred)

    return accuracy
