#
## Author: Yurun SONG
## Date: 2019/01/15
## Project: Deep Learning about detecting irregular heartbeat
#
# Model_testing is used to load the stored model and tests the accuracy.
# K fold cross validation to get the average accuracy of the model.
#


from keras.models import load_model
from sklearn.model_selection import KFold,StratifiedKFold
import util, balanced_sampling as bs,signalDataProcessing as sdp, \
    FeedForwardNN as FFNN, ConvolutionalNN as CNN, LongShortTermMemoryNN as LSTM



# Loading training and testing data
def load_train_test_data(input_shape):

    imbLearn = bs.balanced_Sampling(input_shape)

    X_signal, Y_label = imbLearn.featureExtract()

    X_data, Y_data = imbLearn.balanceSamples(X_signal, Y_label,20000)

    X_train,X_test,y_train,y_test = imbLearn.train_test_data(X_data,Y_data)

    # imbLearn.write_TrainingTesting_toCSV(X_train,y_train,X_test,y_test)
    # X_train, y_train, X_test, y_test = imbLearn.read_TrainingTesting_data()

    return X_train, y_train, X_test, y_test


# Tesing Feed Forward Neural Network Model
def FFNN_Model_Testing():

    print("Feed Forward Neural Network Results:")

    X_train, y_train, X_test, y_test = load_train_test_data(60)

    X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 1)

    md = load_model("./Model/MLII/FFNN_model_2.h5")

    prediction = md.predict(X_test)

    y_true, y_pred = util.get_prediction_Truth_value(prediction, y_test)

    util.metricsMeasurement(y_true, y_pred)



# Tesing Convolutional Neural Network Model
def CNN_Model_Testing():

    print("Convolution Neural Network Results:")

    X_train, y_train, X_test, y_test = load_train_test_data(60)

    X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 2)

    cnn = load_model("./Model/MLII/CNN_model_1.h5")

    prediction = cnn.predict(X_test)

    y_true, y_pred = util.get_prediction_Truth_value(prediction, y_test)

    util.metricsMeasurement(y_true, y_pred)



# Tesing Long Short Term Memory Neural Network Model
def LSTM_Model_Testing():

    print("Long Short Term Memory Neural Network Results:")

    X_train, y_train, X_test, y_test = load_train_test_data(60)


    X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 2)

    lstm = load_model("./Model/MLII/LSTM_model_1.h5")

    prediction = lstm.predict(X_test)

    y_true, y_pred = util.get_prediction_Truth_value(prediction, y_test)

    util.metricsMeasurement(y_true, y_pred)



# K Fold Cross Validation to obtain the average accuracy
def KFoldCrossValidation(k, number):
    imbLearn = bs.balanced_Sampling(60)

    X_signal, Y_label = imbLearn.featureExtract()

    X_data, Y_data = imbLearn.balanceSamples(X_signal, Y_label, 20000)

    kf = KFold(n_splits=k)

    Accuracy = []
    fold = 0

    for train_index, test_index in kf.split(X_data,Y_data):
        result = 0
        fold += 1

        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = Y_data[train_index], Y_data[test_index]

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        # imbLearn.write_TrainingTesting_toCSV(X_train,y_train,X_test,y_test)

        print("-------------------Validation: " + str(fold))

        if number == 1:
            print("Feed Forward Neural Network:")
            X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 1)
            result = FFNN.FeedForwardNeuralNetwork(X_train, y_train, X_test, y_test)

        elif number == 2:
            print("Convolutional Neural Network:")
            X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 2)
            result = CNN.ConvolutionalNeuralNetwork(X_train, y_train, X_test, y_test)

        elif number == 3:
            print("Long Short Term Memory Neural Network:")
            X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 2)
            result =  LSTM.LongShortTermMemoryNeuralNetwork(X_train, y_train, X_test, y_test)

        Accuracy.append(result)

    print(Accuracy)

    print(f"{k} Cross Validation Average Accuracy: {sum(Accuracy)/len(Accuracy)}")


