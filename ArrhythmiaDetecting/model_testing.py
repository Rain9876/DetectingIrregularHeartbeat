from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import *
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

    # Convolution NN, LSTM NN
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    return X_train, y_train, X_test, y_test



X_train, y_train, X_test, y_test = load_train_test_data()





##------------------------------------------------------##

# print("Feed Forward Neural Network Results:")
#
# md = load_model("./Model/FFNN_model_3.h5")
#
# prediction = md.predict(X_test)
#
# y_true, y_pred = util.get_prediction_Truth_value(prediction, y_test)
#
# util.metricsMeasurement(y_true, y_pred)

##------------------------------------------------------##

# print("Convolution Neural Network Results:")
#
# cnn = load_model("./Model/CNN_model_1.h5")
#
# prediction = cnn.predict(X_test)
#
# y_true, y_pred = util.get_prediction_Truth_value(prediction, y_test)
#
# util.metricsMeasurement(y_true, y_pred)
#
# # ##------------------------------------------------------##
#
# print("Long Short Term Memory Neural Network Results:")
#
# lstm = load_model("./Model/LSTM_model_1.h5")
#
# prediction = lstm.predict(X_test)
#
# y_true, y_pred = util.get_prediction_Truth_value(prediction, y_test)
#
# util.metricsMeasurement(y_true, y_pred)
