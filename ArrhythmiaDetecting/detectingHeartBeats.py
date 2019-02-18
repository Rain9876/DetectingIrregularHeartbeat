## Author: Yurun SONG
## Date: 2019/01/30
## Project: Deep Learning about detecting irregular heartbeat
#

import numpy as np
from keras.utils import to_categorical
import util, balanced_sampling as bs,signalDataProcessing as sdp, \
    FeedForwardNN as FFNN, convolutionNN as CNN, LongShortTermMemoryNN as LSTM


## A series of tests starting from extracting data from the MIT-BIH to the models training
def main():

    # print("-----------------------------------------------------")
    #
    # patientNumber = util.getPatientsNumber()
    #
    # for i in patientNumber[:40]:
    #
    #     signal = sdp.SignalDataProcessing(i,"MLII",300,60, False)
    #
    #     signal.writeSignalsToCSV(False)
    #
    #     signal.processingAllSignal("N")
    #     signal.processingAllSignal("VEB")
    #     signal.processingAllSignal("SVEB")
    #     signal.processingAllSignal("F")
    #     signal.processingAllSignal("Q")


    imLearn = bs.balanced_Sampling(60)

    X_signal, Y_label = imLearn.featureExtract()

    X_data, Y_data = imLearn.balanceSamples(X_signal, Y_label,20000)

    X_train,X_test,y_train,y_test = imLearn.train_test_data(X_data,Y_data)

    ## imLearn.write_TrainingTesting_toCSV(X_train,X_test,y_train,y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # print("Feed Forward Neural Network:")
    # FFNN.FeedForwardNeuralNetwork(X_train, y_train, X_test, y_test)
    #
    # X_train = np.expand_dims(X_train, axis=2)
    # X_test = np.expand_dims(X_test, axis=2)

    # print("Convolution Neural Network:")
    # CNN.ConvolutionNeuralNetwork(X_train, y_train, X_test, y_test)


    # print("Long Short Term Memory Neural Network:")
    # LSTM.LongShortTermMemoryNeuralNetwork(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()