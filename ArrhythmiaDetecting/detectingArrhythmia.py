## Author: Yurun SONG
## Date: 2019/01/30
## Project: Deep Learning about detecting irregular heartbeat
#
import numpy as np
from keras.utils import to_categorical
import util, model_testing, balanced_sampling as bs,signalDataProcessing as sdp, \
    FeedForwardNN as FFNN, ConvolutionalNN as CNN, LongShortTermMemoryNN as LSTM


# Signal Processing
#
# It takes patient number from 1 to 47 can be found in PatientNumber.txt
# Other patients can be selected if it changes.
#
# Default Lead type is MLII and others like V1, V5 can also be test
# Set WFDB is true, if other type is wanted
#
# sliceWindow and output_shape can be to control the input and output shape
#
# WFDB is a tool to extract data online from MIT-BIH.
# If false, MLII signal in local file will be used.
#
# writeDown is used to record the processed signal into csv file.
# Empty ProcessSignla folder, If new processed signals are recorded
#
# The ProcessedSignal folder in Dataset should have Five files eventually.
# N.csv, VEB.csv, SVEB.csv, F.csv, Q.csv
#


def ProcessingSignal(start= 1,end= 47,type= "MLII",sliceWindow= 300,output_shape= 60,WFDB= False, writeDown= False):

    patientNumber = util.getPatientsNumber()

    for i in patientNumber[start:end]:

        signal = sdp.SignalDataProcessing(i,type,sliceWindow,output_shape, WFDB)

        signal.writeSignalsToCSV(writeDown)

        signal.processingAllSignal("N")
        signal.processingAllSignal("VEB")
        signal.processingAllSignal("SVEB")
        signal.processingAllSignal("F")
        signal.processingAllSignal("Q")



# balance sampling
#
# Input_shape should be identical with the ProcessedSignal method's output_shape
#
# amount is the criteria that sampling method should reach
# so that all classification type have exactly that amount of samples
#
# Return X_train,X_test,y_train,y_test data, which can be writen down into csv file
#


def data_balancing(input_shape= 60,amount= 20000):

    imbLearn = bs.balanced_Sampling(input_shape)

    X_signal, Y_label = imbLearn.featureExtract()

    X_data, Y_data = imbLearn.balanceSamples(X_signal, Y_label,amount)

    X_train,X_test,y_train,y_test = imbLearn.train_test_data(X_data,Y_data)

    # imbLearn.write_TrainingTesting_toCSV(X_train,y_train,X_test,y_test)

    return X_train,X_test,y_train,y_test



#  No.1 is Feed Forward Neural Network;
#  No.2 is Convolutional Neural Network;
#  No.3 is Long Short Term Memory Neural Network
#
#  Adjust the different Neural Network's input dimension and each contains training, testing & evaluation
#

def constructNeuralNetworkModels(X_train,X_test,y_train,y_test, number):

    if number == 1:
        print("Feed Forward Neural Network:")
        X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 1)
        FFNN.FeedForwardNeuralNetwork(X_train, y_train, X_test, y_test)

    elif number == 2:
        print("Convolutional Neural Network:")
        X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 2)
        CNN.ConvolutionalNeuralNetwork(X_train, y_train, X_test, y_test)

    elif number == 3:
        print("Long Short Term Memory Neural Network:")
        X_train, y_train, X_test, y_test = util.adjustInputDimension(X_train, y_train, X_test, y_test, 2)
        LSTM.LongShortTermMemoryNeuralNetwork(X_train, y_train, X_test, y_test)




# Example
# Processing ECG signal of all MLII type patients from MIT-BIH online
# Balance the classification samples to 20000 each
# Construct the Feed Forward Neural Network
# 5 fold cross vaildation testing of that model

if __name__ == '__main__':

    print("-----------------------------------------------------")

    # ProcessingSignal(1,47,"MLII",300,60,True,False)
    # X_train, X_test, y_train, y_test = data_balancing(60, 20000)
    # constructNeuralNetworkModels(X_train, X_test, y_train, y_test,1)
    # model_testing.KFoldCrossValidation(5,1)