from ArrhythmiaDetecting import util, balanced_sampling as bs,signalDataProcessing as sdp, \
    FeedForwardNN as FFNN, convolutionNN as CNN, LongShortTermMemoryNN as LSTM



def main():

    print("-----------------------------------------------------")

    patientNumber = util.getPatientsNumber()

    for i in patientNumber[:24]:

        signal = sdp.SignalDataProcessing(i,"MLII",300,60, True)

        signal.processingAllSignal("N")
        signal.processingAllSignal("VEB")
        signal.processingAllSignal("SVEB")
        signal.processingAllSignal("F")
        signal.processingAllSignal("Q")


    imLearn = bs.balanced_Sampling()

    X_signal, Y_label = imLearn.featureExtract(60)

    X_data, Y_data = imLearn.balanceSamples(X_signal, Y_label)

    X_train,X_test,y_train,y_test = imLearn.train_test_data(X_data,Y_data)

    imLearn.write_TrainingTesting_toCSV(X_train,X_test,y_train,y_test)


    print("Feed Forward Neural Network:")
    X_train, y_train, X_test, y_test = FFNN.load_train_test_data()
    FFNN.FeedForwardNeuralNetwork(X_train, y_train, X_test, y_test)


    print("Convolution Neural Network:")
    X_train, y_train, X_test, y_test = CNN.load_train_test_data()
    CNN.ConvolutionNeuralNetwork(X_train, y_train, X_test, y_test)


    print("Long Short Term Memory Neural Network:")
    X_train, y_train, X_test, y_test = LSTM.load_train_test_data()
    LSTM.LongShortTermMemoryNeuralNetwork(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()