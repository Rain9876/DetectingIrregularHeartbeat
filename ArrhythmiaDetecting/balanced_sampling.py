#
## Author: Yurun SONG
## Date: 2019/01/15
## Project: Deep Learning about detecting irregular heartbeat
#
# balance_Sampling is a class processing the extracted signal samples.
# Due to the imbalance of five type of samples, the oversampling and undersampling methods should be implemented
# in order to strengthen the learning ability of models and improve the accuracy.
# Feature extraction and split the training and testing data are all illustrated in this class.
#


from imblearn.keras import BalancedBatchGenerator
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours,TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.utils import shuffle
from collections import Counter
import keras
import pandas as pd
import numpy as np
import ArrhythmiaDetecting.util as util
import matplotlib.pyplot as plt



class balanced_Sampling:


    def __init__(self,sampleShape):

        self.path = "../Dataset/ProcessedSignal/"
        self.sampleShape = sampleShape


    ## Read the processed signal sample file and assocaite with their features
    ## Return two lists storing the samples and features individually.
    def featureExtract(self):

        pdHeader = list(range(self.sampleShape))

        typeNameList = util.getTypeNameList()

        # X_data = np.empty([0, self.sampleShape])
        Y_data = np.array([])
        X_samplingList = []
        Y_samplingList = []

        for i in range(len(typeNameList)):

            signalContent = pd.read_csv(self.path + typeNameList[i]+".csv",header=pdHeader).values

            X_samplingList.append(signalContent)

            # X_data = np.append(X_data,signalContent,axis=0)
            y_temp = np.ones(len(signalContent)) * i
            Y_data = np.append(Y_data, y_temp)
            Y_samplingList.append(y_temp)


        print('Original dataset shape %s' % Counter(Y_data))

        return  X_samplingList, Y_samplingList


    ## Handling the balance-sampling
    ## Take two lists from the featureExtract and give the standand samples amount you want to achieve
    ## If nothing is provided, then oversampling (SMOTEENN) all the samples to Maximun
    ## Otherwise, oversamples (NearMiss) the minority class first, then undersamples (SMOTE) to the standard amount
    ## Shuffle the samples and feature at last
    ## Suggested amount: 3000 ~ 70000
    def balanceSamples(self,X_sampling, Y_sampling, amount = 0):

        X_data = np.empty([0, self.sampleShape])
        Y_data = np.array([])

        if amount == 0:

            for i in range(5):
                X_data = np.append(X_data,X_sampling[i],axis=0)
                Y_data = np.append(Y_data,Y_sampling[i],axis=0)

            sme = SMOTEENN(sampling_strategy = "auto", random_state = 42)
            X_data, Y_data = sme.fit_resample(X_data, Y_data)
            print('Resampled dataset shape %s' % Counter(Y_data))

        else:

            X_underSampling = np.empty([0, self.sampleShape])
            Y_underSampling = np.array([])
            X_overSampling = np.empty([0, self.sampleShape])
            Y_overSampling = np.array([])

            strategy = {}
            sub_strategy = {}

            for i in range(5):
                strategy[i] = amount

                if len(X_sampling[i]) > amount:

                    X_underSampling = np.append(X_underSampling,X_sampling[i],axis=0)
                    Y_underSampling = np.append(Y_underSampling,Y_sampling[i],axis=0)

                else:

                    X_overSampling = np.append(X_overSampling, X_sampling[i], axis=0)
                    Y_overSampling = np.append(Y_overSampling, Y_sampling[i], axis=0)
                    sub_strategy[i] = amount


            smote = SMOTE(sampling_strategy= sub_strategy,random_state = 42)
            X_overSampling, Y_overSampling = smote.fit_resample(X_overSampling,Y_overSampling)

            X_data =  np.append(X_underSampling,X_overSampling,axis=0)
            Y_data =  np.append(Y_underSampling,Y_overSampling,axis=0)
            print('1st Resampled dataset shape %s' % Counter(Y_data))

            enn = NearMiss(sampling_strategy= strategy, random_state=42)
            X_data, Y_data = enn.fit_resample(X_data,Y_data)
            print('2nd Resampled dataset shape %s' % Counter(Y_data))


        X_data, Y_data = shuffle(X_data,Y_data,random_state = 2)

        return X_data, Y_data

    ## Split the samples into Training and Testing sets
    ## Scaler the these value between 0 and 1
    def train_test_data(self,X_data,Y_data):
        X_train,X_test,y_train,y_test = train_test_split(X_data,Y_data,random_state=2)

        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        # Data needs to be scaled to a small range like 0 to 1 for the neural network to work well.

        scaler = MinMaxScaler(feature_range=(0, 1))

        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        return X_train,X_test,y_train,y_test

    ## Write the Training and Testing data into csv file
    def write_TrainingTesting_toCSV(self, X_train, X_test, y_train, y_test):
        pd.DataFrame(X_train).to_csv(self.path+"training_signals.csv",mode="w",index=False)
        pd.DataFrame(y_train).to_csv(self.path+"training_labels.csv",mode="w", index= False)
        pd.DataFrame(X_test).to_csv(self.path+"testing_signals.csv",mode="w", index= False)
        pd.DataFrame(y_test).to_csv(self.path+"testing_labels.csv",mode="w", index=False)


    ## Read the Training and Testing data from csv file
    def read_TrainingTesting_data(self):
        X_train = pd.read_csv(self.path + "training_signals.csv", dtype=np.float).values
        X_test = pd.read_csv(self.path + "testing_signals.csv", dtype=np.float).values
        y_train = pd.read_csv(self.path + "training_labels.csv", dtype=np.int32).values
        y_test = pd.read_csv(self.path + "testing_labels.csv", dtype=np.int32).values

        return X_train, y_train, X_test, y_test






###---Testing----------------------------------------------------###

# imLearn = balanced_Sampling(60)
#
# X_signal, Y_label = imLearn.featureExtract()
#
# X_data, Y_data = imLearn.balanceSamples(X_signal, Y_label, 6000)
#
# X_train,X_test,y_train,y_test = imLearn.train_test_data(X_data,Y_data)

# t = np.arange(60)
# for i in range(10):
#     print(Y_data[i])
#     plt.plot(t, X_data[i], marker='.')
#     plt.xlabel("elapsed time")
#     plt.ylabel("signal mv")
#     plt.show()


# imLearn.write_TrainingTesting_toCSV(X_train,X_test,y_train,y_test)



