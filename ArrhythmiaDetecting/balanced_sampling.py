from imblearn.keras import BalancedBatchGenerator
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.utils import shuffle
from collections import Counter
import keras
import pandas as pd
import numpy as np
import ArrhythmiaDetecting.util as util


class balanced_Sampling:

    def __init__(self):

        self.path = "../Dataset/ProcessedSignal/"


    def featureExtract(self, sampleShape):

        pdHeader = list(range(sampleShape))

        typeNameList = util.getTypeNameList()

        Y_data = np.array([])
        X_data = np.empty([0, sampleShape])


        for i in range(len(typeNameList)):

            signalContent = pd.read_csv(self.path + typeNameList[i]+".csv",header=pdHeader).values

            X_data = np.append(X_data,signalContent,axis=0)
            y_temp = np.ones(len(signalContent)) * i
            Y_data = np.append(Y_data, y_temp)


        print(X_data.shape)
        print(Y_data.shape)

        return (X_data, Y_data)


    def balanceSamples(self,X_input,Y_input):

        # X_input = X_input[20000:]
        # Y_input = Y_input[20000:]

        strategy = 'auto'

        X_data, Y_data = shuffle(X_input,Y_input,random_state = 2)

        print('Original dataset shape %s' % Counter(Y_data))

        sme = SMOTEENN(sampling_strategy = strategy, random_state = 42)
        X_data, Y_data = sme.fit_resample(X_data, Y_data)

        # smote = SMOTE(sampling_strategy="auto",random_state = 42)
        # X_data, Y_data = smote.fit_resample(X_data,Y_data)
        #
        # print('Resampled 1st dataset shape %s' % Counter(Y_data))
        #
        # enn = EditedNearestNeighbours(sampling_strategy= "auto", random_state=42)
        # X_data, Y_data = enn.fit_resample(X_data,Y_data)


        print('Resampled dataset shape %s' % Counter(Y_data))

        return X_data, Y_data


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


    def write_TrainingTesting_toCSV(self, X_train, X_test, y_train, y_test):
        pd.DataFrame(X_train).to_csv(self.path+"training_signals.csv",mode="w",index=False)
        pd.DataFrame(y_train).to_csv(self.path+"training_labels.csv",mode="w", index= False)
        pd.DataFrame(X_test).to_csv(self.path+"testing_signals.csv",mode="w", index= False)
        pd.DataFrame(y_test).to_csv(self.path+"testing_labels.csv",mode="w", index=False)



    def read_TrainingTesting_data(self):
        X_train = pd.read_csv(self.path + "training_signals.csv", dtype=np.float).values
        X_test = pd.read_csv(self.path + "testing_signals.csv", dtype=np.float).values
        y_train = pd.read_csv(self.path + "training_labels.csv", dtype=np.int32).values
        y_test = pd.read_csv(self.path + "testing_labels.csv", dtype=np.int32).values

        return X_train, y_train, X_test, y_test






###---Testing----------------------------------------------------###

# imLearn = balanced_Sampling()
#
# X_signal, Y_label = imLearn.featureExtract(60)
#
# X_data, Y_data = imLearn.balanceSamples(X_signal, Y_label)
#
# X_train,X_test,y_train,y_test = imLearn.train_test_data(X_data,Y_data)
#
# imLearn.write_TrainingTesting_toCSV(X_train,X_test,y_train,y_test)



