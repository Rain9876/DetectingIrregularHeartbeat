import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import annotationDataProcessing as adp,signalDataProcessing as sdp, balanced_sampling as bs, util



###---SignalDataProcessing----------------------------------------------------###



## Test Part 1: process all patient in PatientsNumber txt file

# patientNumber = util.getPatientsNumber()
#
# for i in patientNumber[:24]:
#
#     signal = sdp.SignalDataProcessing(i,"MLII",300,60, True)
#
#     signal.processingAllSignal("N")
#     signal.processingAllSignal("VEB")
#     signal.processingAllSignal("SVEB")
#     signal.processingAllSignal("F")
#     signal.processingAllSignal("Q")



### Test Part 2: show a long-time heart beats graph for baseline drift

# signal = sdp.SignalDataProcessing("101","MLII",300,60,False)
# aa = signal.signalContent.ElapsedTime.values[4000:20000]
# bb = signal.signalContent.Signal.values[4000:20000]
# signal.plotSignal(aa,bb,"all")



### Test Part 3: read csv file and check problems

# kk = pd.read_csv("../Dataset/ProcessedSignal/N/N.csv")
# print(len(kk["0"]))
# print(kk.head())
#
# aa = kk.iloc[1].values



### Test Part 4: process a single patient, all heart beats

# signal = sdp.SignalDataProcessing("217", "MLII", 300, 60, False)
# signal.processingAllSignal("N")
# signal.processingAllSignal("VEB")
# signal.processingAllSignal("SVEB")
# signal.processingAllSignal("F")
# signal.processingAllSignal("Q")


### Test Part 5: process a single patient, a single heart beat

# signal = sdp.SignalDataProcessing("101", "MLII", 300, 60,False)
# signal.processingSignal("N",3)





###---Balanced_Sampling----------------------------------------------------###



# imLearn = bs.balanced_Sampling()
#
# X_signal, Y_label = imLearn.featureExtract(60)
#
# X_data, Y_data = imLearn.balanceSamples(X_signal, Y_label)
#
# X_train,X_test,y_train,y_test = imLearn.train_test_data(X_data,Y_data)
#
# imLearn.write_TrainingTesting_toCSV(X_train,X_test,y_train,y_test)





## ------------------------------------------------------------------------------------ ##

