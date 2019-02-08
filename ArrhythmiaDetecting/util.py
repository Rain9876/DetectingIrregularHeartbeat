from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,accuracy_score
from wfdb import rdsamp,rdann
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 0
NormalType = ["N", "L", "R"]
# 1
VEB = ["V","E"]
# 2
SVEB = ["a","J","A","S","e","j"]
# 3
F = ["F"]
# 4
Q = ["Q","/","f","u"]



def getNormalType():
    return NormalType

def getVEBType():
    return VEB

def getSVEBType():
    return SVEB

def getFType():
    return F

def getQType():
    return Q

def getTypeNameList():
    return ["N","VEB","SVEB","F","Q"]



def metricsMeasurement(y_true, y_pred):

    print("-------------------------------")

    accuracy = accuracy_score(y_true,y_pred)

    print("Accuracy: " + str(accuracy))

    conMatr = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    print("Confusion Matrix:")
    print(conMatr)

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred,average=None,labels=[0,1,2,3,4])

    typeNameList  = getTypeNameList()

    for i in range(len(precision)):
        temp = "| precision: {:.2f} | recall: {:.2f} | F1 score: {:.2f} | {}".format(precision[i],recall[i],fscore[i],typeNameList[i])
        print (temp)



def get_prediction_Truth_value(prediction, y_test):
    y_true = []
    y_pred = []

    for row in range(len(prediction)):
        pred_sample = prediction[row]
        true_sample = y_test[row]
        predMaxIndex = np.argmax(pred_sample)
        trueMaxIndex = np.argmax(true_sample)
        y_pred.append(predMaxIndex)
        y_true.append(trueMaxIndex)

    return y_true, y_pred



def plotAccuracyGraph(model):
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(["training_acc", "testing_acc"])
    plt.show()


# Open the PatientNumber.txt
def getPatientsNumber():

    with open("../Dataset/PatientsNumber.txt","r+") as f:
        patientNumber = f.read().split('\n')
    f.close()
    return patientNumber


def getPatientsAtrFile(patientsNumber):
    ann = rdann(patientsNumber,"atr",pb_dir="mitdb")
    return ann.sample, ann.symbol


def getPatientsDatFile(patientsNumber,signalName):

    signals, fields = rdsamp(patientsNumber,pb_dir="mitdb")
    index = fields['sig_name'].index(signalName)
    frequency = fields['fs']
    df = pd.Series((v[index] for v in signals), name=signalName)

    return df, frequency