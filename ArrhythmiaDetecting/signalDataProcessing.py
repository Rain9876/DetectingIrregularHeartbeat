#
## Author: Yurun SONG
## Date: 2018/10/15
## Project: Deep Learning about detecting irregular heartbeat
#
# ignalDataProcessing class handles the signal file (dat) derived from the MIT-BIH arrhythmia database.
# The dat file has ElapsedTime and signal attributes
#
# According the sampleList getting from annotationDataProcessing file, Slicing each relevant heart beat signal
# into a small part.
# Due to roughly 300 signals per heart beat, we make a slice window, size of 300, capturing the signals around the R peak
#
# Others pre-processing steps including removing high-frequency noise, adjusting baseline drift and
# interpolation are implemented in order to obtain more accurate signals as classification input
#


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import annotationDataProcessing as ap
import util
from scipy.interpolate import spline
from scipy.interpolate import make_interp_spline, BSpline, CubicSpline
from scipy.signal import cspline1d
import pywt


class SignalDataProcessing:

    ## Processing the signal file of the patient with input patientNumber
    ## leadType is "MLII", which is most common one in MIT-BIH.
    ## windowsize is the slice window size where we extract signal points around the R peak.
    ## linspace is the number of signal points we are about to input into classifiers.

    def __init__(self, patientNumber, leadType, windowSize, linspace, WFDB = True):
        self.patientNumber = patientNumber
        self.leadType = leadType
        self.signalContent = None
        self.annotation = None
        self.windowSize = windowSize
        self.linspace = linspace
        self.writeDown = False

        if WFDB:

            self.getSigalFileByWFDB()
        else:

            self.read_signalCSV()


    ## Read the signal csv file locally and decomposing the signal data
    def read_signalCSV(self):

        self.signalContent = pd.read_csv(f"../Dataset/Signal/{self.patientNumber}_signal.csv",skiprows=[1])

        self.annotation = ap.AnnotationProcessing(self.patientNumber)

        # The signal header format is a string "'Signal'".
        signal = "'" + self.leadType + "'"

        self.signalContent.rename(columns = {"'Elapsed time'":"ElapsedTime",signal:"Signal"}, inplace = True)

        signalArray = self.signalContent["Signal"].values

        # Slightly adjustment, causing the floor & ceil function in DWT, the size
        # after decomposition and reconstruction is roughly 2 signal units more than original signals.

        self.signalContent.Signal = self.baseLineDrift_denoise(signalArray)[:len(self.signalContent.Signal)]

        self.printPatientInfor()


    ## Read the signal file directly from MIT-BIH database by WFDB and decomposing the signal data
    def getSigalFileByWFDB(self):

        self.annotation = ap.AnnotationProcessing(self.patientNumber)

        dfSignal, frequency = util.getPatientsDatFile(self.patientNumber,self.leadType)

        signalArray = self.baseLineDrift_denoise(dfSignal.values)

        # timeList = list(range(len(signalArray)))*(1/frequency)
        timeArray = np.arange(len(signalArray))*(1/frequency)

        timeArray = np.around(timeArray,decimals=3)

        self.signalContent =  pd.DataFrame({"ElapsedTime":timeArray, "Signal":signalArray})

        self.printPatientInfor()


    ## Based on the atr samples list provided, extracting R peak signal and signals around R peak
    ## P wave takes roughly 30% of total signals, S-T wave has the rest 70%
    def extractSignal(self, AtrSamples):

        df_collection = []
        left = int (0.3 * self.windowSize)
        right = int (0.7 * self.windowSize)

        for i in AtrSamples:

            # small slice window to achieve the accurate R Peak
            rangeSignal =  self.signalContent.Signal.values[int(i)-5:int(i)+ 5]

            maxIndex = np.argmax(rangeSignal)

            i = int(i) + (maxIndex - 5)

            a = self.signalContent.loc[(int(i)-left):(int(i)+right-1),['ElapsedTime','Signal']]

            a.ElapsedTime = a.ElapsedTime - a.ElapsedTime.values[0]

            df_collection.append(a)

        return df_collection



    # Write down patient information about how many heart beat of each type for analysis
    def writePatientInfor(self):

        allBeatsList = self.annotation.getAll_AtrList()

        with open("../Dataset/NumberofHeartBeat.txt", "a+") as f:

            for i in allBeatsList:
                f.write(str(len(i)))
                f.write(",")

            f.write("\n")

        f.close()


    ## print out the patient information
    def printPatientInfor(self):

        print(self.patientNumber)

        typeNameList = util.getTypeNameList()
        allTypeList = self.annotation.getAll_AtrList()

        for i in range(len(typeNameList)):
            print(typeNameList[i] + ": " + str(len(allTypeList[i])))

        print("---------------")


    ## According to type, ploting the time & signal graph
    def plotSignal(self, time, signal, type):

        plt.plot(time, signal,marker = '.')
        plt.title(self.patientNumber +"_" + type + "_" + str(len(time)))
        plt.xlabel("elapsed time")
        plt.ylabel("signal mv")
        plt.show()


    ## Using multi Discrete Wavelet Tranform (DWT) to decompose the signal to 8 levels
    ## wavelet mother is bior3.9, which is best for this kinds signal after analysis
    ## Remove the baseline drift and high frequency noise, then reconstruct the signals
    def baseLineDrift_denoise(self,inputSignal):

        decSignal = pywt.wavedec(inputSignal, 'bior3.9',"smooth",level=8)
        decSignal[0] = np.zeros(len(decSignal[1]))        # Remove CA8, lower frequency baseline drift
        decSignal[-1] = None                              # Remove CD1 and CD2, high frequency noise
        decSignal[-2] = None

        recSignal =  pywt.waverec(decSignal, "bior3.9")

        return recSignal


    ## Passing the type and number of type signals we want to extract and process individually
    ## concatenate these scattered signals together
    def processingSignal(self, type, number):

        signalArray = np.empty([0,self.linspace])
        atr = []

        if type == "N":
            atr = self.annotation.getNormal_AtrSample()[:number]

        elif type == "VEB":
            atr = self.annotation.getVEB_AtrSample()[:number]

        elif type == "SVEB":
            atr = self.annotation.getSVEB_AtrSample()[:number]

        elif type == "F":
            atr = self.annotation.getF_AtrSample()[:number]

        elif type == "Q":
            atr = self.annotation.getQ_AtrSample()[:number]

        # Extract the signals in the atr list
        originalSignal  = self.extractSignal(atr)


        for i in originalSignal:

            newTime, newSignal = self.scatteredSignalPoint(i.ElapsedTime.values,i.Signal.values)

            ## Plot time-signal graph for every heart beat
            # self.plotSignal(newTime,newSignal,type)

            signalArray = np.append(signalArray,[newSignal],axis=0)

        return signalArray



    ## CubicSpline is used to transfrom 300 (windowSize) signal data points
    ## to 60 (linspace) signal data points.
    ## In order to reserving the R peak data in scattered points, spliting these signals
    ## to left and right parts of R peak, then interpolate each part.
    def scatteredSignalPoint(self, time , signal):

        RPosition = int (0.3 * self.windowSize)

        RPeak = (time[RPosition],signal[RPosition])

        spl = CubicSpline(time, signal)  # BSpline interpolate

        leftTime = np.linspace(time[0],time[RPosition],int(0.3* self.linspace))

        leftSignal = spl(leftTime)

        rightTime = np.linspace(time[RPosition], time[-1], int(0.7* self.linspace))

        rightSignal = spl(rightTime)

        newTime = np.concatenate([leftTime,rightTime], axis=None)

        newSignal = np.concatenate([leftSignal,rightSignal] ,axis=None)

        # # print(RPeak[1])

        # plt.plot(time, signal)

        # plt.plot(newTime, newSignal, marker = ".")

        # plt.plot(RPeak[0],RPeak[1], marker = "o")

        # plt.xlabel("elapsed time")

        # plt.ylabel("signal mv")

        # plt.legend(["Old","New","R_Peak"])

        # plt.show()

        return (newTime,newSignal)


    ## Write all the signals into csv file
    def writeSignalsToCSV(self, writeDown):
        self.writeDown = writeDown

    ## Processing all signals of particular type and then writing into csv file.
    def processingAllSignal(self,type):

        array = np.array([])

        path = "../Dataset/ProcessedSignal/"

        if type == "N":
            array = self.processingSignal(type,len(self.annotation.getNormal_AtrSample()))
            path += "N.csv"

        elif type == "VEB":
            array = self.processingSignal(type,len(self.annotation.getVEB_AtrSample()))
            path += "VEB.csv"

        elif type == "SVEB":
            array = self.processingSignal(type,len(self.annotation.getSVEB_AtrSample()))
            path += "SVEB.csv"

        elif type == "F":
            array = self.processingSignal(type,len(self.annotation.getF_AtrSample()))
            path += "F.csv"

        elif type == "Q":
            array = self.processingSignal(type,len(self.annotation.getQ_AtrSample()))
            path += "Q.csv"

        if self.writeDown and array.size != 0:
            pd.DataFrame(array).to_csv(path, mode="a", header=False, index=False)



###--Testing-----------------------------------------------------###


## Open the PatientNumber.txt
# with open("../Dataset/PatientsNumber.txt","r+") as f:
#     patientNumber = f.read().split('\n')
# f.close()



### Test Part 1: process all patient in PatientsNumber txt file

# for i in patientNumber[:15]:
#
#     signal = SignalDataProcessing(i,"MLII",300,60)
#
#     # signal.processingAllSignal("N")
#     # signal.processingAllSignal("VEB")
#     # signal.processingAllSignal("SVEB")
#     # signal.processingAllSignal("F")
#     # signal.processingAllSignal("Q")



### Test Part 2: show a long-time heart beats graph for baseline drift

# signal = SignalDataProcessing("109","MLII",300,60)
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

# signal = SignalDataProcessing("217", "MLII", 300, 60)
# signal.processingAllSignal("N")
# signal.processingAllSignal("VEB")
# signal.processingAllSignal("SVEB")
# signal.processingAllSignal("F")
# signal.processingAllSignal("Q")


### Test Part 5: process a single patient, a single heart beat

# signal = SignalDataProcessing("101", "MLII", 300, 60)
# signal.processingSignal("N",3)
