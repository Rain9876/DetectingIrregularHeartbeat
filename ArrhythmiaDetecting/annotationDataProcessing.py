#
## Author: Yurun SONG
## Date: 2018/10/15
## Project: Deep Learning about detecting irregular heartbeat
#
# AnnotationDataProcessing class handles the annotation file (atr) derived from the MIT-BIH arrhythmia database.
# The atr file has Time, Sample, Type and other attributes, which are corresponded to the signal file in database.
#
#
# The signal symbolList is a relative feature collection that each heart beat is diagnosed.
# These signal symbols are classified into five types: Normal, Ventricular ectopic beat (VEB),
# Supra-ventricular ectopic beat (SVEB), Fusion Beat (F) and unknown beat (Q).
# These symbols in type list are demonstrated in util.py file and atr files are stored in Dataset folder
#
#
# This class is going to extract the symbol list combined with the sample list
# so that we can pre-process the signal data with relevant signal type later.
#

import util

class AnnotationProcessing:

    def __init__(self,patientName, WFDB = True):
        self.fileList = []
        self.timeList = []
        self.symbolList = []
        self.sampleList = []
        self.combinedList = []

        if WFDB:

            self.getFileByWFDB(patientName)
        else:

            self.getFileList(patientName)

        self.filiterAtrSample()


    ## Read txt file locally and store necessary information into fileList
    def getFileList(self,patientName):
        with open(f"../Dataset/Annotation/{patientName}_atr.txt") as txt:
            for row in txt:
                rowList = row.split()[:3]
                self.fileList.append(rowList)

        self.fileList[0][2] = "Type"

        # skip first three rows including the header
        for i in self.fileList[3:]:
            self.timeList.append(i[0])
            self.sampleList.append((i[1]))
            self.symbolList.append((i[2]))


    ## Read atr file directly from MIT-BIH database by WFDB
    def getFileByWFDB(self,patientName):

        sampleList,symbolList = util.getPatientsAtrFile(patientName)

        # skip first two rows, excluding the header
        self.sampleList = sampleList[2:]
        self.symbolList = symbolList[2:]


    ## According the signal typeList, classifying the relative sample index into five classification lists
    ## All heart beats are processed in this method.
    def filiterAtrSample(self):

        normalList = []
        VEBList = []
        SVEBList = []
        FList = []
        QList = []

        for i in range(len(self.symbolList)-1):

            if self.symbolList[i] in util.NormalType:
                normalList.append(self.sampleList[i])

            elif self.symbolList[i] in util.VEB:
                VEBList.append(self.sampleList[i])

            elif self.symbolList[i] in util.SVEB:
                SVEBList.append(self.sampleList[i])

            elif self.symbolList[i] in util.F:
                FList.append(self.sampleList[i])

            elif self.symbolList[i] in util.Q:
                QList.append(self.sampleList[i])

        self.combinedList = [normalList,VEBList,SVEBList,FList,QList]


    ## return the time list
    def getTimeList(self):
        return self.timeList

    ## return the sample list
    def getSampleList(self):
        return self.sampleList

    ## return the signal symbol list
    def getsymbolList(self):
        return self.symbolList


    ## Return all the normal heart beat samples of the patient
    def getNormal_AtrSample(self):
        return self.combinedList[0]

    ## Return all the VEB heart beat samples of the patient
    def getVEB_AtrSample(self):
        return self.combinedList[1]

    ## Return all the SVEB heart beat samples of the patient
    def getSVEB_AtrSample(self):
        return self.combinedList[2]

    ## Return all the fusion heart beat samples of the patient
    def getF_AtrSample(self):
        return self.combinedList[3]

    ## Return all the unknown heart beat samples of the patient
    def getQ_AtrSample(self):
        return  self.combinedList[4]

    ## Return all the heart beat samples of the patient
    def getAll_AtrList(self):
        return  self.combinedList

