import numpy as np
import sklearn as sk
import csv as cs
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier


proteinDict = {"test" : [(True, float)]}
proteinDict.pop("test")
proteinList = []

dataList = [['sampleID','cancer']]
rowIndex = 0
with open('./gastricDataSetTotalFormatted.csv') as csv_file:
    csv = cs.reader(csv_file, delimiter=',')
    for row in csv:
        rowReal = [row[0]]
        rowReal.extend(row[3:])
        if rowIndex == 0:
            rowReal.pop(0)
            for column in range(len(rowReal)):
                dataList.append([rowReal[column]])
        if rowIndex == 2:
            for column in range(len(rowReal)):
                dataList[column].extend([rowReal[column]])
            #dataList.pop(2)
        elif rowIndex >0:
            for column in range(len(rowReal)):
                dataList[column].extend([rowReal[column]])
            if rowIndex == 3:
                dataList[0].pop(3)
        if rowIndex > 3440:
            print()
        rowIndex += 1
proteinList = dataList[0][3:]
for protein in proteinList:
    proteinDict[protein] = []
dataList[0].pop(2)
def remove_items(test_list, item):
    # using filter() + __ne__ to perform the task
    res = list(filter((item).__ne__, test_list))
    return res

dataListTemp = dataList.copy()
dataList = []
for row in dataListTemp:
    dataList.append(remove_items(row,''))

dataList.pop(0)
#print(*dataList[0])

kf = KFold(n_splits=5, random_state=None, shuffle=True)
folds = [[]]
folds.pop(0)


#for LOOCV
for x in range(0,len(dataList)):
    tempArrayFolds = []
    for i in range(0,len(dataList)):
        if (x != i):
            tempArrayFolds.append(i)
    folds.append([tempArrayFolds,[x]])

learningrate = 0.25
maxDepth = 2
thresholdInput = 1.8
method = "CausalScores_S2" + str(thresholdInput)[0] + "_" + str(thresholdInput)[2]

disj10Proba = []
disj15Proba = []
disj30Proba = []
disj10wSProba = []
disj15wSProba = []
disj30wSProba = []

rocCurveYtest = []
rocCurveYProba = []
totalBiomarkers = []

bioMarkerCausalValues = {}
for protein in proteinList:
    bioMarkerCausalValues[protein] = []

finalPrintData = [["Threshold"," ","   ","      ","Baseline Approach","           "," ","      ","Disj 10  ","          "," ","      ","Disj 15  ","          "," ","      ","Disj 30  ","          "," ","      ","Disj 10 w/ Sample Data","          "," ","      ","Disj 15 w/ Sample Data","          "," ","      ","Disj 30 w/ Sample Data","          "],
                 ["         "," ","Folds","Recall","Precision        ","Specificity"," ","Recall","Precision","Specificty"," ","Recall","Precision","Specificty"," ","Recall","Precision","Specificty"," ","Recall","Precision             ","Specificty"," ","Recall","Precision             ","Specificty"," ","Recall","Precision             ","Specificty"]
]
#0.8,0.9,1.0,1.1,1.2,1.3,1.4
varyingThresholds = [thresholdInput]
for threshold in varyingThresholds:
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    roc10 = []
    roc15 = []
    roc30 = []
    roc10wS = []
    roc15wS = []
    roc30wS = []
    #fig, ax = plt.subplots()

    tprs10 = []
    aucs10 = []
    mean_fpr10 = np.linspace(0, 1, 100)
    fig10, ax10 = plt.subplots()
    
    tempString = ""

    def baselineApproach(clf, train, testDataBaseline, testAffectedData):
        recallNum = 0.0
        precisionNum = 0.0
        specNum = 0.0

        recallDenom = 0.0
        precisionDenom = 0.0
        specDenom = 0.0
        tempStorage = []

        x_test = []
        y_test = []

        for x in testDataBaseline:
            programPredict = clf.predict([x[1]])
            realData = x[0]
            tempStorage.append((programPredict,realData))
            #print(programPredict, "  ans:", realData, programPredict==realData)
            #for Recall & Precision
            if (programPredict and realData):
                recallNum += 1.0
                precisionNum += 1.0
            if (realData):
                recallDenom += 1.0
            if (programPredict):
                precisionDenom += 1.0
            #for Specificity 
            if (programPredict == False and realData == False):
                specNum += 1.0
            if (realData == False):
                specDenom += 1.0

            x_test.append(x[1])
            y_test.append(x[0])

        #fig, ax = plt.subplots()
        if recallDenom == 0:
            recallDenom = 1
            recallNum = -1
        if precisionDenom == 0:
            precisionDenom = 1
            precisionNum = -1
        if specDenom == 0:
            specDenom = 1
            specNum = -1    
        #rfc_disp = RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax, name="Baseline")
        rfc_disp = ""
        recall = recallNum/recallDenom
        precision = precisionNum/precisionDenom
        specificity = specNum/specDenom
        
        return ([recall, precision, specificity],rfc_disp)

    tempString = "23"
    def combinedCausalityApproach(indexs, trainTotalData,counter,testData,testIndex):
        for protein in proteinDict:
            proteinDict[protein] = []
        for dataPoint in trainTotalData:
            index = 0
            for singleData in dataPoint[1]:
                proteinDict[proteinList[index]].append((dataPoint[0], singleData))
                index += 1
        S2UsableData = []    
        s2TotalData = []
                
        #computes the S2 Value for all Biomarkers
        for protein in proteinDict:
            dataList2 = proteinDict[protein]

            senstivity = 0.0
            specificity = 0.0
            dPos = 0.0
            dNeg = 0.0

            for dataPoint in dataList2:
                if (dataPoint[0]):
                    dPos += 1
                    if (float(dataPoint[1]) > threshold):
                        senstivity += 1
                if (dataPoint[0] == False): 
                    dNeg += 1
                    if (float(dataPoint[1]) <= threshold):
                        specificity += 1
            p_Y1_D1 = senstivity/dPos
            p_Y0_D0 = specificity/dNeg
            s2 = p_Y0_D0 * p_Y1_D1
            proteinDict[protein] = s2
            s2TotalData.append(s2)
        S2UsableData = s2TotalData.copy()
        s2TotalData.sort()

        averageS2 = np.average(S2UsableData)
        usableProteins = []
        #creates list of proteins with s2 metric > avg s2 metric
        for bioMarker in proteinDict:
            if proteinDict[bioMarker] > averageS2:
                usableProteins.append(bioMarker)

        #ProteinData = dict with "BioMarker Name": [tempData]
        #D_r_Cases = dict with "BioMarker Name": [samples where r/c > 2] (each sample is generic and common to every protein that is in that sample)
        ProteinData = {"test":[]}
        ProteinData.pop("test")
        D_r_Cases = {"test": []}
        D_r_Cases.pop("test")
        for bioMarker in usableProteins:
            #tempData = list containing (detection of disease by bioMarker (r), sample details (age, location, gender, etc)) for each sample in the current fold
            tempData = [(float, [])]
            tempData.pop(0)
            tempD_r_ = [[]]
            tempD_r_.pop(0)
            indexOfBioMarker = proteinList.index(bioMarker) + 2
            for index in indexs:
                c = dataList[index][indexOfBioMarker]
                sampleData = dataList[index][1]
                tempData.append((c, sampleData))
                if float(c) > threshold:
                    tempD_r_.append(sampleData)

            ProteinData[bioMarker] = tempData
            D_r_Cases[bioMarker] = tempD_r_
        #print(*ProteinData["HpCD00780232"], *D_r_Cases["HpCD00780232"], indexs)

        #finds the amount of shared cases between two proteins and if # is above required amount returns True else False
        def Similiarity(Protein1, Protein2):
                dataSet1 = D_r_Cases[Protein1]
                dataSet2 = D_r_Cases[Protein2]
                sharedSamples = 0
                minimumSamples = 1
                for data in dataSet1:
                    if data in dataSet2:
                        sharedSamples += 1
                if sharedSamples >= minimumSamples:
                    return True
                else:
                    return False
            
        #takes in a R_r P(R, r) which is all the associated proteins and their data for one single protein and the cooressponding P(~R, r) and computes/returns the eAVG for that Protein
        def compute(R_r, NotR_r):
                sensR_r = 0.0
                specR_r = 0.0
                sensR_rDenom = 0.0
                specR_rDenom = 0.0

                sensNotR_r = 0.0
                specNotR_r = 0.0
                sensNotR_rDenom = 0.0
                specNotR_rDenom = 0.0

                eAVGNum = 0.0
                eAVGDenom = len(R_r)
                for index in range(len(R_r)):
                    a = R_r[index][1:]
                    for dataPoint in R_r[index][1:]:
                        if dataPoint[0] == "1":
                            sensR_rDenom += 1
                            if dataPoint[1]:
                                sensR_r += 1
                        else:
                            specR_rDenom += 1
                            if not dataPoint[1]:
                                specR_r += 1

                    for dataPoint in NotR_r[index][1:]:
                        if dataPoint[0] == "1":
                            sensNotR_rDenom += 1
                            if dataPoint[1]:
                                sensNotR_r += 1
                        else:
                            specNotR_rDenom += 1
                            if not dataPoint[1]:
                                specNotR_r += 1                

                    eAVGNum += ((sensR_r/sensR_rDenom) * (specR_r/specR_rDenom)) - ((sensNotR_r/sensNotR_rDenom) * (specNotR_r/specNotR_rDenom))
                if eAVGDenom != 0:
                    eAVG = eAVGNum / eAVGDenom
                else:
                    eAVG = 0
                return eAVG
        DisjunctiveEAVG = {}

        runtime = 1
        #iterates through each Protein that has S2 > average S2   [:30] -> first 30 Proteins only for testing
        for Protein in usableProteins:
                if (runtime % 10 == 0):
                    print(runtime)
                runtime += 1

                #temporary storage for values stored as (Case or Control, Detectence of Both Proteins)
                primaryTempDisjunctiveR_r = []
                primaryTempDisjunctiveNotR_r = []

                #iterates through every Protein for each Protein
                for SecondaryProtein in usableProteins:

                    #checks if Protein and Secondary Protein share a sample which they both detected
                    if (Similiarity(Protein, SecondaryProtein) and Protein != SecondaryProtein):
                        tempDisjunctiveR_r_ = [SecondaryProtein]
                        tempDisjunctiveNotR_r_ = [SecondaryProtein]

                        ProteinCalculateData = ProteinData[Protein]
                        SecondaryProteinCalculateData = ProteinData[SecondaryProtein]

                        #Iterates through every data set (240 samples if 5 folds) and adds to Conjunctive and Disjunctive R_r and ~R_r in form of ("Case / Control", True/False)
                        #True/False -> alters depending on if conjunctive or disjunctive and if R_r or ~R_r 
                        for index in range(len(ProteinCalculateData)):
                            sampleType = ProteinCalculateData[index][1]
                            tempDisjunctiveR_r_.append((sampleType, float(ProteinCalculateData[index][0]) > threshold or float(SecondaryProteinCalculateData[index][0]) > threshold))
                            tempDisjunctiveNotR_r_.append((sampleType,float(ProteinCalculateData[index][0]) < threshold or float(SecondaryProteinCalculateData[index][0]) > threshold ))
                            # p(r'): tempDisjunctiveNotR_r_.append((sampleType,float(SecondaryProteinCalculateData[index][0]) > threshold ))
                            #tempDisjunctiveNotR_r_.append((sampleType,float(ProteinCalculateData[index][0]) < threshold and float(SecondaryProteinCalculateData[index][0]) > threshold ))



                        #Adds the current Protein and Secondary Protein Specific data to overall set correlated with all R(r) for Protein R
                        primaryTempDisjunctiveR_r.append(tempDisjunctiveR_r_)
                        primaryTempDisjunctiveNotR_r.append(tempDisjunctiveNotR_r_)

                DisjunctiveEAVG[Protein] = compute(primaryTempDisjunctiveR_r,primaryTempDisjunctiveNotR_r)
            #print(ConjunctiveEAVG[Protein])
            #print(DisjunctiveEAVG[Protein])
            
        DisjunctiveEAVGList = list(DisjunctiveEAVG.values())
        DisjunctiveEAVGList.sort()
        
        for value in DisjunctiveEAVG.keys():
            bioMarkerCausalValues[value].append(DisjunctiveEAVG[value])
        
        top10Disjunctive = DisjunctiveEAVGList[-10:]
        top15Disjunctive = DisjunctiveEAVGList[-20:]
        top30Disjunctive = DisjunctiveEAVGList[-30:]


        top10S2 = s2TotalData[-10:]
        top30S2 = s2TotalData[-30:]

        top10DisjunctiveBioMarkers = []
        top15DisjunctiveBioMarkers = []
        top30DisjunctiveBioMarkers = []
    

        top10S2BioMarkers = []
        top30S2BioMarkers = []

                #created List shown above with top 10/30 bioMarkers depending on Conjunctive/Disjunctive/S2 - - - List contains names of BioMarkers 
        for index in range(30): #30
            if index < 10: #10
                top10DisjunctiveBioMarkers.append(list(DisjunctiveEAVG.keys())[list(DisjunctiveEAVG.values()).index(top10Disjunctive[index])])
            if index < 15: #15
                top15DisjunctiveBioMarkers.append(list(DisjunctiveEAVG.keys())[list(DisjunctiveEAVG.values()).index(top15Disjunctive[index])])
            top30DisjunctiveBioMarkers.append(list(DisjunctiveEAVG.keys())[list(DisjunctiveEAVG.values()).index(top30Disjunctive[index])])
        totalBiomarkers.append(top30DisjunctiveBioMarkers)
        top1DisjunctiveBioMarkers = top30DisjunctiveBioMarkers[-1:]
        top3DisjunctiveBioMarkers = top30DisjunctiveBioMarkers[-3:]
        top4DisjunctiveBioMarkers = top30DisjunctiveBioMarkers[-4:]

        def decisionTreeSample(bioMarkerList, type):
            clfTrainData = [[]]
            clfTrainData.pop(0)
            clfAffectedData = []

            for dataSet in trainTotalData:
                    currentSet = []
                    #print(len(dataSet[1]))
                    for biomarker in bioMarkerList:
                        temp = float(dataSet[1][proteinList.index(biomarker)])
                        currentSet.append(temp)
                    clfTrainData.append(currentSet)
                    clfAffectedData.append(dataSet[0])
            #clf = RandomForestClassifier(n_estimators=10)
            #clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1)
            """params = {
            'max_depth': 2,
            'learning_rate': 1.0,
            'n_estimators':10,
            'random_state':0
            }
            clf = XGBClassifier(**params)"""
            clf = GradientBoostingClassifier(n_estimators=10, learning_rate=learningrate,max_depth=1, random_state=0)
            clf = clf.fit(clfTrainData, clfAffectedData)

            """            dot_data = export_graphviz(clf.estimators_[0],filled=True, impurity=True, rounded=True) 
            graph = graphviz.Source(dot_data, format = 'png')
            file_name = "Random Forest Model" + type 
            graph.render(file_name)"""

            clfTestData = [(False, [])]
            clfTestData.pop(0)
            clfTestAffectedRealData = []
            for set in testData:
                currentSet = []
                #print(len(dataSet[1]))
                for biomarker in bioMarkerList:
                    temp = float(set[1][proteinList.index(biomarker)])
                    currentSet.append(temp)
                    #currentSet.append(set[1][proteinList.index(biomarker)])
                clfTestData.append((set[0],currentSet))
                clfTestAffectedRealData.append(set[0])
            #print(*clfTestData)


            """            counter = 0
                        for index in testIndexs:
                            tempSampleData = sampleDataFile[index]
                            isIgG = "IgG" in tempSampleData[0]
                            isIgA = "IgA" in tempSampleData[0]
                            is30_39 = float(tempSampleData[1]) >= 30 and float(tempSampleData[1]) <= 39
                            is81_ = float(tempSampleData[1]) >= 81
                            isM = "M" in tempSampleData[2]
                            isF = "F" in tempSampleData[2]

                            clfTestData[counter][1].extend([isIgG,isIgA,is30_39,is81_,isM,isF])
                            counter+=1"""
            
            x_test = []
            y_test = []

            algoResults = [()]
            algoResults.pop(0)
            for predict in clfTestData:
                programPredict = clf.predict([predict[1]])
                predictProba = clf.predict_proba([predict[1]])[:,1]
                realData = predict[0]
                if "10 Disj" in type and "S" not in type:
                    rocCurveYtest.append(realData)
                    rocCurveYProba.append(predictProba)
                algoResults.append((programPredict,realData))

            def getConfidenceScoresRF():
                                """                                #splits RF to indvidual trees
                                dt1 = clf.estimators_[0]
                                dt2 = clf.estimators_[1]
                                dt3 = clf.estimators_[2]
                                dt4 = clf.estimators_[3]
                                dt5 = clf.estimators_[4]
                                dt6 = clf.estimators_[5]
                                dt7 = clf.estimators_[6]
                                dt8 = clf.estimators_[7]
                                dt9 = clf.estimators_[8]
                                dt10 = clf.estimators_[9]

                                predictTestData = clfTestData[0]
                                groundTruth = predictTestData[0]
                                dTrees = [dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8,dt9,dt10]
                                confidenceScore = 0
                                #alternate method to calculate ROC curves for RF Model
                                for tree in dTrees:
                                    predicition = tree.predict([predictTestData[1]])
                                    score = predicition[0] + 1
                                    score = score / 2
                                    confidenceScore += score
                                confidenceScore /= 10.0"""
                                confidenceScore = 0
                                predictTestData = clfTestData[0]
                                groundTruth = predictTestData[0]
                                #based on which method running, appends predictProba Scores to correct array
                                if "10 Disj" in type and "S" not in type:
                                    roc10.append((confidenceScore,groundTruth))
                                    disj10Proba.append((realData,predictProba))
                                elif "15 Disj" in type and "S" not in type:
                                    roc15.append((confidenceScore,groundTruth))
                                    disj15Proba.append((realData,predictProba))
                                elif "30 Disj" in type and "S" not in type:
                                    roc30.append((confidenceScore,groundTruth))
                                    disj30Proba.append((realData,predictProba))
                                elif "10 Disj" in type and "S" in type:
                                    roc10wS.append((confidenceScore,groundTruth))
                                    disj10wSProba.append((realData,predictProba))
                                elif "15 Disj" in type and "S"  in type:
                                    roc15wS.append((confidenceScore,groundTruth))
                                    disj15wSProba.append((realData,predictProba))
                                elif "30 Disj" in type and "S"  in type:
                                    roc30wS.append((confidenceScore,groundTruth))
                                    disj30wSProba.append((realData,predictProba))
            def getConfidenceScores():
                                dt1 = clf.estimators_[0]
                                dt2 = clf.estimators_[1]
                                dt3 = clf.estimators_[2]
                                dt4 = clf.estimators_[3]
                                dt5 = clf.estimators_[4]
                                dt6 = clf.estimators_[5]
                                dt7 = clf.estimators_[6]
                                dt8 = clf.estimators_[7]
                                dt9 = clf.estimators_[8]
                                dt10 = clf.estimators_[9]

                                predictTestData = clfTestData[0]
                                groundTruth = predictTestData[0]
                                dTrees = [dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8,dt9,dt10]
                                confidenceScore = 0

                                for tree in dTrees:
                                    predicition = tree[0].predict([predictTestData[1]])
                                    score = predicition[0] + 1
                                    score = score / 2
                                    confidenceScore += score
                                confidenceScore /= 10.0
                                if "10 Disj" in type and "S" not in type:
                                    roc10.append((confidenceScore,groundTruth))
                                    disj10Proba.append((realData,predictProba))
                                elif "15 Disj" in type and "S" not in type:
                                    roc15.append((confidenceScore,groundTruth))
                                    disj15Proba.append((realData,predictProba))
                                elif "30 Disj" in type and "S" not in type:
                                    roc30.append((confidenceScore,groundTruth))
                                    disj30Proba.append((realData,predictProba))
                                elif "10 Disj" in type and "S" in type:
                                    roc10wS.append((confidenceScore,groundTruth))
                                    disj10wSProba.append((realData,predictProba))
                                elif "15 Disj" in type and "S"  in type:
                                    roc15wS.append((confidenceScore,groundTruth))
                                    disj15wSProba.append((realData,predictProba))
                                elif "30 Disj" in type and "S"  in type:
                                    roc30wS.append((confidenceScore,groundTruth))
                                    disj30wSProba.append((realData,predictProba))
                    


            getConfidenceScoresRF()
            recallNum = 0.0
            precisionNum = 0.0
            specNum = 0.0

            recallDenom = 0.0
            precisionDenom = 0.0
            specDenom = 0.0

            for result in algoResults:
                programPredict = result[0]
                realData = result[1]
                if (programPredict and realData):
                    recallNum += 1.0
                    precisionNum += 1.0
                if (realData):
                    recallDenom += 1.0
                if (programPredict):
                    precisionDenom += 1.0
                #for Specificity 
                if (programPredict == False and realData == False):
                    specNum += 1.0
                if (realData == False):
                    specDenom += 1.0
            
            if recallDenom == 0:
                recallDenom = 1
                recallNum = -1
            if precisionDenom == 0:
                precisionDenom = 1
                precisionNum = -1
            if specDenom == 0:
                specDenom = 1
                specNum = -1

            if "none" in type:
                viz = RocCurveDisplay.from_estimator(
                    clf,
                    x_test,
                    y_test,
                    name="Fold " + str(len(tprs) + 1),
                    alpha=0.3,
                    lw=1,
                    ax=ax,
                )
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            #fig, ax = plt.subplots()
            #rfc_disp = RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax,name=type[24:])
            rfc_disp = ""

            recall = recallNum/recallDenom
            precision = precisionNum/precisionDenom
            specificity = specNum/specDenom

            treeDepth =  "n/a"
            treeLeafNodes = "n/a"

            return ([type, recall, precision, specificity,bioMarkerList, treeDepth, treeLeafNodes], rfc_disp)

        def decisionTree(bioMarkerList, type):
            clfTrainData = [[]]
            clfTrainData.pop(0)
            clfAffectedData = []

            for dataSet in trainTotalData:
                    currentSet = []
                    #print(len(dataSet[1]))
                    for biomarker in bioMarkerList:
                        temp = float(dataSet[1][proteinList.index(biomarker)])
                        currentSet.append(temp)
                    clfTrainData.append(currentSet)
                    clfAffectedData.append(dataSet[0])
            #clf = RandomForestClassifier(n_estimators=10)
            clf = GradientBoostingClassifier(n_estimators=10, learning_rate=learningrate,max_depth=1, random_state=0)
            #clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1)
            """params = {
            'max_depth': 2,
            'learning_rate': 1.0,
            'n_estimators':10,
            'random_state':0
            }
            clf = XGBClassifier(**params)"""
            clf = clf.fit(clfTrainData, clfAffectedData)

            """            dot_data = export_graphviz(clf.estimators_[0],filled=True, impurity=True, rounded=True) 
            graph = graphviz.Source(dot_data, format = 'png')
            file_name = "Random Forest Model" + type 
            graph.render(file_name)"""

            clfTestData = [(False, [])]
            clfTestData.pop(0)
            clfTestAffectedRealData = []

            for set in testData:
                currentSet = []
                #print(len(dataSet[1]))
                for biomarker in bioMarkerList:
                    temp = float(set[1][proteinList.index(biomarker)])
                    currentSet.append(temp)
                    #currentSet.append(set[1][proteinList.index(biomarker)])
                clfTestData.append((set[0],currentSet))
                clfTestAffectedRealData.append(set[0])
            #print(*clfTestData)
            
            x_test = []
            y_test = []

            algoResults = [()]
            algoResults.pop(0)
            for predict in clfTestData:
                programPredict = clf.predict([predict[1]])
                predictProba = clf.predict_proba([predict[1]])[:,1]
                realData = predict[0]
                if "10 Disj" in type and "S" not in type:
                    rocCurveYtest.append(realData)
                    rocCurveYProba.append(predictProba)
                algoResults.append((programPredict,realData))

            def getConfidenceScoresRF():
                                """                                #splits RF to indvidual trees
                                dt1 = clf.estimators_[0]
                                dt2 = clf.estimators_[1]
                                dt3 = clf.estimators_[2]
                                dt4 = clf.estimators_[3]
                                dt5 = clf.estimators_[4]
                                dt6 = clf.estimators_[5]
                                dt7 = clf.estimators_[6]
                                dt8 = clf.estimators_[7]
                                dt9 = clf.estimators_[8]
                                dt10 = clf.estimators_[9]

                                predictTestData = clfTestData[0]
                                groundTruth = predictTestData[0]
                                dTrees = [dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8,dt9,dt10]
                                confidenceScore = 0
                                #alternate method to calculate ROC curves for RF Model
                                for tree in dTrees:
                                    predicition = tree.predict([predictTestData[1]])
                                    score = predicition[0] + 1
                                    score = score / 2
                                    confidenceScore += score
                                confidenceScore /= 10.0"""
                                confidenceScore = 0
                                predictTestData = clfTestData[0]
                                groundTruth = predictTestData[0]
                                #based on which method running, appends predictProba Scores to correct array
                                if "10 Disj" in type and "S" not in type:
                                    roc10.append((confidenceScore,groundTruth))
                                    disj10Proba.append((realData,predictProba))
                                elif "15 Disj" in type and "S" not in type:
                                    roc15.append((confidenceScore,groundTruth))
                                    disj15Proba.append((realData,predictProba))
                                elif "30 Disj" in type and "S" not in type:
                                    roc30.append((confidenceScore,groundTruth))
                                    disj30Proba.append((realData,predictProba))
                                elif "10 Disj" in type and "S" in type:
                                    roc10wS.append((confidenceScore,groundTruth))
                                    disj10wSProba.append((realData,predictProba))
                                elif "15 Disj" in type and "S"  in type:
                                    roc15wS.append((confidenceScore,groundTruth))
                                    disj15wSProba.append((realData,predictProba))
                                elif "30 Disj" in type and "S"  in type:
                                    roc30wS.append((confidenceScore,groundTruth))
                                    disj30wSProba.append((realData,predictProba))
            def getConfidenceScores():
                                dt1 = clf.estimators_[0]
                                dt2 = clf.estimators_[1]
                                dt3 = clf.estimators_[2]
                                dt4 = clf.estimators_[3]
                                dt5 = clf.estimators_[4]
                                dt6 = clf.estimators_[5]
                                dt7 = clf.estimators_[6]
                                dt8 = clf.estimators_[7]
                                dt9 = clf.estimators_[8]
                                dt10 = clf.estimators_[9]

                                predictTestData = clfTestData[0]
                                groundTruth = predictTestData[0]
                                dTrees = [dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8,dt9,dt10]
                                confidenceScore = 0

                                for tree in dTrees:
                                    predicition = tree[0].predict([predictTestData[1]])
                                    score = predicition[0] + 1
                                    score = score / 2
                                    confidenceScore += score
                                confidenceScore /= 10.0
                                if "10 Disj" in type and "S" not in type:
                                    roc10.append((confidenceScore,groundTruth))
                                    disj10Proba.append((realData,predictProba))
                                elif "15 Disj" in type and "S" not in type:
                                    roc15.append((confidenceScore,groundTruth))
                                    disj15Proba.append((realData,predictProba))
                                elif "30 Disj" in type and "S" not in type:
                                    roc30.append((confidenceScore,groundTruth))
                                    disj30Proba.append((realData,predictProba))
                                elif "10 Disj" in type and "S" in type:
                                    roc10wS.append((confidenceScore,groundTruth))
                                    disj10wSProba.append((realData,predictProba))
                                elif "15 Disj" in type and "S"  in type:
                                    roc15wS.append((confidenceScore,groundTruth))
                                    disj15wSProba.append((realData,predictProba))
                                elif "30 Disj" in type and "S"  in type:
                                    roc30wS.append((confidenceScore,groundTruth))
                                    disj30wSProba.append((realData,predictProba))
                    


            getConfidenceScoresRF()

            recallNum = 0.0
            precisionNum = 0.0
            specNum = 0.0

            recallDenom = 0.0
            precisionDenom = 0.0
            specDenom = 0.0

            for result in algoResults:
                programPredict = result[0]
                realData = result[1]
                if (programPredict and realData):
                    recallNum += 1.0
                    precisionNum += 1.0
                if (realData):
                    recallDenom += 1.0
                if (programPredict):
                    precisionDenom += 1.0
                #for Specificity 
                if (programPredict == False and realData == False):
                    specNum += 1.0
                if (realData == False):
                    specDenom += 1.0
            
            if recallDenom == 0:
                recallDenom = 1
                recallNum = -1
            if precisionDenom == 0:
                precisionDenom = 1
                precisionNum = -1
            if specDenom == 0:
                specDenom = 1
                specNum = -1
            
            #change "none" to "10" to run

            #rfc_disp = RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax,name=type[24:])
            rfc_disp = ""

            recall = recallNum/recallDenom
            precision = precisionNum/precisionDenom
            specificity = specNum/specDenom

            treeDepth =  "n/a"
            treeLeafNodes = "n/a"

            return ([type, recall, precision, specificity,bioMarkerList, treeDepth, treeLeafNodes], rfc_disp)

        foldReturnData = []

        top10DisjResults = decisionTree(top10DisjunctiveBioMarkers, "Fold " + str(counter) + " ~ Decision Tree ~ Top 10 Disj")
        top15DisjResults = decisionTree(top15DisjunctiveBioMarkers, "Fold " + str(counter) + " ~ Decision Tree ~ Top 15 Disj")
        top30DisjResults = decisionTree(top30DisjunctiveBioMarkers, "Fold " + str(counter) + " ~ Decision Tree ~ Top 30 Disj")

        top10DisjResultsSample = decisionTreeSample(top1DisjunctiveBioMarkers, "Fold " + str(counter) + " ~ Decision Tree ~ Top 10 Disj w/ S")
        top15DisjResultsSample = decisionTreeSample(top3DisjunctiveBioMarkers, "Fold " + str(counter) + " ~ Decision Tree ~ Top 15 Disj w/ S")
        top30DisjResultsSample = decisionTreeSample(top4DisjunctiveBioMarkers, "Fold " + str(counter) + " ~ Decision Tree ~ Top 30 Disj w/ S")

        foldReturnData.append(top10DisjResults[0])
        foldReturnData.append(top30DisjResults[0])
        foldReturnData.append(top10DisjResultsSample[0])
        foldReturnData.append(top30DisjResultsSample[0])

        foldReturnData.append(top10DisjResults[1])
        foldReturnData.append(top30DisjResults[1])
        foldReturnData.append(top10DisjResultsSample[1])
        foldReturnData.append(top30DisjResultsSample[1])

        foldReturnData.append(top15DisjResults[0])
        foldReturnData.append(top15DisjResultsSample[0])
        foldReturnData.append(top15DisjResults[1])
        foldReturnData.append(top15DisjResultsSample[1])
        
        return foldReturnData




    finalTop10DisjunctiveResults = []
    finalTop15DisjunctiveResults = []
    finalTop30DisjunctiveResults = []

    finalTop10DisjunctiveSampleResults = []
    finalTop15DisjunctiveSampleResults = []
    finalTop30DisjunctiveSampleResults = []

    finalBaselineResults = []
    finalBaselineRecall = []
    finalBaselinePrecision = []
    finalBaselineSpecificity = []

    finalFoldTestSet = []
    counter = 1
    #newfolds = [folds[4]]
    for set in folds:
        #print(set[0][0], "    ", set[0][-1])
        #print(set[1])
        trainData = [(True,[])]
        trainTotalData = [[]]
        affectedData = []
        trainTotalData.pop(0)
        trainData.pop(0)

        testData = [(True,[])]
        #testTotalData = [[]]
        testData.pop(0)
        #testTotalData.pop(0)
        testAffectedData = []
        testCombinedCausalityData = []

        baselineTrain = []
        baselineAffected = []
        for x in set[0]:
            if (dataList[x][1] == "0"):
                affected = False
                affectedData.append(False)
            if (dataList[x][1] == "1"):
                affected = True
                affectedData.append(True)
            tempArray = []
            testing = []
            for point in dataList[x][2:]:
                if float(point) > threshold:
                    tempArray.append(1)
                    testing.append((point,1))
                else:
                    tempArray.append(0)
                    testing.append((point,0))
            baselineTrain.append(tempArray)
            baselineAffected.append(affected)
            trainData.append(tempArray)
            trainTotalData.append((affected, dataList[x][2:]))
            #print(*trainTotalData)
        for x in set[1]:
            if (dataList[x][1] == '0'):
                testAffectedData.append(False)
                affected = False
            if (dataList[x][1] == '1'):
                testAffectedData.append(True)
                affected = True
            tempArray = []
            for point in dataList[x][2:]:       
                if float(point) > threshold:
                    tempArray.append(1)
                else:
                    tempArray.append(0)
            testData.append((affected, tempArray))
            testCombinedCausalityData.append((affected, dataList[x][2:]))
        #print(*trainData[-1])
        #print(affectedData[-1])
        #clf = RandomForestClassifier(n_estimators=10)
        #clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1)
        clf = GradientBoostingClassifier(n_estimators=10, learning_rate=learningrate,max_depth=maxDepth, random_state=0)
        clf = clf.fit(baselineTrain, baselineAffected)
        
        #recall, precision, specificity
        finalFoldTestSet.append(set[1])
        indexs = set[0]
        testIndex = set[1]
        
        print("Fold: " + str(counter) + " ~ Baseline Approach")
        baselineResults = baselineApproach(clf, trainData, testData, testAffectedData)
        baseline_disp = baselineResults[1]
        baselineResults = baselineResults[0]        
        baselineResults.append("n/a")
        baselineResults.append("n/a")
        finalBaselineResults.append(baselineResults)
        finalBaselineRecall.append([baselineResults[0]])
        finalBaselinePrecision.append([baselineResults[1]])
        finalBaselineSpecificity.append([baselineResults[2]])
        

        combinedResults = combinedCausalityApproach(indexs,trainTotalData,counter,testCombinedCausalityData,testIndex)
        finalTop10DisjunctiveResults.append(combinedResults[0])
        finalTop30DisjunctiveResults.append(combinedResults[1])

        finalTop10DisjunctiveSampleResults.append(combinedResults[2])
        finalTop30DisjunctiveSampleResults.append(combinedResults[3])

        disj10_disp = combinedResults[4]
        disj30_disp = combinedResults[5]
        disj10Sample_disp = combinedResults[6]
        disj30Sample_disp = combinedResults[7]

        finalTop15DisjunctiveResults.append(combinedResults[8])
        finalTop15DisjunctiveSampleResults.append(combinedResults[9])

        disj15_disp = combinedResults[10]
        disj15Sample_disp = combinedResults[11]

        typertwo = 0
        if typertwo == 10:
            fig, ax = plt.subplots()

            baseline_disp.plot(ax=ax)
            disj10_disp.plot(ax=ax)
            disj30_disp.plot(ax=ax)
            disj10Sample_disp.plot(ax=ax)
            disj30Sample_disp.plot(ax=ax)

            disj15_disp.plot(ax=ax)
            disj15Sample_disp.plot(ax=ax)

            Plotname = str(threshold) + " ~ ROC Curves Comparision ~ Fold 5"

            ax.set_title(Plotname)
            ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            ax.set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            plt.savefig(Plotname + ".png")
        

        counter += 1
        

        print("done")   
        print()

    def makeROC():
        fpr, tpr, thresholds = roc_curve(rocCurveYtest, rocCurveYProba)
        plt.plot(fpr, tpr, label="Disj 10")

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig('1.png')
        plt.show()  

        with open('predictProbaScores' + str(method) + str(threshold) + '_50.csv','w') as csvfile:
            rocPrinter = [["Fold","d10_y_test","d10_y_proba","d15_y_test","d15_y_proba","d30_y_test","d30_y_proba","d10wS_y_test","d10wS_y_proba","d15wS_y_test","d15wS_y_proba","d30wS_y_test","d30wS_y_proba"]]
            for fold in range(len(disj10Proba)):
                rocPrinter.append([
                    fold,
                    disj10Proba[fold][0],disj10Proba[fold][1][0],
                    disj15Proba[fold][0],disj15Proba[fold][1][0],
                    disj30Proba[fold][0],disj30Proba[fold][1][0],
                    disj10wSProba[fold][0],disj10wSProba[fold][1][0],
                    disj15wSProba[fold][0],disj15wSProba[fold][1][0],
                    disj30wSProba[fold][0],disj30wSProba[fold][1][0]
                ])
            ROCwriter = cs.writer(csvfile, delimiter=',',
                quotechar='|', quoting=cs.QUOTE_MINIMAL)
            #filewriter.writerows(fieldName)
            ROCwriter.writerows(rocPrinter)   
    makeROC()
    print("All Folds Done")

    #print(*finalCombinedBioMarkers)
    def finalComputation():
        
        finalComputeBaselinePrecision = []
        finalComputeBaselineRecall = []
        finalComputeBaselineSpecificity = []

        for x in range(len(finalBaselinePrecision)):
            if finalBaselinePrecision[x] != -1:
                finalComputeBaselinePrecision.append(finalBaselinePrecision[x])
        for x in range(len(finalBaselineRecall)):
            if finalBaselineRecall[x] != -1:
                finalComputeBaselineRecall.append(finalBaselineRecall[x])
        for x in range(len(finalBaselineSpecificity)):
            if finalBaselineSpecificity[x] != -1:
                finalComputeBaselineSpecificity.append(finalBaselineSpecificity[x])

        avgBaselineRecall = np.average(finalComputeBaselineRecall)
        avgBaselinePrecision = np.average(finalComputeBaselinePrecision)
        avgBaselineSpecificity = np.average(finalComputeBaselineSpecificity)

        sdBaselineRecall = np.std(finalComputeBaselineRecall)
        sdBaselinePrecision = np.std(finalComputeBaselinePrecision)
        sdBaselineSpecificity = np.std(finalComputeBaselineSpecificity)

        top10DisjunctiveRecall = []
        top15DisjunctiveRecall = []
        top30DisjunctiveRecall = []

        top10DisjunctivePrecision = []
        top15DisjunctivePrecision = []
        top30DisjunctivePrecision = []

        top10DisjunctiveSpecificity = []
        top15DisjunctiveSpecificity = []
        top30DisjunctiveSpecificity = []

        top10DisjunctiveSampleRecall = []
        top15DisjunctiveSampleRecall = []
        top30DisjunctiveSampleRecall = []

        top10DisjunctiveSamplePrecision = []
        top15DisjunctiveSamplePrecision = []
        top30DisjunctiveSamplePrecision = []

        top10DisjunctiveSampleSpecificity = []
        top15DisjunctiveSampleSpecificity = []
        top30DisjunctiveSampleSpecificity = []
        

        for index in range(len(finalTop10DisjunctiveResults)):
            top10DisjunctiveRecall.append(finalTop10DisjunctiveResults[index][1])
            top15DisjunctiveRecall.append(finalTop15DisjunctiveResults[index][1])
            top30DisjunctiveRecall.append(finalTop30DisjunctiveResults[index][1])

            top10DisjunctivePrecision.append(finalTop10DisjunctiveResults[index][2])
            top15DisjunctivePrecision.append(finalTop15DisjunctiveResults[index][2])
            top30DisjunctivePrecision.append(finalTop30DisjunctiveResults[index][2])
            

            top10DisjunctiveSpecificity.append(finalTop10DisjunctiveResults[index][3])
            top15DisjunctiveSpecificity.append(finalTop15DisjunctiveResults[index][3])
            top30DisjunctiveSpecificity.append(finalTop30DisjunctiveResults[index][3])

            top10DisjunctiveSampleRecall.append(finalTop10DisjunctiveSampleResults[index][1])
            top15DisjunctiveSampleRecall.append(finalTop15DisjunctiveSampleResults[index][1])
            top30DisjunctiveSampleRecall.append(finalTop30DisjunctiveSampleResults[index][1])

            top10DisjunctiveSamplePrecision.append(finalTop10DisjunctiveSampleResults[index][2])
            top15DisjunctiveSamplePrecision.append(finalTop15DisjunctiveSampleResults[index][2])
            top30DisjunctiveSamplePrecision.append(finalTop30DisjunctiveSampleResults[index][2])
            

            top10DisjunctiveSampleSpecificity.append(finalTop10DisjunctiveSampleResults[index][3])
            top15DisjunctiveSampleSpecificity.append(finalTop15DisjunctiveSampleResults[index][3])
            top30DisjunctiveSampleSpecificity.append(finalTop30DisjunctiveSampleResults[index][3])

        

        top10ComputeDisjunctiveRecall = []
        top15ComputeDisjunctiveRecall = []
        top30ComputeDisjunctiveRecall = []
 
        top10ComputeDisjunctivePrecision = []
        top15ComputeDisjunctivePrecision = []
        top30ComputeDisjunctivePrecision = []
 
        top10ComputeDisjunctiveSpecificity = []
        top15ComputeDisjunctiveSpecificity = []
        top30ComputeDisjunctiveSpecificity = []
 
        top10ComputeDisjunctiveSampleRecall = []
        top15ComputeDisjunctiveSampleRecall = []
        top30ComputeDisjunctiveSampleRecall = []
 
        top10ComputeDisjunctiveSamplePrecision = []
        top15ComputeDisjunctiveSamplePrecision = []
        top30ComputeDisjunctiveSamplePrecision = []
 
        top10ComputeDisjunctiveSampleSpecificity = []
        top15ComputeDisjunctiveSampleSpecificity = []
        top30ComputeDisjunctiveSampleSpecificity = []



        #formatting 
        for t in range(1):
            for x in range(len(top10DisjunctivePrecision)):
                if top10DisjunctivePrecision[x] != -1:
                    top10ComputeDisjunctivePrecision.append(top10DisjunctivePrecision[x])
            for x in range(len(top10DisjunctiveRecall)):
                if top10DisjunctiveRecall[x] != -1:
                    top10ComputeDisjunctiveRecall.append(top10DisjunctiveRecall[x])
            for x in range(len(top10DisjunctiveSpecificity)):
                if top10DisjunctiveSpecificity[x] != -1:
                    top10ComputeDisjunctiveSpecificity.append(top10DisjunctiveSpecificity[x])
    
            for x in range(len(top15DisjunctivePrecision)):
                if top15DisjunctivePrecision[x] != -1:
                    top15ComputeDisjunctivePrecision.append(top15DisjunctivePrecision[x])
            for x in range(len(top15DisjunctiveRecall)):
                if top15DisjunctiveRecall[x] != -1:
                    top15ComputeDisjunctiveRecall.append(top15DisjunctiveRecall[x])
            for x in range(len(top15DisjunctiveSpecificity)):
                if top15DisjunctiveSpecificity[x] != -1:
                    top15ComputeDisjunctiveSpecificity.append(top15DisjunctiveSpecificity[x])
    
            for x in range(len(top30DisjunctivePrecision)):
                if top30DisjunctivePrecision[x] != -1:
                    top30ComputeDisjunctivePrecision.append(top30DisjunctivePrecision[x])
            for x in range(len(top30DisjunctiveRecall)):
                if top30DisjunctiveRecall[x] != -1:
                    top30ComputeDisjunctiveRecall.append(top30DisjunctiveRecall[x])
            for x in range(len(top30DisjunctiveSpecificity)):
                if top30DisjunctiveSpecificity[x] != -1:
                    top30ComputeDisjunctiveSpecificity.append(top30DisjunctiveSpecificity[x])

            for x in range(len(top10DisjunctiveSamplePrecision)):
                if top10DisjunctiveSamplePrecision[x] != -1:
                    top10ComputeDisjunctiveSamplePrecision.append(top10DisjunctiveSamplePrecision[x])
            for x in range(len(top10DisjunctiveSampleRecall)):
                if top10DisjunctiveSampleRecall[x] != -1:
                    top10ComputeDisjunctiveSampleRecall.append(top10DisjunctiveSampleRecall[x])
            for x in range(len(top10DisjunctiveSampleSpecificity)):
                if top10DisjunctiveSampleSpecificity[x] != -1:
                    top10ComputeDisjunctiveSampleSpecificity.append(top10DisjunctiveSampleSpecificity[x])
    
            for x in range(len(top15DisjunctiveSamplePrecision)):
                if top15DisjunctiveSamplePrecision[x] != -1:
                    top15ComputeDisjunctiveSamplePrecision.append(top15DisjunctiveSamplePrecision[x])
            for x in range(len(top15DisjunctiveSampleRecall)):
                if top15DisjunctiveSampleRecall[x] != -1:
                    top15ComputeDisjunctiveSampleRecall.append(top15DisjunctiveSampleRecall[x])
            for x in range(len(top15DisjunctiveSampleSpecificity)):
                if top15DisjunctiveSampleSpecificity[x] != -1:
                    top15ComputeDisjunctiveSampleSpecificity.append(top15DisjunctiveSampleSpecificity[x])
    
            for x in range(len(top30DisjunctiveSamplePrecision)):
                if top30DisjunctiveSamplePrecision[x] != -1:
                    top30ComputeDisjunctiveSamplePrecision.append(top30DisjunctiveSamplePrecision[x])
            for x in range(len(top30DisjunctiveSampleRecall)):
                if top30DisjunctiveSampleRecall[x] != -1:
                    top30ComputeDisjunctiveSampleRecall.append(top30DisjunctiveSampleRecall[x])
            for x in range(len(top30DisjunctiveSampleSpecificity)):
                if top30DisjunctiveSampleSpecificity[x] != -1:
                    top30ComputeDisjunctiveSampleSpecificity.append(top30DisjunctiveSampleSpecificity[x])
 
        avgTop10DisjunctiveRecall = np.average(top10ComputeDisjunctiveRecall)
        sdTop10DisjunctiveRecall = np.std(top10ComputeDisjunctiveRecall)
 
        avgTop15DisjunctiveRecall = np.average(top15ComputeDisjunctiveRecall)
        sdTop15DisjunctiveRecall = np.std(top15ComputeDisjunctiveRecall)
 
        avgTop30DisjunctiveRecall = np.average(top30ComputeDisjunctiveRecall)
        sdTop30DisjunctiveRecall = np.std(top30ComputeDisjunctiveRecall)
        #--------------------------------------------------------------#
        avgTop10DisjunctivePrecision = np.average(top10ComputeDisjunctivePrecision)
        sdTop10DisjunctivePrecision = np.std(top10ComputeDisjunctivePrecision)
 
        avgTop15DisjunctivePrecision = np.average(top15ComputeDisjunctivePrecision)
        sdTop15DisjunctivePrecision = np.std(top15ComputeDisjunctivePrecision)
 
        avgTop30DisjunctivePrecision = np.average(top30ComputeDisjunctivePrecision)
        sdTop30DisjunctivePrecision = np.std(top30ComputeDisjunctivePrecision)
        #--------------------------------------------------------------#
        avgTop10DisjunctiveSpecificity = np.average(top10ComputeDisjunctiveSpecificity)
        sdTop10DisjunctiveSpecificity = np.std(top10ComputeDisjunctiveSpecificity)
 
        avgTop15DisjunctiveSpecificity = np.average(top15ComputeDisjunctiveSpecificity)
        sdTop15DisjunctiveSpecificity = np.std(top15ComputeDisjunctiveSpecificity)
 
        avgTop30DisjunctiveSpecificity = np.average(top30ComputeDisjunctiveSpecificity)
        sdTop30DisjunctiveSpecificity = np.std(top30ComputeDisjunctiveSpecificity)
 
        #//////////////////////////////////////////////////////////////#
 
        avgTop10DisjunctiveSampleRecall = np.average(top10ComputeDisjunctiveSampleRecall)
        sdTop10DisjunctiveSampleRecall = np.std(top10ComputeDisjunctiveSampleRecall)
 
        avgTop15DisjunctiveSampleRecall = np.average(top15ComputeDisjunctiveSampleRecall)
        sdTop15DisjunctiveSampleRecall = np.std(top15ComputeDisjunctiveSampleRecall)
 
        avgTop30DisjunctiveSampleRecall = np.average(top30ComputeDisjunctiveSampleRecall)
        sdTop30DisjunctiveSampleRecall = np.std(top30ComputeDisjunctiveSampleRecall)
        #--------------------------------------------------------------#
        avgTop10DisjunctiveSamplePrecision = np.average(top10ComputeDisjunctiveSamplePrecision)
        sdTop10DisjunctiveSamplePrecision = np.std(top10ComputeDisjunctiveSamplePrecision)
 
        avgTop15DisjunctiveSamplePrecision = np.average(top15ComputeDisjunctiveSamplePrecision)
        sdTop15DisjunctiveSamplePrecision = np.std(top15ComputeDisjunctiveSamplePrecision)
 
        avgTop30DisjunctiveSamplePrecision = np.average(top30ComputeDisjunctiveSamplePrecision)
        sdTop30DisjunctiveSamplePrecision = np.std(top30ComputeDisjunctiveSamplePrecision)
        #--------------------------------------------------------------#
        avgTop10DisjunctiveSampleSpecificity = np.average(top10ComputeDisjunctiveSampleSpecificity)
        sdTop10DisjunctiveSampleSpecificity = np.std(top10ComputeDisjunctiveSampleSpecificity)
 
        avgTop15DisjunctiveSampleSpecificity = np.average(top15ComputeDisjunctiveSampleSpecificity)
        sdTop15DisjunctiveSampleSpecificity = np.std(top15ComputeDisjunctiveSampleSpecificity)
 
        avgTop30DisjunctiveSampleSpecificity = np.average(top30ComputeDisjunctiveSampleSpecificity)
        sdTop30DisjunctiveSampleSpecificity = np.std(top30ComputeDisjunctiveSampleSpecificity)
        tempPrint =["Average:            "," "," ",avgBaselineRecall,avgBaselinePrecision,avgBaselineSpecificity," ",avgTop10DisjunctiveRecall,avgTop10DisjunctivePrecision,avgTop10DisjunctiveSpecificity," ",avgTop15DisjunctiveRecall,avgTop15DisjunctivePrecision,avgTop15DisjunctiveSpecificity," ",avgTop30DisjunctiveRecall,avgTop30DisjunctivePrecision,avgTop30DisjunctiveSpecificity," ",avgTop10DisjunctiveSampleRecall,avgTop10DisjunctiveSamplePrecision,avgTop10DisjunctiveSampleSpecificity," ",avgTop15DisjunctiveSampleRecall,avgTop15DisjunctiveSamplePrecision,avgTop15DisjunctiveSampleSpecificity," ",avgTop30DisjunctiveSampleRecall,avgTop30DisjunctiveSamplePrecision,avgTop30DisjunctiveSampleSpecificity]
        print(tempPrint)
        print(len(top10ComputeDisjunctivePrecision))

        print()
        print("Threshold: " + str(threshold) + " ~ All Fold Calculations Done ~ Adding to final data")

        for fold in range(5):  
            currentFoldTest = finalFoldTestSet[fold]
            foldTestString = ""
            for test in currentFoldTest:
                foldTestString = foldTestString + str(test) + " "
            finalPrintData.append([threshold," ",str(fold+1),finalBaselineResults[fold][0],finalBaselineResults[fold][1],finalBaselineResults[fold][2]," ",
                                        
                                        finalTop10DisjunctiveResults[fold][1],finalTop10DisjunctiveResults[fold][2],finalTop10DisjunctiveResults[fold][3]," ",
                                        finalTop15DisjunctiveResults[fold][1],finalTop15DisjunctiveResults[fold][2],finalTop15DisjunctiveResults[fold][3]," ",
                                        finalTop30DisjunctiveResults[fold][1],finalTop30DisjunctiveResults[fold][2],finalTop30DisjunctiveResults[fold][3]," ",
                                        finalTop10DisjunctiveSampleResults[fold][1],finalTop10DisjunctiveSampleResults[fold][2],finalTop10DisjunctiveSampleResults[fold][3]," ",
                                        finalTop15DisjunctiveSampleResults[fold][1],finalTop15DisjunctiveSampleResults[fold][2],finalTop15DisjunctiveSampleResults[fold][3]," ",
                                        finalTop30DisjunctiveSampleResults[fold][1],finalTop30DisjunctiveSampleResults[fold][2],finalTop30DisjunctiveSampleResults[fold][3]
                                        ])
        
        finalPrintData.append(["Average:            "," "," ",avgBaselineRecall,avgBaselinePrecision,avgBaselineSpecificity," ",avgTop10DisjunctiveRecall,avgTop10DisjunctivePrecision,avgTop10DisjunctiveSpecificity," ",avgTop15DisjunctiveRecall,avgTop15DisjunctivePrecision,avgTop15DisjunctiveSpecificity," ",avgTop30DisjunctiveRecall,avgTop30DisjunctivePrecision,avgTop30DisjunctiveSpecificity," ",avgTop10DisjunctiveSampleRecall,avgTop10DisjunctiveSamplePrecision,avgTop10DisjunctiveSampleSpecificity," ",avgTop15DisjunctiveSampleRecall,avgTop15DisjunctiveSamplePrecision,avgTop15DisjunctiveSampleSpecificity," ",avgTop30DisjunctiveSampleRecall,avgTop30DisjunctiveSamplePrecision,avgTop30DisjunctiveSampleSpecificity])
        finalPrintData.append(["Standard Deviation: "," "," ",sdBaselineRecall,sdBaselinePrecision,sdBaselineSpecificity," ",sdTop10DisjunctiveRecall,sdTop10DisjunctivePrecision,sdTop10DisjunctiveSpecificity," ",sdTop15DisjunctiveRecall,sdTop15DisjunctivePrecision,sdTop15DisjunctiveSpecificity," ",sdTop30DisjunctiveRecall,sdTop30DisjunctivePrecision,sdTop30DisjunctiveSpecificity," ",sdTop10DisjunctiveSampleRecall,sdTop10DisjunctiveSamplePrecision,sdTop10DisjunctiveSampleSpecificity," ",sdTop15DisjunctiveSampleRecall,sdTop15DisjunctiveSamplePrecision,sdTop15DisjunctiveSampleSpecificity," ",sdTop30DisjunctiveSampleRecall,sdTop30DisjunctiveSamplePrecision,sdTop30DisjunctiveSampleSpecificity])
        finalPrintData.append(["-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-","-"])
    
    finalComputation()

print(finalPrintData)

def printing():
    with open('resultsGastricCancer'+ str(method) +str(threshold)+'_50.csv', 'w') as csvfile:
        filewriter = cs.writer(csvfile, delimiter=',',
            quotechar='|', quoting=cs.QUOTE_MINIMAL)
        #filewriter.writerows(fieldName)
        filewriter.writerows(finalPrintData)
        print("Data printing done")

    with open('./biomarkers' + str(method) + str(threshold) + '_50.csv', 'w') as csvfile:
                ROCwriter = cs.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=cs.QUOTE_MINIMAL,lineterminator = '\n')
                #filewriter.writerows(fieldName)
                ROCwriter.writerows(totalBiomarkers)
    with open('./causalScoresS2Causal' + str(threshold) + '.csv', 'w') as csvfile:
                ROCwriter = cs.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=cs.QUOTE_MINIMAL,lineterminator = '\n')
                #filewriter.writerows(fieldName)
                printData = [["Biomarker","1"]]
                temp = ["Biomarker"]
                for i in range(100):
                    temp.append(i+1)
                printData = [temp]
                for protein in bioMarkerCausalValues.keys():
                    tempNew = [protein]
                    for value in bioMarkerCausalValues[protein]:
                        tempNew.append(value)
                    printData.append(tempNew)
                ROCwriter.writerows(printData)
printing()
print(tempString)  