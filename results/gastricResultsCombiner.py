import csv as cs
import numpy as np

threshold = "1.8"
newthreshold = str(threshold)[0] + "_" + str(threshold)[2]
method = "C:\\Users\\shash\\VS Code Programs\\Python\\ResearchWork\\Gastric Cancer Data\\gastricSBATCHFiles\\results\\MLP UV RESULTS\\resultsGastricCancerFINAL_RF_UV_All_Biomarkers_"+newthreshold+threshold
fileName = "C:\\Users\\shash\\VS Code Programs\\Python\\ResearchWork\\Gastric Cancer Data\\gastricSBATCHFiles\\results\\MLP UV RESULTS\\resultsGastricFINAL_RF_UV_"+threshold+".csv"
data = []
def inputData():
    with open (method+"_25.csv") as csv_file:
        csv = cs.reader(csv_file, delimiter=',')
        index = 0
        for row in csv:
            if index == 7:
                data.append(row)
            index += 1
    with open (method+"_50.csv") as csv_file:
        csv = cs.reader(csv_file, delimiter=',')
        index = 0
        for row in csv:
            if index == 7:
                data.append(row)
            index += 1
    with open (method+"_75.csv") as csv_file:
        csv = cs.reader(csv_file, delimiter=',')
        index = 0
        for row in csv:
            if index == 7:
                data.append(row)
            index += 1
    with open (method+"_100.csv") as csv_file:
        csv = cs.reader(csv_file, delimiter=',')
        index = 0
        for row in csv:
            if index == 7:
                data.append(row)
            index += 1
inputData()

BaselineSens = []
BaselineSpec = []

Sens1 = []
Spec1 = []
Sens3 = []
Spec3 = []
Sens4 = []
Spec4 = []
Sens10 = []
Spec10 = []
Sens15 = []
Spec15 = []
Sens30 = []
Spec30 = []

index = 0
for row in data:
    if index < 2:  
        BaselineSens.append(float(row[3]))
        Sens10.append(float(row[7]))
        Sens15.append(float(row[11]))
        Sens30.append(float(row[15]))
        Sens1.append(float(row[19]))
        Sens3.append(float(row[23]))
        Sens4.append(float(row[27]))
    else:
        BaselineSpec.append(float(row[5]))
        Spec10.append(float(row[9]))
        Spec15.append(float(row[13]))
        Spec30.append(float(row[17]))
        Spec1.append(float(row[21]))
        Spec3.append(float(row[25]))
        Spec4.append(float(row[29]))
    index += 1


printData = [
    ["Method","Sensitivity","Specificity"],
    ["Baseline",np.average(BaselineSens),np.average(BaselineSpec)],
    ["1 Biomarker",np.average(Sens1),np.average(Spec1)],
    ["3 Biomarker",np.average(Sens3),np.average(Spec3)],
    ["4 Biomarker",np.average(Sens4),np.average(Spec4)],
    ["10 Biomarker",np.average(Sens10),np.average(Spec10)],
    ["15 Biomarker",np.average(Sens15),np.average(Spec15)],
    ["30 Biomarker",np.average(Sens30),np.average(Spec30)]
]
with open(fileName, 'w') as csvfile:
                ROCwriter = cs.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=cs.QUOTE_MINIMAL,lineterminator = '\n')
                #filewriter.writerows(fieldName)
                ROCwriter.writerows(printData)
print()