import csv as cs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve

threshold = '1.4'

data = []
index = 0
with open('./predictProbaScores134BiomarkersNB'+threshold+'_25.csv') as csv_file:
    csv = cs.reader(csv_file, delimiter=',')
    for row in csv:
        if index == 0:
            data.append(row)
        else:
            temp = row[1:]
            temp.insert(0,index)
            data.append(temp)
        index += 1

with open('./predictProbaScores134BiomarkersNB'+threshold+'_50.csv') as csv_file:
    csv = cs.reader(csv_file, delimiter=',')
    tempIndex = 0
    for row in csv:
        if tempIndex != 0:
            temp = row[1:]
            temp.insert(0,index)
            data.append(temp)
            index += 1

        tempIndex += 1
with open('./predictProbaScores134BiomarkersNB'+threshold+'_75.csv') as csv_file:
    csv = cs.reader(csv_file, delimiter=',')
    tempIndex = 0
    for row in csv:
        if tempIndex != 0:
            temp = row[1:]
            temp.insert(0,index)
            data.append(temp)
            index += 1
        tempIndex += 1
with open('./predictProbaScores134BiomarkersNB'+threshold+'_100.csv') as csv_file:
    csv = cs.reader(csv_file, delimiter=',')
    tempIndex = 0
    for row in csv:
        if tempIndex != 0:
            temp = row[1:]
            temp.insert(0,index)
            data.append(temp)
            index += 1
        tempIndex += 1


newData = []
for row in data[1:]:
    temp = []
    for point in row:
        if point == "True":
            temp.append(True)
        elif point == "False":
            temp.append(False)
        else:
            temp.append(float(point))
    newData.append(temp)

with open('./predictProbaScores134BiomarkersNB'+threshold+'.csv', 'w') as csvfile:
    ROCwriter = cs.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=cs.QUOTE_MINIMAL)
    #filewriter.writerows(fieldName)
    ROCwriter.writerow(['Fold', '10_y_test', '10_y_proba', '15_y_test', '15_y_proba', '30_y_test', '30_y_proba', '1_y_test', '1_y_proba', '3_y_test', '3_y_proba', '4_y_test', '4_y_proba'])
    ROCwriter.writerows(newData)

columnNames = ['Fold', 'd10_y_test', 'd10_y_proba', 'd15_y_test', 'd15_y_proba', 'd30_y_test', 'd30_y_proba', 'd10wS_y_test', 'd10wS_y_proba', 'd15wS_y_test', 'd15wS_y_proba', 'd30wS_y_test', 'd30wS_y_proba']
df = pd.DataFrame(newData, columns =columnNames)
print(df)

disj10YProba = df.loc[:,"d10_y_proba"]
disj15YProba = df.loc[:,"d15_y_proba"]
disj30YProba = df.loc[:,"d30_y_proba"]
disj10wSYProba = df.loc[:,"d10wS_y_proba"]
disj15wSYProba = df.loc[:,"d15wS_y_proba"]
disj30wSYProba = df.loc[:,"d30wS_y_proba"]

disj10YTest = df.loc[:,"d10_y_test"]
disj15YTest = df.loc[:,"d15_y_test"]
disj30YTest = df.loc[:,"d30_y_test"]
disj10wSYTest = df.loc[:,"d10wS_y_test"]
disj15wSYTest = df.loc[:,"d15wS_y_test"]
disj30wSYTest = df.loc[:,"d30wS_y_test"]

fpr, tpr, thresholds = roc_curve(disj10YTest.values, disj10YProba.values)
plt.plot(fpr, tpr, label="10")
fpr, tpr, thresholds = roc_curve(disj15YTest.values, disj15YProba.values)
plt.plot(fpr, tpr, label="15")
fpr, tpr, thresholds = roc_curve(disj30YTest.values, disj30YProba.values)
plt.plot(fpr, tpr, label="30")

fpr, tpr, thresholds = roc_curve(disj10wSYTest.values, disj10wSYProba.values)
plt.plot(fpr, tpr, label="1")
fpr, tpr, thresholds = roc_curve(disj15wSYTest.values, disj15wSYProba.values)
plt.plot(fpr, tpr, label="3")
fpr, tpr, thresholds = roc_curve(disj30wSYTest.values, disj30wSYProba.values)
plt.plot(fpr, tpr, label="4")

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(threshold+ 'ROC curve')
plt.legend(loc='best')
plt.savefig(''+threshold+'GBTRocCurve134NB.png')
plt.show()
