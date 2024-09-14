import pandas as pd
import csv as cs
import numpy as np

def select(cauu,k):
  cauu = cauu.drop(labels=0, axis=0)
  csort = cauu.sort_values(by=['1'], ascending=False)
  csort = csort.iloc[:k,:]
  sel= csort['0'].tolist()
  return sel

selectedBiomarkers = []
for x in range(100):
    cauu= pd.read_csv("C:\\Users\\shash\\Downloads\\causalBiomarkers_1.4\\Copy of c_1.4_["+str(x)+"].csv")
    selectedBiomarkers.append(select(cauu,10))

sortedBiomarkers = {}
for x in range(len(selectedBiomarkers)):
   data = selectedBiomarkers[x]
   for biomarker in data:
      if biomarker not in sortedBiomarkers.keys():
         sortedBiomarkers[biomarker] = 1
      else:
         if biomarker in sortedBiomarkers.keys():
            sortedBiomarkers[biomarker]+=1

keys = list(sortedBiomarkers.keys())
values = list(sortedBiomarkers.values())
sorted_value_index = np.argsort(values)
sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
 
print(list(sorted_dict.keys())[-10:])
         
         
      

