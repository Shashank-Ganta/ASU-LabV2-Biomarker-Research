import csv as cs

data = []
index = 0



with open ('./gastricDataSetIgAFormatted.csv') as csv_file:
    csv = cs.reader(csv_file, delimiter=',')
    for row in csv:
        newRow = row.copy()
        row = []
        for string in newRow:
            
            newString = string.replace(',','')
            row.append(newString)
            if "," in newString:
                print()
        proteinName = str(row[0])
        if index < 10:
            if proteinName != 'null':
                if proteinName == '':
                    a = 1
                else:
                    proteinName += "_IgA"
                temp = row[1:]
                temp.insert(0,proteinName)
                data.append(temp)
        else:
            if proteinName != 'null' and proteinName != '':

                if proteinName == '':
                    a = 1
                else:
                    proteinName += "_IgA"
                temp = row[1:]
                temp.insert(0,proteinName)
                data.append(temp) 
                if proteinName == 'HP0840_IgA':
                    print('hi')
        index += 1
with open ('./gastricDataSetIgGFormatted.csv') as csv_file:
    csv = cs.reader(csv_file, delimiter=',')
    for row in csv:
        newRow = row.copy()
        row = []
        for string in newRow:
            
            newString = string.replace(',','')
            row.append(newString)
            if "," in newString:
                print()
        proteinName = str(row[0])
        if proteinName != '' and proteinName != 'null' and proteinName != 'Protein name':
            proteinName += "_IgG"
            temp = row[1:]
            temp.insert(0,proteinName)
            data.append(temp)

with open('gastricDataSetTotalFormatted.csv', 'w') as csvfile:
        filewriter = cs.writer(csvfile, delimiter=',',
            quotechar='|', quoting=cs.QUOTE_MINIMAL, lineterminator = '\n')
        #filewriter.writerows(fieldName)
        filewriter.writerows(data)
        print("Data printing done")