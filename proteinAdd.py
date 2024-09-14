import csv as cs

data = []
index = 0
organisms = {}
with open ('./gastricDataSetIgA.csv') as csv_file:
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
        organism = str(row[2])
        if proteinName == 'null':
            if organism not in organisms.keys():
                organisms[organism] = 1
            else:
                organisms[organism] = organisms[organism] + 1
            count = organisms[organism]
            proteinName = organism + str(count)
        newRow[0] = proteinName
        data.append(newRow)

with open('./gastricDataSetIgAFormatted.csv', 'w') as csvfile:
        filewriter = cs.writer(csvfile, delimiter=',',
            quotechar='|', quoting=cs.QUOTE_MINIMAL, lineterminator = '\n')
        #filewriter.writerows(fieldName)
        filewriter.writerows(data)
        print("Data printing done")