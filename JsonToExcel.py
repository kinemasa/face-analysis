import pandas as pd
import json
import glob

# Creating an empty Dataframe with column names only
jsonData = pd.DataFrame(columns=['fileName','respirationRateMelanin', 'heartRateStddev', 'lfhf','lf','respirationRate','hf','sampleCount','heartRate'])

##Input data
INPUT_DIRS = ''
INPUT_FILES = glob.glob(INPUT_DIRS+"*\\vitalsensing*.json")
INPUT_FILES =sorted(INPUT_FILES)


##Getting data from files
for i in range(len(INPUT_FILES)):
    with open(INPUT_FILES[i]) as f:
        data = json.load(f)
        jsonData = jsonData.append({'fileName':f.name,'respirationRateMelanin':data['respirationRateMelanin'],'heartRateStddev':data['heartRateStddev'],'lfhf':data['lfhf'],'lf':data['lf'],'respirationRate':data['respirationRate'],'hf':data['hf'],'sampleCount':data['sampleCount'],'heartRate':data['heartRate']}, ignore_index=True)
        print(f.name)

##exports to an excel file
jsonData.to_excel('vitalsensing.xlsx', index=False)