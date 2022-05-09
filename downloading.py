### 1. DOWNLOADING THE DATASET

import os
import json
import pandas as pd
import numpy as np
import pandas as pd

def handling_json(json_file):
    # Get the json entries from the downloaded json file
    jsonFile = open(json_file, 'r')
    values = json.load(jsonFile)
    jsonFile.close()

    # Create a pandas dataframe of records & convert to .csv file
    path, ext = os.path.splitext(json_file)
    record_df = pd.DataFrame(values['recordings'])
    
    # Remove recordings with other bird species
    other_birds = []
    for i in range(len(record_df.index)):
        if len(list(filter(None, record_df['also'][i]))) != 0:
            other_birds.append(i)
    record_df = record_df.drop(other_birds)    
    record_df.to_csv(path + '.csv', index=False)   

    # Make wget input file
    urls = []
    for file in record_df['file'].tolist():
        urls.append(file)
    with open(path + '.txt', 'w+') as f:
        for item in urls:
            f.write("{}\n".format(item))
            
    # Remove JSON temporary file
    os.remove(json_file)

if __name__ == "__main__":
    
    print('Creating directories...')
    directory = "data"
    records = "recordings"
    # Path
    path = os.path.join(os.getcwd(), directory)
    os.mkdir(path)
    path = os.path.join(path, records)
    os.mkdir(path)
   
    print('Downloading the dataset...')
    area = "europe"
    numPages = 2

    for i in range(1, numPages+1):
        cmd = f'wget -O "./data/{area}_temp{i}.json" "https://xeno-canto.org/api/2/recordings?query=chloris+area:{area}+q:a&page={i}"'    
        os.system(cmd)
        # Handling JSON files to create CSV and TXT files
        json_file = f"./data/{area}_temp{i}.json"
        handling_json(json_file)
        cmd = f'wget -P "./data/recordings" --trust-server-names -i "./data/{area}_temp{i}.txt"'
        os.system(cmd)    
            
    print('Checking for missing files...')
    # Concatenate csv files
    df1 = pd.read_csv(f'./data/{area}_temp1.csv')
    df2 = pd.read_csv(f'./data/{area}_temp2.csv')
    df = pd.concat([df1,df2])

    # Checking for missing files
    print(f'Number of european recordings before checking: {df.shape[0]}')
    notfoundid = []
    for idx, row in df.iterrows():
        if row['file-name'] not in os.listdir('./data/recordings'):
            notfoundid.append(idx)
    df = df.drop(notfoundid) # removing missing audio file
    print(f'Number of european recordings after checking: {df.shape[0]}\n')
    
    # Remove temp files
    for i in range(1, numPages+1): 
        os.remove(f"./data/{area}_temp{i}.csv")
        os.remove(f"./data/{area}_temp{i}.txt")

    print("Saving dataset...")
    # Keep only the columns of interest
    df = df[['id','gen','en','cnt','type','file-name','length']]
    # Store dataset csv file
    df.to_csv('dataset.csv', index=False)
    print('Done!')