import os
import csv
import pandas as pd

def getNoiseByID(cls):
    metafile = '/media/trung/Data/UrbanSound8K/metadata/UrbanSound8K.csv'
    metadata = pd.read_csv(metafile)
    metadata = metadata[metadata['classID']==cls]#.sort_values(by=['classID'])
    metadata = metadata[['slice_file_name','fold','classID']]
    metadata['path'] = '/media/trung/Data/UrbanSound8K/audio/fold' + metadata['fold'].astype(str) + '/' + metadata['slice_file_name']
    #print(metadata)
    #print(os.path.exists(metadata['path'].to_numpy()[0]))
    return metadata['path'].to_numpy()

#a = getNoiseByID(2)
#print(a)