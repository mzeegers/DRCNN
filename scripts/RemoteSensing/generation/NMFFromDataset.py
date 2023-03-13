#NMF compression for hyperspectral remote sensing images

#This script applies NMF compression to the (training set of the) hyperspectral remote sensing based images
#The code assumes that the EnvironmentsGenerator(PartiallyOverlapping).py, SpectralRemoteSensingDataGenerator.py and 
# ApplyNoise.py have been carried out first to produce the hyperspectral remote sensing datasets with ground truths,
# and that those (four) datasets are available in the folders data/RemoteSensingDatasets/PartiallyOverlapping(True/False)/(Data/DataSourceNoise)/

#The compression is applied to the first 'TrainingSetSize' datafiles in the input folders (which form the training data)

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import numpy as np
from sklearn.decomposition import NMF
import tifffile
import os
import random

#Main NMF compression function
def NMF_Reduction(DatasetPath, Dataset, n_comp, LimLow, LimHigh, Sample):

    #Set random seeds
    np.random.seed(123)
    random.seed(123)
    
    #Set the input path of the data
    DatasetPathData = DatasetPath + Dataset + '/'

    #Make an output folder
    if(LimLow > 0 or LimHigh < 200):
        NMFDataFolder = DatasetPath + 'FULLTrainingSet_Sample' + str(Sample) + '_LIMITED_' + str(LimLow) + '_' + str(LimHigh) + 'NMFData_' + str(n_comp) + 'comp' + Dataset + '/'
    else:
        NMFDataFolder = DatasetPath + 'FULLTrainingSet_Sample' + str(Sample) + '_NMFData_' + str(n_comp) + 'comp' + Dataset + '/'
    os.makedirs(NMFDataFolder, exist_ok=True)
    print(NMFDataFolder)

    #Prepare the data for NMF analysis
    print("Full training set")
    TrainingSetSize = 1#70
    DataList = sorted(os.listdir(DatasetPathData))
    ImDim = tifffile.imread(DatasetPathData + DataList[0]).shape
    
    Data = np.zeros((ImDim[0], TrainingSetSize, ImDim[1], ImDim[2]), dtype=np.float32)

    cnt = 0
    print("Loading...")
    for d in DataList[0:TrainingSetSize]:
        print(d)
        Data[:,cnt,:,:] = tifffile.imread(DatasetPathData + d)
        cnt += 1

    DataShape = Data.shape
    print("Data shape:", DataShape)

    print("Spectral cutting...")
    Data = Data[np.max((0,LimLow)):np.min((Data.shape[0],LimHigh)),:,:,:]

    print("Reshaping the array...")
    DataFlat = Data.reshape((np.min((LimHigh-LimLow,DataShape[0])), DataShape[1]*DataShape[2]*DataShape[3]))
    
    del Data
    print("Swapping the axes...")
    DataFlat = np.swapaxes(DataFlat,0,1)

    #Making the data nonnegative
    print(np.min(DataFlat))
    incr = 0
    if(np.min(DataFlat) < 0):
        incr = -1*np.min(DataFlat)
        DataFlat = DataFlat + incr

    print(DataFlat.shape, ", sampling every" , str(Sample), "points randomly")

    #Compute the NMF transformation
    print("NMF analysis...")
    nmf = NMF(n_components=n_comp, solver="mu")
    randints = np.random.permutation(DataFlat.shape[0])
    randints = randints[::Sample]
    nmf.fit(DataFlat[randints,:])

    print("Finished!")

    del DataFlat

    #Apply the NMF compression to all data files and save in the new folder
    print("Transforming and saving the data one by one...")
    print(NMFDataFolder)
    for d in DataList:
        print(d)
        E = tifffile.imread(DatasetPathData + d)
        E = E.reshape(ImDim[0], ImDim[1]*ImDim[2])
        E = E[np.max((0,LimLow)):np.min((E.shape[0],LimHigh)),:]
        E = E.swapaxes(0,1)
        E = E + incr
        E[E < 0] = 0
        E = nmf.transform(E)
        E = E.swapaxes(0,1)
        E = E.reshape(n_comp, ImDim[1], ImDim[2])
        tifffile.imsave(NMFDataFolder + d, E.astype(np.float32))
        
#Carry out all the necessary NMF compressions 
DatasetPath = '../../../data/RemoteSensingDatasets/PartiallyOverlappingFalse/'
Dataset = 'Data'
Sample = 5
LimLow = -1
LimHigh = 1000

NMF_Reduction(DatasetPath, Dataset, 1, LimLow, LimHigh, Sample)
NMF_Reduction(DatasetPath, Dataset, 2, LimLow, LimHigh, Sample)
NMF_Reduction(DatasetPath, Dataset, 10, LimLow, LimHigh, Sample)

Dataset = 'DataSourceNoise'
NMF_Reduction(DatasetPath, Dataset, 1, LimLow, LimHigh, Sample)
NMF_Reduction(DatasetPath, Dataset, 2, LimLow, LimHigh, Sample)
NMF_Reduction(DatasetPath, Dataset, 10, LimLow, LimHigh, Sample)

DatasetPath = '../../../data/RemoteSensingDatasets/PartiallyOverlappingTrue/'
Dataset = 'Data'

NMF_Reduction(DatasetPath, Dataset, 1, LimLow, LimHigh, Sample)
NMF_Reduction(DatasetPath, Dataset, 2, LimLow, LimHigh, Sample)
NMF_Reduction(DatasetPath, Dataset, 10, LimLow, LimHigh, Sample)

Dataset = 'DataSourceNoise'
NMF_Reduction(DatasetPath, Dataset, 1, LimLow, LimHigh, Sample)
NMF_Reduction(DatasetPath, Dataset, 2, LimLow, LimHigh, Sample)
NMF_Reduction(DatasetPath, Dataset, 10, LimLow, LimHigh, Sample)
