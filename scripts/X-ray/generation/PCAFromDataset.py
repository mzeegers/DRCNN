#PCA compression from hyperspectral X-ray projections

#This script applies PCA compression to the hyperspectral X-ray projections to the training sets
#The code assumes that the MaterialProjectionsGenerator.py and SpectralDataGenerator.py have been carried out first to produce the hyperspectral X-ray datasets
# and that those (four) datasets are available in the folders data/X-rayDatasets/ManyMaterials(True/False)/(Data/DataNoise)

#The compression is applied to the first 'TrainingSetSize' datafiles in the input folders (which form the training data)

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import numpy as np
from sklearn.decomposition import PCA
import tifffile
import os
import random

#Main PCA compression function
def PCA_Reduction(DatasetPath, Dataset, n_comp, LimLow, LimHigh, Sample):

    #Set random seeds
    np.random.seed(123)
    random.seed(123)
    
    #Set the input path of the data
    DatasetPathData = DatasetPath + Dataset + '/'

    #Make an output folder
    if(LimLow > 0 or LimHigh < 300):
        PCADataFolder = DatasetPath + 'FULLTrainingSet_Sample' + str(Sample) + '_LIMITED_' + str(LimLow) + '_' + str(LimHigh) + 'PCAData_' + str(n_comp) + 'comp' + Dataset + '/'
    else:
        PCADataFolder = DatasetPath + 'FULLTrainingSet_Sample' + str(Sample) + '_PCAData_' + str(n_comp) + 'comp' + Dataset + '/'
    os.makedirs(PCADataFolder, exist_ok=True)
    print(PCADataFolder)

    #Prepare the data for PCA analysis
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

    #Normalization
    print("Normalizing...")
    MEAN = np.mean(DataFlat, axis = 0)
    DataFlat = DataFlat - MEAN

    print(DataFlat.shape, ", sampling every" , str(Sample), "points randomly")

    #Compute the PCA transformation
    print("PCA analysis...")
    pca = PCA(n_components = n_comp)
    randints = np.random.permutation(DataFlat.shape[0])
    randints = randints[::Sample]
    pca.fit(DataFlat[randints,:])
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(pca.components_)
    print("Finished!")

    del DataFlat

    #Apply the PCA compression to all data files and save in the new folder
    print("Transforming and saving the data one by one...")
    print(PCADataFolder)
    for d in DataList:
        print(d)
        E = tifffile.imread(DatasetPathData + d)
        E = E.reshape(ImDim[0], ImDim[1]*ImDim[2])
        E = E[np.max((0,LimLow)):np.min((E.shape[0],LimHigh)),:]
        E = E.swapaxes(0,1)
        E = E - MEAN
        E = pca.transform(E)
        E = E.swapaxes(0,1)
        E = E.reshape(n_comp, ImDim[1], ImDim[2])
        tifffile.imsave(PCADataFolder + d, E.astype(np.float32))
        
#Carry out all the necessary PCA compressions 
DatasetPath = '../../../data/X-rayDatasets/ManyMaterialsFalse/'
Dataset = 'Data'
Sample = 2
LimLow = -1
LimHigh = 1000

PCA_Reduction(DatasetPath, Dataset, 2, LimLow, LimHigh, Sample)

Dataset = 'DataNoise'
PCA_Reduction(DatasetPath, Dataset, 2, LimLow, LimHigh, Sample)

DatasetPath = DatasetPath = '../../../data/X-rayDatasets/ManyMaterialsTrue/'
Dataset = 'Data'
PCA_Reduction(DatasetPath, Dataset, 2, LimLow, LimHigh, Sample)

Dataset = 'DataNoise'
PCA_Reduction(DatasetPath, Dataset, 1, LimLow, LimHigh, Sample)
PCA_Reduction(DatasetPath, Dataset, 2, LimLow, LimHigh, Sample)
PCA_Reduction(DatasetPath, Dataset, 10, LimLow, LimHigh, Sample)
PCA_Reduction(DatasetPath, Dataset, 60, LimLow, LimHigh, Sample)
Sample = 3
PCA_Reduction(DatasetPath, Dataset, 200, LimLow, LimHigh, Sample)
