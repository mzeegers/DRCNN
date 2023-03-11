#LDA compression for hyperspectral X-ray projections

#This script applies LDA compression to the (training set of the) hyperspectral X-ray projections
#The code assumes that the MaterialProjectionsGenerator.py and SpectralDataGenerator.py have been carried out first to produce the hyperspectral X-ray datasets with ground truths
# and that those (four) datasets are available in the folders data/X-rayDatasets/ManyMaterials(True/False)/(Data/DataNoise)/
# as well as ground truth for limited material case in the data/X-rayDatasets/ManyMaterialsFalse/GT /
# and for the many material case in the folders data/X-rayDatasets/ManyMaterialsTrue/(GT/GTLDA2/GTLDA10/GTLDA60)/ folders

#The compression is applied to the first 'TrainingSetSize' datafiles in the input folders (which form the training data)

#For more information on LDA and the usage, see: https://scikit-learn.org/stable/modules/lda_qda.html

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tifffile
import os
import random

#Main LDA compression function (Note: n_comp can't be higher than the number of classes minus 1)
def LDA_Reduction(DatasetPath, Dataset, GT, n_comp, LimLow, LimHigh, Sample):

    #Set random seeds
    np.random.seed(123)
    random.seed(123)

    #Set the input paths of the data and ground truth
    DatasetPathData = DatasetPath + Dataset + '/'
    if(n_comp == 1):
        DatasetPathGT = DatasetPath + GT + '/'
    else:
        DatasetPathGT = DatasetPath + 'GTLDA' + str(n_comp) + '/'

    #Make an output folder
    if(LimLow > 0 or LimHigh < 300):
        LDADataFolder = DatasetPath + 'FULLTrainingSet_Sample' + str(Sample) + '_LIMITED_' + str(LimLow) + '_' + str(LimHigh) + 'LDAData_' + str(n_comp) + 'comp' + Dataset + GT + '/'
    else:
        LDADataFolder = DatasetPath + 'FULLTrainingSet_Sample' + str(Sample) + '_LDAData_' + str(n_comp) + 'comp' + Dataset + GT + '/'
    os.makedirs(LDADataFolder, exist_ok=True)
    print(LDADataFolder)

    #Prepare the data for LDA analysis
    print("Full training set")
    TrainingSetSize = 1#70
    DataList = sorted(os.listdir(DatasetPathData))
    GTList = sorted(os.listdir(DatasetPathGT))
    ImDim = tifffile.imread(DatasetPathData + DataList[0]).shape
    
    Data = np.zeros((ImDim[0], TrainingSetSize, ImDim[1], ImDim[2]), dtype=np.float32)
        
    GT = np.zeros((TrainingSetSize, ImDim[1], ImDim[2]), dtype=np.uint8)

    cnt = 0
    print("Loading...")
    for d in DataList[0:TrainingSetSize]:
        print(d)
        Data[:,cnt,:,:] = tifffile.imread(DatasetPathData + d)
        cnt += 1

    cnt = 0
    for d in GTList[0:TrainingSetSize]:
        print(d)
        GT[cnt,:,:] = tifffile.imread(DatasetPathGT + d)
        cnt += 1

    DataShape = Data.shape
    print("Data shape:", DataShape)

    print("Spectral cutting...")
    Data = Data[np.max((0,LimLow)):np.min((Data.shape[0],LimHigh)),:,:,:]

    print("Reshaping the array...")
    DataFlat = Data.reshape((np.min((LimHigh-LimLow,DataShape[0])), DataShape[1]*DataShape[2]*DataShape[3]))
    GTFlat = GT.reshape((GT.shape[0]*GT.shape[1]*GT.shape[2]))    
    
    del Data
    del GT
    print("Swapping the axes...")
    DataFlat = np.swapaxes(DataFlat,0,1)
    
    print(DataFlat.shape, ", sampling every" , str(Sample), "points randomly")

    #Compute the LDA transformation
    print("LDA analysis...")
    randints = np.random.permutation(DataFlat.shape[0])
    randints = randints[::Sample]
    lda = LinearDiscriminantAnalysis()
    lda.fit(DataFlat[randints,:], GTFlat[randints])

    print("Explained variance", lda.explained_variance_ratio_)
 
    print("Finished!")

    del DataFlat
    del GTFlat
    
    #Apply the LDA compression to all data files and save in the new folder
    print("Transforming and saving the data one by one...")
    print(LDADataFolder)
    for d in DataList:
        print(d)
        E = tifffile.imread(DatasetPathData + d)
        E = E.reshape(ImDim[0], ImDim[1]*ImDim[2])
        E = E[np.max((0,LimLow)):np.min((E.shape[0],LimHigh)),:]
        E = E.swapaxes(0,1)
        E = lda.transform(E)
        E = E.swapaxes(0,1)
        E = E.reshape(n_comp, ImDim[1], ImDim[2])
        tifffile.imsave(LDADataFolder + d, E.astype(np.float32))
        
#Carry out all the necessary LDA compressions 
DatasetPath = '../../../data/X-rayDatasets/ManyMaterialsFalse/'
Dataset = 'Data'
GT = 'GT'
LimLow = -1
LimHigh = 1000
Sample = 6

LDA_Reduction(DatasetPath, Dataset, GT, 1, LimLow, LimHigh, Sample)

Dataset = 'DataNoise'
LDA_Reduction(DatasetPath, Dataset, GT, 1, LimLow, LimHigh, Sample)

DatasetPath = '../../../data/X-rayDatasets/ManyMaterialsTrue/'
Dataset = 'Data'
LDA_Reduction(DatasetPath, Dataset, GT, 1, LimLow, LimHigh, Sample)

Dataset = 'DataNoise'
LDA_Reduction(DatasetPath, Dataset, GT, 1, LimLow, LimHigh, Sample)
LDA_Reduction(DatasetPath, Dataset, GT, 2, LimLow, LimHigh, Sample)
LDA_Reduction(DatasetPath, Dataset, GT, 10, LimLow, LimHigh, Sample)
LDA_Reduction(DatasetPath, Dataset, GT, 60, LimLow, LimHigh, Sample)
