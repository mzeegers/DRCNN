#Testing scripts for trained MSD networks on X-ray data to compute segmentation accuracies

# This script carries out the testing of MSD networks on the X-ray based testing data
# The results are the csv files in the results/RemoteSensing/quantitative/ folder

# The code assumes that the MSD networks are trained and available in the scripts/X-ray/training/MSD/ManyMaterials(True/False)/ folders
# It also assumes the (testing) datasets to be available in the /data/X-rayDatasets/ManyMaterials(True/False)/ folders

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import msdnet
from pathlib import Path
import DRMSD
import glob
import tifffile
import os
import numpy as np

def computeNetworkResults(Conf):
    print("Config", Conf)
    Data = Conf[0]
    DR = Conf[1]
    bins = Conf[2]
    sample = Conf[3]
    Noise = Conf[4]

    #Create data paths
    RootDataPath = '../../../data/X-rayDatasets/' + Data + '/'
    if (DR == 'PCA' or DR == 'NMF' or DR == 'LDA'):
        subname = 'FULLTrainingSet_Sample' + str(sample) + '_' + DR + 'Data_' + str(bins) + 'comp' + Noise
        if DR == 'LDA':
            subname = subname + 'GT'
    else:
        subname = Noise
    FullDataPath = RootDataPath + subname + '/'
    FullGTPath = RootDataPath + 'GT/'

    #Create network paths
    NetworkPath = '../../../scripts/X-ray/training/MSD/' + Data + '/'
    if(DR == 'DRMSD'):
        FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRTrueL_' + str(bins) + '.h5'
    else:
        FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRFalse.h5'

    #Load the datanames
    flsin = sorted(os.listdir(FullDataPath))[90:100] #[1:2] for quick testing
    flstg = sorted(os.listdir(FullGTPath))[90:100] #[1:2] for quick testing
    flsin = [FullDataPath + s for s in flsin]
    flstg = [FullGTPath + s for s in flstg]

    #Load the network
    if(DR == 'DRMSD'):
        n = network.DataReductionMSDNet.from_file(FullNetworkPath)
    else:
        n = msdnet.network.SegmentationMSDNet.from_file(FullNetworkPath)

    #Initialize
    TotalPixErr = 0
    TotalPixErrNorm = 0
    TotalTPrate = 0 
    
    for i in range(len(flsin)):
        
        #Load files and compute network result
        d = msdnet.data.ImageFileDataPoint(flsin[i])
        output = n.forward(d.input)
        segment = np.argmax(output,0)
        target = tifffile.imread(flstg[i])

        pixerr = np.count_nonzero(segment != target)
        pixerrnorm = np.count_nonzero(segment != target)/target.size

        #Compute average class accuracy
        TPrate = 0
        for i in range(0,len(np.unique(target))):
            TPrate += np.count_nonzero(np.logical_and((segment == i),(target == i)))/np.count_nonzero(target == i)
        TPrate = TPrate/len(np.unique(target))

        TotalPixErr += pixerr
        TotalPixErrNorm += pixerrnorm
        TotalTPrate += TPrate
    TotalPixErr = TotalPixErr/len(flsin)
    TotalPixErrNorm = TotalPixErrNorm/len(flsin)
    TotalTPrate = TotalTPrate/len(flsin)

    print('Overall accuracy', (1-TotalPixErrNorm)*100)
    print('Average class accuracy', TotalTPrate*100)

    return [Data, DR, bins, sample, (1-TotalPixErrNorm)*100, TotalTPrate*100]


#Get all results
os.makedirs('../../../results/X-ray/quantitative/', exist_ok=True)

#Get results for number of datareduction channels comparison
#List of configurations (setting, reduction algorithm, bins, sampling, data type)
Configs = [['ManyMaterialsTrue', 'MSD',   300, 0, 'DataNoise'],
           ['ManyMaterialsTrue', 'DRMSD', 1,   0, 'DataNoise'],
           ['ManyMaterialsTrue', 'DRMSD', 2,   0, 'DataNoise'],
           ['ManyMaterialsTrue', 'DRMSD', 10,  0, 'DataNoise'],
           ['ManyMaterialsTrue', 'DRMSD', 60,  0, 'DataNoise'],
           ['ManyMaterialsTrue', 'DRMSD', 200, 0, 'DataNoise'],
           ['ManyMaterialsTrue', 'PCA',   1,   2, 'DataNoise'],
           ['ManyMaterialsTrue', 'PCA',   2,   2, 'DataNoise'],
           ['ManyMaterialsTrue', 'PCA',   10,  2, 'DataNoise'],
           ['ManyMaterialsTrue', 'PCA',   60,  2, 'DataNoise'],
           ['ManyMaterialsTrue', 'PCA',   200, 3, 'DataNoise'],
           ['ManyMaterialsTrue', 'NMF',   1,   5, 'DataNoise'],
           ['ManyMaterialsTrue', 'NMF',   2,   5, 'DataNoise'],
           ['ManyMaterialsTrue', 'NMF',   10,  5, 'DataNoise'],
           ['ManyMaterialsTrue', 'NMF',   60,  5, 'DataNoise'],
           ['ManyMaterialsTrue', 'NMF',   200, 6, 'DataNoise'],
           ['ManyMaterialsTrue', 'LDA',   1,   6, 'DataNoise'],
           ['ManyMaterialsTrue', 'LDA',   2,   6, 'DataNoise'],
           ['ManyMaterialsTrue', 'LDA',   10,  6, 'DataNoise'],
           ['ManyMaterialsTrue', 'LDA',   60,  6, 'DataNoise']]   

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../../results/X-ray/quantitative/X-rayDRMSDReductionMethodsResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data

    
#Get results for datareduction method comparison
#List of configurations (setting, reduction algorithm, bins, sampling, data type)
Configs = [['ManyMaterialsFalse', 'PCA',   2, 2, 'Data'],
           ['ManyMaterialsFalse', 'NMF',   2, 5, 'Data'],
           ['ManyMaterialsFalse', 'LDA',   1, 6, 'Data'],
           ['ManyMaterialsFalse', 'DRMSD', 2, 0, 'Data'],
           ['ManyMaterialsFalse', 'MSD',   0, 0, 'Data'],
           ['ManyMaterialsTrue',  'PCA',   2, 2, 'Data'],
           ['ManyMaterialsTrue',  'NMF',   2, 5, 'Data'],
           ['ManyMaterialsTrue',  'LDA',   1, 6, 'Data'],
           ['ManyMaterialsTrue',  'DRMSD', 2, 0, 'Data'],
           ['ManyMaterialsTrue',  'MSD',   0, 0, 'Data'],
           ['ManyMaterialsFalse', 'PCA',   2, 2, 'DataNoise'],
           ['ManyMaterialsFalse', 'NMF',   2, 5, 'DataNoise'],
           ['ManyMaterialsFalse', 'LDA',   1, 6, 'DataNoise'],
           ['ManyMaterialsFalse', 'DRMSD', 2, 0, 'DataNoise'],
           ['ManyMaterialsFalse', 'MSD',   0, 0, 'DataNoise'],
           ['ManyMaterialsTrue',  'PCA',   2, 2, 'DataNoise'],
           ['ManyMaterialsTrue',  'NMF',   2, 5, 'DataNoise'],
           ['ManyMaterialsTrue',  'LDA',   1, 6, 'DataNoise'],
           ['ManyMaterialsTrue',  'DRMSD', 2, 0, 'DataNoise'],
           ['ManyMaterialsTrue',  'MSD',   0, 0, 'DataNoise']]
           
#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../../results/X-ray/quantitative/X-rayDRMSDReductionChannelsResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data
