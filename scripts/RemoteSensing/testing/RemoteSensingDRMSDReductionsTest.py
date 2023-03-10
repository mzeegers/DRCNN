#Testing scripts for trained MSD networks on remote sensing data to compute segmentation accuracies

# This script carries out the testing of MSD networks on the remote sensing based testing data
# The results are the csv files in the results/RemoteSensing/quantitative/ folder

# The code assumes that the MSD networks are trained and available in the scripts/RemoteSensing/training/MSD/ManyMaterialsTrue folder
# It also assumes the (testing) datasets to be available in the /data/RemoteSensingDatasets/ManyMaterialsTrue/ folder

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import msdnet
from pathlib import Path
import network
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
    RootDataPath = '../../../data/RemoteSensingDatasets/' + Data + '/'
    if (DR == 'PCA' or DR == 'NMF' or DR == 'LDA'):
        subname = 'FULLTrainingSet_Sample' + str(sample) + '_' + DR + 'Data_' + str(bins) + 'comp' + Noise
        if(DR == 'LDA'):
            subname = subname + 'GT10Labels'
    else:
        subname = Noise
    FullDataPath = RootDataPath + subname + '/'
    FullGTPath = RootDataPath + 'GT10Labels/'

    #Create network paths
    NetworkPath = '../../../scripts/RemoteSensing/training/MSD/' + Data + '/'
    if(DR == 'DRMSD'):
        if(bins == 21):
            FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRTrueL_2_1.h5'
        else:
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
os.makedirs('../../../results/RemoteSensing/quantitative/', exist_ok=True)

#Get results for datareduction method comparison
#Setting, reduction algorithm, bins, sample, data type
Configs = [['PartiallyOverlappingFalse', 'PCA',   2, 2, 'Data'],
           ['PartiallyOverlappingFalse', 'NMF',   2, 5, 'Data'],
           ['PartiallyOverlappingFalse', 'LDA',   2, 6, 'Data'],
           ['PartiallyOverlappingFalse', 'DRMSD', 2, 0, 'Data'],
           ['PartiallyOverlappingFalse', 'MSD',   0, 0, 'Data'],
           ['PartiallyOverlappingFalse', 'PCA',   2, 2, 'Data'],
           ['PartiallyOverlappingFalse', 'NMF',   2, 5, 'Data'],
           ['PartiallyOverlappingFalse', 'LDA',   2, 6, 'Data'],
           ['PartiallyOverlappingFalse', 'DRMSD', 2, 0, 'Data'],
           ['PartiallyOverlappingFalse', 'MSD',   0, 0, 'Data'],
           ['PartiallyOverlappingTrue',  'PCA',   2, 2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'NMF',   2, 5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'LDA',   2, 6, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'DRMSD', 2, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'MSD',   0, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'PCA',   2, 2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'NMF',   2, 5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'LDA',   2, 6, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'DRMSD', 2, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'MSD',   0, 0, 'DataSourceNoise']]

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRMSDReductionMethodsResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data

#Get results for number of datareduction channels comparison - Clean and not overlapping
#Setting, reduction algorithm, bins, sample, data type
Configs =  [['PartiallyOverlappingFalse', 'MSD',   200, 0, 'Data'],
            ['PartiallyOverlappingFalse', 'DRMSD', 1,   0, 'Data'],
            ['PartiallyOverlappingFalse', 'DRMSD', 21,  0, 'Data'],
            ['PartiallyOverlappingFalse', 'DRMSD', 2,   0, 'Data'],
            ['PartiallyOverlappingFalse', 'DRMSD', 10,  0, 'Data'],
            ['PartiallyOverlappingFalse', 'PCA',   1,   2, 'Data'],
            ['PartiallyOverlappingFalse', 'PCA',   2,   2, 'Data'],
            ['PartiallyOverlappingFalse', 'PCA',   10,  2, 'Data'],
            ['PartiallyOverlappingFalse', 'NMF',   1,   5, 'Data'],
            ['PartiallyOverlappingFalse', 'NMF',   2,   5, 'Data'],
            ['PartiallyOverlappingFalse', 'NMF',   10,  5, 'Data'],
            ['PartiallyOverlappingFalse', 'LDA',   1,   6, 'Data'],
            ['PartiallyOverlappingFalse', 'LDA',   2,   6, 'Data'],
            ['PartiallyOverlappingFalse', 'LDA',   10,  6, 'Data']]   

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)
       
#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRMSDReductionChannelsCleanResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data 

#Get results for number of datareduction channels comparison - Clean and overlapping
#Setting, reduction algorithm, bins, sample, data type
Configs = [['PartiallyOverlappingTrue', 'MSD',   200, 0, 'Data'],
           ['PartiallyOverlappingTrue', 'DRMSD', 1,   0, 'Data'],
           ['PartiallyOverlappingTrue', 'DRMSD', 21,  0, 'Data'],
           ['PartiallyOverlappingTrue', 'DRMSD', 2,   0, 'Data'],
           ['PartiallyOverlappingTrue', 'DRMSD', 10,  0, 'Data'],
           ['PartiallyOverlappingTrue', 'PCA',   1,   2, 'Data'],
           ['PartiallyOverlappingTrue', 'PCA',   2,   2, 'Data'],
           ['PartiallyOverlappingTrue', 'PCA',   10,  2, 'Data'],
           ['PartiallyOverlappingTrue', 'NMF',   1,   5, 'Data'],
           ['PartiallyOverlappingTrue', 'NMF',   2,   5, 'Data'],
           ['PartiallyOverlappingTrue', 'NMF',   10,  5, 'Data'],
           ['PartiallyOverlappingTrue', 'LDA',   1,   6, 'Data'],
           ['PartiallyOverlappingTrue', 'LDA',   2,   6, 'Data'],
           ['PartiallyOverlappingTrue', 'LDA',   10,  6, 'Data']]   

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRMSDReductionChannelsCleanOverlappingResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data

#Get results for number of datareduction channels comparison - Noisy and not overlapping
#Setting, reduction algorithm, bins, sample, data type
Configs = [['PartiallyOverlappingFalse', 'MSD',   200, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRMSD', 1,   0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRMSD', 21,  0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRMSD', 2,   0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRMSD', 10,  0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'PCA',   1,   2, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'PCA',   2,   2, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'PCA',   10,  2, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'NMF',   1,   5, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'NMF',   2,   5, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'NMF',   10,  5, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'LDA',   1,   6, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'LDA',   2,   6, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'LDA',   10,  6, 'DataSourceNoise']]   

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)
        
#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRMSDReductionChannelsNoisyResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data    

#Get results for number of datareduction channels comparison - Noisy and overlapping
#Setting, reduction algorithm, bins, sample, data type
Configs = [['PartiallyOverlappingTrue', 'MSD',   200, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'DRMSD', 1,   0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'DRMSD', 21,  0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'DRMSD', 2,   0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'DRMSD', 10,  0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'PCA',   1,   2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'PCA',   2,   2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'PCA',   10,  2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'NMF',   1,   5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'NMF',   2,   5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'NMF',   10,  5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'LDA',   1,   6, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'LDA',   2,   6, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'LDA',   10,  6, 'DataSourceNoise']] 

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRMSDReductionChannelsNoisyOverlappingResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data
