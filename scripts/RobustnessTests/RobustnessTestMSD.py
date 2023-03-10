#Testing scripts for trained MSD networks to compute network robustness to different random seeds

# This script carries out the testing of MSD networks on the X-ray and remote sensing based testing data
# The results are the csv files in the results/Robustness/ folder

# The code assumes that the MSD networks are trained and available in the scripts/X-ray/training/MSD/ManyMaterialsTrue/RobustnessExperiment/
#  and scripts/RemoteSensing/training/MSD/PartiallyOverlappingTrue/RobustnessExperiment/ folders
# It also assumes the (testing) datasets to be available in the /data/X-rayDatasets/ManyMaterialsTrue/ and /data/RemoteSensingDatasets/PartiallyOverlappingTrue/ folders

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
import pyqtgraph as pq

def computeRobustnessResults(Conf):
    
    ExpType = Conf[0]
    Data = Conf[1]
    DR = Conf[2]
    bins = Conf[3]
    sample = Conf[4]
    Noise = Conf[5]
    seeds = Conf[6]
    
    cnt = 0
    Res = np.zeros(8)
    
    for seed in range(0,seeds):

        print("Config", Conf, "seed", seed)

        #Create data paths
        if(ExpType == 'RemoteSensing'):
            RootDataPath = '../../data/RemoteSensingDatasets/' + Data + '/'
            if (DR == 'PCA' or DR == 'NMF' or DR == 'LDA'):
                subname = 'FULLTrainingSet_Sample' + str(sample) + '_' + DR + 'Data_' + str(bins) + 'comp' + Noise
                if(DR == 'LDA'):
                    subname = subname + 'GT10Labels'
            else:
                subname = Noise
            FullDataPath = RootDataPath + subname + '/'
            FullGTPath = RootDataPath + 'GT10Labels/'
        else:
            RootDataPath = '../data/X-rayDatasets/' + Data + '/'
            if (DR == 'PCA' or DR == 'NMF' or DR == 'LDA'):
                subname = 'FULLTrainingSet_Sample' + str(sample) + '_' + DR + 'Data_' + str(bins) + 'comp' + Noise 
                if DR == 'LDA':
                    subname = subname + 'GT'
            else:
                subname = Noise
            FullDataPath = RootDataPath + subname + '/'
            FullGTPath = RootDataPath + 'GT/'
            
        #Create network paths
        if(ExpType == 'RemoteSensing'):
            NetworkPath = '../RemoteSensing/training/MSD/' + Data + '/RobustnessExperiment/'
            if(DR == 'DRMSD'):
                FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRTrueL_' + str(bins) + '_seed' + str(seed) + '.h5'
            else:
                FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRFalse' + '_seed' + str(seed) + '.h5'
        else:
            NetworkPath = 'X-ray/training/MSD/' + Data + '/RobustnessExperiment/'
            if(DR == 'DRMSD'):
                FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRTrueL_' + str(bins) + '_seed' + str(seed) + '.h5'
            else:
                FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRFalse' + '_seed' + str(seed) + '.h5'

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

        #Statistics
        Res[cnt] = TotalTPrate
        cnt += 1
        
    print("Average", np.average(Res)*100)
    print("Std", np.std(Res)*100)
    print("Min", np.min(Res)*100)
    print("Max", np.max(Res)*100)
    print("Median", np.median(Res)*100)
    
    return [ExpType, Data, DR, bins, sample, np.average(Res)*100, np.std(Res)*100, np.min(Res)*100, np.max(Res)*100, np.median(Res)*100]


#Get all results
os.makedirs('../results/Robustness/', exist_ok=True)

#Exptype, setting, reduction algorithm, bins, sample, data type, seeds
Configs = [['RemoteSensing', 'PartiallyOverlappingTrue', 'DRMSD', 1, 0, 'DataSourceNoise', 7],
           ['RemoteSensing', 'PartiallyOverlappingTrue', 'LDA',   1, 6, 'DataSourceNoise', 7],
           ['X-ray',         'ManyMaterialsTrue',        'DRMSD', 2, 0, 'DataNoise', 7],
           ['X-ray',         'ManyMaterialsTrue',        'LDA',   2, 6, 'DataNoise', 7]] 

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeRobustnessResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../results/Robustness/RobustnessResultsMSD.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(["ExpType", " Data", " Reduction", " Bins", " Sample", " ACA avg", " ACA std", " ACA min", " ACA max", " ACA median"])   #Write header
    writer.writerows(Results)                                                                                                                #Write data
