#MSD and DRMSD network training script using generated hyperspectral remote sensing based data

# This script carries out the MSD and DRMSD network training on hyperspectral remote sensing based data (and on data compressed by PCA, NMF and LDA methods) with overlapping objects and different random seeds
# The results are the networks and log files generated from the (DR)MSD code with different random seeds
# The code assumes that the training data to be available, otherwise run the script in scripts/RemoteSensing/generation/ folder first
# The number of training:validation:test files are hardcoded as 70:20:10 (but can relatively easily be changed)

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)
#MSD code derived from the original MSD code (https://github.com/dmpelt/msdnet) by Daniël Pelt


import DRMSD
import msdnet
import msdnet.operations
import numpy as np
import os
from pathlib import Path
import random
import re
import sys
import tifffile

#Get script arguments
Dataset = sys.argv[1]
GTSet = sys.argv[2]
if(len(sys.argv) > 3):
    DataReductionArg = sys.argv[3]
else:
    DataReductionArg = '0'
if(len(sys.argv) > 4):
    seed = int(sys.argv[4])
else:
    seed = 124

#Settings
layers = 100
dilations = 10
DataPath = '../../../../../../data/RemoteSensingDatasets/PartiallyOverlappingTrue/'
DataReduction = False
DataReductionList = []
if DataReductionArg[0] == '[':
    DataReduction = True
    DataReductionList = [int(el) for el in re.findall(r'\b\d+\b', DataReductionArg)]
spectralDim = tifffile.imread(DataPath + Dataset + '/00000Data.tiff').shape[0]
targetLabels = 11

#Set random seeds
np.random.seed(seed)
random.seed(seed)

print(Dataset)
print(GTSet)
print(DataReductionList)  
print(spectralDim, targetLabels)

#Set number of CPU
msdnet.operations.setthreads(4)

#Set dilations: repeatedly increasing from 1 to dilations value
dil = msdnet.dilations.IncrementDilations(dilations)

#Create main network object for segmentation, either DRMSD or MSD
if DataReduction:
    n = network.DataReductionMSDNet(DataReductionList, layers, dil, spectralDim, targetLabels, gpu = True)
else:
    n = msdnet.network.SegmentationMSDNet(layers, dil, spectralDim, targetLabels, gpu = True)

#Initialize network parameters
n.initialize()

#Construct the training set
flsin = sorted(os.listdir(DataPath + Dataset + '/'))[0:70]
flstg = sorted(os.listdir(DataPath + GTSet + '/'))[0:70]

flsin = [DataPath + Dataset + '/' + s for s in flsin]
flstg = [DataPath + GTSet + '/' + s for s in flstg]

#Create list of datapoints (i.e. input/target pairs) for the training set
dats = []
for i in range(len(flsin)):
    print(i)
    #Create datapoint with file names
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    #Convert datapoint to one-hot
    d_oh = msdnet.data.OneHotDataPoint(d, range(0, targetLabels))
    #Augment data by rotating and flipping
    d_augm = msdnet.data.RotateAndFlipDataPoint(d_oh)
    #Add augmented datapoint to list
    dats.append(d_augm)

#Normalize input and output of network to zero mean and unit variance using training data images
n.normalizeinout(dats)

#Use image batches of a single image
bprov = msdnet.data.BatchProvider(dats,1,seed=124)

#Construct the validation set (no data augmentation)
flsin = sorted(os.listdir(DataPath + Dataset + '/'))[70:90]
flstg = sorted(os.listdir(DataPath + GTSet + '/'))[70:90]
flsin = [DataPath + Dataset + '/' + s for s in flsin]
flstg = [DataPath + GTSet + '/' + s for s in flstg]

#Create list of datapoints for the validation set
datsv = []
for i in range(len(flsin)):
    print(i)
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
    d_oh = msdnet.data.OneHotDataPoint(d, range(0, targetLabels))
    datsv.append(d_oh)

#Validate with Mean-Squared Error
val = msdnet.validate.MSEValidation(datsv)

#Use ADAM training algorithms
t = msdnet.train.AdamAlgorithm(n)

#Setup name to for network and log files
if(DataReduction):
    SettingsExt = '_' + Dataset + '_layers' + str(layers) + 'dil' + str(dilations) + 'DR' + str(DataReduction) + 'L'
    for v in DataReductionList:
        SettingsExt = SettingsExt + '_' + str(v)
else:
    SettingsExt = '_' + Dataset + '_layers' + str(layers) + 'dil' + str(dilations) + 'DR' + str(DataReduction)

#Log error metrics to console
consolelog = msdnet.loggers.ConsoleLogger()
#Log error metrics to file
filelog = msdnet.loggers.FileLogger('log_segm' + SettingsExt + '.txt')
#Log typical, worst, and best images to image files
imagelog = msdnet.loggers.ImageLabelLogger('log_segm' + SettingsExt , onlyifbetter=True, imsize = 500)

#Log images for each channel
singlechannellogs = []
outfolder = Path('Singlechannellogs' + SettingsExt)
outfolder.mkdir(exist_ok=True)
for i in range(0,targetLabels):
    singlechannellogs.append(msdnet.loggers.ImageLogger(str(outfolder) + '/log_segm' + SettingsExt + '_singlechannel_' + str(i), chan_out=i, onlyifbetter=True, imsize = 500))

#Train network until program is stopped manually or given time runs out
print("Training starting...")
msdnet.train.train(n, t, val, bprov, 'segm_params' + SettingsExt + '.h5', loggers=[consolelog,filelog,imagelog] + singlechannellogs, val_every=len(datsv), progress = True)
