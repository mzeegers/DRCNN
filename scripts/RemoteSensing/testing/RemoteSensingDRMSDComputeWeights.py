#Testing scripts for trained MSD networks on remote sensing data to compute network weights

# This script carries out the testing of MSD networks on the remote sensing based testing data
# The results are the csv files in the results/RemoteSensing/quantitative/ folder and the reduced image in the results/RemoteSensing/plots/ folder

# The code assumes that the MSD networks are trained and available in the scripts/RemoteSensing/training/MSD/ManyMaterialsTrue folder
# It also assumes the (testing) datasets to be available in the /data/RemoteSensingDatasets/ManyMaterialsTrue/ folder

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import msdnet
from pathlib import Path
import DRMSD
import glob
import tifffile
import matplotlib.pyplot as plt
import os
import numpy as np

#Configurations (setting, reduction algorithm, bins, sampling, data type)
conf = ['PartiallyOverlappingTrue', 'DRMSD', 1, 0, 'DataSourceNoise']
print("Config", conf)
Data = conf[0]
DR = conf[1]
bins = conf[2]
sample = conf[3]
Noise = conf[4]

#Create data path
RootDataPath = '../../../data/RemoteSensingDatasets/' + Data + '/'
if (DR == 'PCA' or DR == 'NMF' or DR == 'LDA'):
    subname = 'FULLTrainingSet_Sample' + str(sample) + '_' + DR + 'Data_' + str(bins) + 'comp' + Noise
    if DR == 'LDA':
        subname = subname + 'GT10Labels/'
else:
    subname = Noise
FullDataPath = RootDataPath + subname + '/'
FullGTPath = RootDataPath + 'GT10Labels/'

#Create network path
NetworkPath = '../../../scripts/RemoteSensing/training/MSD/' + Data + '/'
if(DR == 'DRMSD'):
    if(bins == 21):
        FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRTrueL_2_1.h5'
    else:
        FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRTrueL_' + str(bins) + '.h5'
else:
    FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRFalse.h5'

#Load the network
if(DR == 'DRMSD'):
    n = network.DataReductionMSDNet.from_file(FullNetworkPath)
else:
    n = msdnet.network.SegmentationMSDNet.from_file(FullNetworkPath)

#Load the dataname
flsin = sorted(os.listdir(FullDataPath))[90:91] #[1:2] for quick testing
flsin = [FullDataPath + s for s in flsin]

#Load files and compute network result        
d = msdnet.data.ImageFileDataPoint(flsin[0])
output = n.forward(d.input)

#Get the weights and plot these
x = np.arange(1,201)
y = n.redw[0]

plt.figure()
plt.xlim(left = 0, right = 200)
plt.plot(x,y)

plt.xlabel('Spectral bin')
plt.ylabel('Output weight')
plt.title('Output weight values per spectral bin for DRMSD')
plt.savefig("../../../results/RemoteSensing/plots/RemoteSensingDRMSDWeights.png", dpi=500)
plt.savefig("../../../results/RemoteSensing/plots/RemoteSensingDRMSDWeights.eps", format='eps', dpi=500)
plt.show()

#Get the reduced image and store it
tifffile.imsave('../../../results/RemoteSensing/plots/OutputRemoteSensingDRMSDLayer.tiff', n.redim[-1].astype(np.float32))
