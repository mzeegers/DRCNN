#Testing scripts for trained MSD networks on X-ray data to compute network weights

# This script carries out the testing of MSD networks on the X-ray based testing data
# The results are the csv files in the results/X-ray/quantitative/ folder and the reduced image in the results/X-ray/plots/ folder

# The code assumes that the MSD networks are trained and available in the scripts/X-ray/training/MSD/ManyMaterialsTrue folder
# It also assumes the (testing) datasets to be available in the /data/X-rayDatasets/ManyMaterialsTrue/ folder

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
conf = ['ManyMaterialsTrue', 'DRMSD', 1, 0, 'DataNoise']
print("Config", conf)
Data = conf[0]
DR = conf[1]
bins = conf[2]
sample = conf[3]
Noise = conf[4]

#Create data path
RootDataPath = '../../../data/X-rayDatasets/' + Data + '/'
if (DR == 'PCA' or DR == 'NMF' or DR == 'LDA'):
    subname = 'FULLTrainingSet_Sample' + str(sample) + '_' + DR + 'Data_' + str(bins) + 'comp' + Noise
    if DR == 'LDA':
        subname = subname + 'GT'
else:
    subname = Noise
FullDataPath = RootDataPath + subname + '/'
FullGTPath = RootDataPath + 'GT/'

#Create network path
NetworkPath = '../../../scripts/X-ray/training/MSD/' + Data + '/'
if(DR == 'DRMSD'):
    FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRTrueL_' + str(bins) + '.h5'
else:
    FullNetworkPath = NetworkPath + 'segm_params_' + subname + '_layers100dil10DRFalse.h5'

#Load the network
if(DR == 'DRMSD'):
    n = network.DataReductionMSDNet.from_file(FullNetworkPath)
else:
    n = msdnet.network.SegmentationMSDNet.from_file(FullNetworkPath)

#Load the dataname
flsin = sorted(os.listdir(FullDataPath))[90:91] #[0:1] for quick testing
flsin = [FullDataPath + s for s in flsin]

#Load files and compute network result    
d = msdnet.data.ImageFileDataPoint(flsin[0])
output = n.forward(d.input)

#Get the weights and plot these
x = np.arange(1,301)
y = n.redw[0]

plt.figure()
plt.xlim(left = 0, right = 300)
plt.ylim(bottom = -0.7, top = 0.7)
plt.axvline(x=63.076, color='orange', lw=2)
plt.axvline(x=69.334, color='lightgreen', lw=2)
plt.plot(x,y)

plt.xlabel('Spectral bin')
plt.ylabel('Output weight')
plt.title('Output weight values per spectral bin for DRMSD')
plt.savefig("../../../results/X-ray/plots/X-rayDRMSDWeights.png", dpi=500)
plt.savefig("../../../results/X-ray/plots/X-rayDRMSDWeights.eps", format='eps', dpi=500)
plt.show()

#Get the reduced image and store it
tifffile.imsave('../../../results/X-ray/plots/OutputX-rayDRMSDLayer.tiff', n.redim[-1].astype(np.float32))
