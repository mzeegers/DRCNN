#Testing scripts for trained UNet networks on X-ray data to compute network weights

# This script carries out the testing of UNet networks on the X-ray based testing data
# The results are the csv files in the results/X-ray/quantitative/ folder and the reduced image in the results/X-ray/plots/ folder

# The code assumes that the UNet networks are trained and available in the scripts/X-ray/training/UNet/ManyMaterialsTrue/ folder
# It also assumes the (testing) datasets to be available in the /data/X-rayDatasets/ManyMaterialsTrue/ folder

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.utils
import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )   

class DRUNet(nn.Module):
    def __init__(self, n_class, DataReduction, DataReductionList, spectralDim):
        super().__init__()

        self.DataReduction = DataReduction
        self.DataReductionList = DataReductionList

        if self.DataReduction == False:
            self.dconv_down1 = double_conv(spectralDim, 128)
        else:
            DROp = []
            OutDim = spectralDim
            for v in self.DataReductionList:
                InDim = OutDim
                OutDim = v
                DROp.append(nn.Linear(InDim, OutDim))
                DROp.append(nn.LeakyReLU(0.01, inplace=True))
            
            self.DRSeq = nn.Sequential(*DROp)
            for idx, v in enumerate(self.DataReductionList):
                if(idx%2 == 0):
                    self.DRSeq[idx].weight.data.fill_(0)
                    self.DRSeq[idx].bias.data.fill_(0)
            self.dconv_down1 = double_conv(self.DataReductionList[-1], 128)
       
        self.dconv_down2 = double_conv(128, 256)
        self.dconv_down3 = double_conv(256, 512)  

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up2 = double_conv(256 + 512, 256)
        self.dconv_up1 = double_conv(256 + 128, 128)
        
        self.conv_last = nn.Conv2d(128, n_class, 1)

    def forward(self, input):

        if self.DataReduction == False:
            x = input
        else:
            for idx, v in enumerate(self.DataReductionList):
                input.transpose_(1, 3)
                input = self.DRSeq[idx*2](input).clone()
                input.transpose_(1, 3)
                input = self.DRSeq[idx*2+1](input).clone()
            x = input

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        x = self.dconv_down3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

    def dataReduction(self, input):
        for idx, v in enumerate(DataReductionList):
            input.transpose_(1, 3)
            input = self.DRSeq[idx*2](input).clone()
            input.transpose_(1, 3)
            input = self.DRSeq[idx*2+1](input).clone()
        return input

#Configurations (setting, reduction algorithm, bins, sampling, data type)
conf = ['ManyMaterialsTrue', 'DRUNet', 1, 0, 'DataNoise']
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
NetworkPath = '../../../scripts/X-ray/training/UNet/' + Data + '/'
if(DR == 'DRUNet'):
    FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRTrueL_' + str(bins) + '.pth'
    DataReductionList = [bins]
    DataReduction = True
else:
    FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRFalse.pth'
    DataReductionList = [0]
    DataReduction = False

#Load the dataname
flsin = sorted(os.listdir(FullDataPath))[90:91] #[0:1] for quick testing
flsin = [FullDataPath + s for s in flsin]

#Load the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

spectralDim = tifffile.imread(FullDataPath + '/00000Data.tiff').shape[0]
num_class = len(np.unique(tifffile.imread(FullGTPath + '/00000GT.tiff')))
model = DRUNet(num_class, DataReduction, DataReductionList, spectralDim)
model.load_state_dict(torch.load(FullNetworkPath))
model.eval()
model.to(device)

#Get the weights and plot these
x = range(0,300)
y = model.DRSeq[0].weight.cpu().detach().numpy()[0,:]

plt.figure()
plt.xlim(left = 0, right = 300)
plt.ylim(bottom = -0.7, top = 0.7)
plt.axvline(x=63.076, color='orange', lw=2)
plt.axvline(x=69.334, color='lightgreen', lw=2)
plt.plot(x,y)

plt.xlabel('Spectral bin')
plt.ylabel('Output weight')
plt.title('Output weight values per spectral bin for DRUNet')
plt.savefig("../../../results/X-ray/plots/RemoteSensingDRUNetWeights.png", dpi=500)
plt.savefig("../../../results/X-ray/plots/RemoteSensingDRUNetWeights.eps", format='eps', dpi=500)
plt.show()

#Get the reduced image and store it
Data = tifffile.imread(flsin[0])
Data = Data[np.newaxis,:,:,:]
Data = torch.from_numpy(Data)
Data = Data.to(device)
DRed = model.dataReduction(Data)
DRed = DRed.data.cpu().numpy()
tifffile.imsave('../../../results/X-ray/plots/OutputX-rayDRUNetLayer.tiff', DRed[0,:,:,:].astype(np.float32))
