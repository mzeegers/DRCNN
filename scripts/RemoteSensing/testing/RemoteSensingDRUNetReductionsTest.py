#Testing scripts for trained UNet networks on remote sensing data to compute segmentation accuracies

# This script carries out the testing of UNet networks on the remote sensing based testing data
# The results are the csv files in the results/RemoteSensing/quantitative/ folder

# The code assumes that the UNet networks are trained and available in the scripts/RemoteSensing/training/UNet/ManyMaterialsTrue folder
# It also assumes the (testing) datasets to be available in the /data/RemoteSensingDatasets/ManyMaterialsTrue/ folder

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
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



def computeNetworkResults(Conf):

    #Load configuration
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
    NetworkPath = '../../../scripts/RemoteSensing/training/UNet/' + Data + '/'
    if(DR == 'DRUNet'):
        if(bins == 21):
            FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRTrueL_2_1.pth'
            DataReductionList = [2,1]
        else:
            FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRTrueL_' + str(bins) + '.pth'
            DataReductionList = [bins]
        DataReduction = True
    else:
        FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRFalse.pth'
        DataReductionList = [0]
        DataReduction = False
    
    #Load the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    spectralDim = tifffile.imread(FullDataPath + '/00000Data.tiff').shape[0]
    num_class = len(np.unique(tifffile.imread(FullGTPath + '/00000GT.tiff')))
    model = DRUNet(num_class, DataReduction, DataReductionList, spectralDim)
    model.load_state_dict(torch.load(FullNetworkPath))
    model.eval()
    model.to(device)

    #Load the datanames
    flsin = sorted(os.listdir(FullDataPath))[90:100] #[1:2] for quick testing
    flstg = sorted(os.listdir(FullGTPath))[90:100] #[1:2] for quick testing
    flsin = [FullDataPath + s for s in flsin]
    flstg = [FullGTPath + s for s in flstg]

    #Initialize
    TotalPixErr = 0
    TotalPixErrNorm = 0
    TotalTPrate = 0 

    for i in range(len(flsin)):

        #Load files and compute network result
        DataIm = tifffile.imread(flsin[i])
        DataIm = DataIm[np.newaxis,:,:,:]
        DataIm = torch.from_numpy(DataIm)
        inputs = DataIm.to(device)
        output = model.forward(inputs)
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()[0,:,:,:]

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
Configs = [['PartiallyOverlappingTrue',  'PCA',    2, 2, 'Data'],
           ['PartiallyOverlappingTrue',  'NMF',    2, 5, 'Data'],
           ['PartiallyOverlappingTrue',  'LDA',    2, 6, 'Data'],
           ['PartiallyOverlappingTrue',  'DRUNet', 2, 0, 'Data'],
           ['PartiallyOverlappingTrue',  'UNet',   0, 0, 'Data'],
           ['PartiallyOverlappingFalse', 'PCA',    2, 2, 'Data'],
           ['PartiallyOverlappingFalse', 'NMF',    2, 5, 'Data'],
           ['PartiallyOverlappingFalse', 'LDA',    2, 6, 'Data'],
           ['PartiallyOverlappingFalse', 'DRUNet', 2, 0, 'Data'],
           ['PartiallyOverlappingFalse', 'UNet',   0, 0, 'Data'],
           ['PartiallyOverlappingTrue',  'PCA',    2, 2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'NMF',    2, 5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'LDA',    2, 6, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'DRUNet', 2, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue',  'UNet',   0, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'PCA',    2, 2, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'NMF',    2, 5, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'LDA',    2, 6, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRUNet', 2, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'UNet',   0, 0, 'DataSourceNoise']]

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRUNetReductionMethodsResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data


#Get results for number of datareduction channels comparison - Clean and not overlapping
#Setting, reduction algorithm, bins, sample, data type
Configs = [['PartiallyOverlappingFalse', 'UNet',   200, 0, 'Data'],
           ['PartiallyOverlappingFalse', 'DRUNet', 1,   0, 'Data'],
           ['PartiallyOverlappingFalse', 'DRUNet', 21,  0, 'Data'],
           ['PartiallyOverlappingFalse', 'DRUNet', 2,   0, 'Data'],
           ['PartiallyOverlappingFalse', 'DRUNet', 10,  0, 'Data'],
           ['PartiallyOverlappingFalse', 'PCA',    1,   2, 'Data'],
           ['PartiallyOverlappingFalse', 'PCA',    2,   2, 'Data'],
           ['PartiallyOverlappingFalse', 'PCA',    10,  2, 'Data'],
           ['PartiallyOverlappingFalse', 'NMF',    1,   5, 'Data'],
           ['PartiallyOverlappingFalse', 'NMF',    2,   5, 'Data'],
           ['PartiallyOverlappingFalse', 'NMF',    10,  5, 'Data'],
           ['PartiallyOverlappingFalse', 'LDA',    1,   6, 'Data'],
           ['PartiallyOverlappingFalse', 'LDA',    2,   6, 'Data'],
           ['PartiallyOverlappingFalse', 'LDA',    10,  6, 'Data']]  
       
#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)
       
#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRUNetReductionChannelsCleanResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data 
        
#Get results for number of datareduction channels comparison - Clean and overlapping
#Setting, reduction algorithm, bins, sample, data type
Configs = [['PartiallyOverlappingTrue', 'UNet',   200, 0, 'Data'],
           ['PartiallyOverlappingTrue', 'DRUNet', 1,   0, 'Data'],
           ['PartiallyOverlappingTrue', 'DRUNet', 21,  0, 'Data'],
           ['PartiallyOverlappingTrue', 'DRUNet', 2,   0, 'Data'],
           ['PartiallyOverlappingTrue', 'DRUNet', 10,  0, 'Data'],
           ['PartiallyOverlappingTrue', 'PCA',    1,   2, 'Data'],
           ['PartiallyOverlappingTrue', 'PCA',    2,   2, 'Data'],
           ['PartiallyOverlappingTrue', 'PCA',    10,  2, 'Data'],
           ['PartiallyOverlappingTrue', 'NMF',    1,   5, 'Data'],
           ['PartiallyOverlappingTrue', 'NMF',    2,   5, 'Data'],
           ['PartiallyOverlappingTrue', 'NMF',    10,  5, 'Data'],
           ['PartiallyOverlappingTrue', 'LDA',    1,   6, 'Data'],
           ['PartiallyOverlappingTrue', 'LDA',    2,   6, 'Data'],
           ['PartiallyOverlappingTrue', 'LDA',    10,  6, 'Data']]    

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRUNetReductionChannelsCleanOverlappingResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data
        
#Get results for number of datareduction channels comparison - Noisy and not overlapping
#Setting, reduction algorithm, bins, sample, data type
Configs = [['PartiallyOverlappingFalse', 'UNet',   200, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRUNet', 1,   0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRUNet', 21,  0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRUNet', 2,   0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'DRUNet', 10,  0, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'PCA',    1,   2, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'PCA',    2,   2, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'PCA',    10,  2, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'NMF',    1,   5, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'NMF',    2,   5, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'NMF',    10,  5, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'LDA',    1,   6, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'LDA',    2,   6, 'DataSourceNoise'],
           ['PartiallyOverlappingFalse', 'LDA',    10,  6, 'DataSourceNoise']]  
        
#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)
        
#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRUNetReductionChannelsNoisyResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data     
        
#Get results for number of datareduction channels comparison - Noisy and overlapping
#Setting, reduction algorithm, bins, sample, data type
Configs = [['PartiallyOverlappingTrue', 'UNet',   200, 0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'DRUNet', 1,   0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'DRUNet', 21,  0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'DRUNet', 2,   0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'DRUNet', 10,  0, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'PCA',    1,   2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'PCA',    2,   2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'PCA',    10,  2, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'NMF',    1,   5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'NMF',    2,   5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'NMF',    10,  5, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'LDA',    1,   6, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'LDA',    2,   6, 'DataSourceNoise'],
           ['PartiallyOverlappingTrue', 'LDA',    10,  6, 'DataSourceNoise']]   
        
#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeNetworkResults(conf)
    Results.append(Res)
  
#Write raw results to csv file
with open("../../../results/RemoteSensing/quantitative/RemoteSensingDRUNetReductionChannelsNoisyOverlappingResults.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter = ",")
    writer.writerow(["Data", "Reduction", "Bins", "Sample", "OO", "ACA"])   #Write header
    writer.writerows(Results)                                               #Write data     
