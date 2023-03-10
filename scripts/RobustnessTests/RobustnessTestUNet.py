#Testing scripts for trained UNet networks to compute network robustness to different random seeds

# This script carries out the testing of UNet networks on the X-ray and remote sensing based testing data
# The results are the csv files in the results/Robustness/ folder

# The code assumes that the UNet networks are trained and available in the scripts/X-ray/training/UNet/ManyMaterialsTrue/RobustnessExperiment/
#  and scripts/RemoteSensing/training/UNet/PartiallyOverlappingTrue/RobustnessExperiment/ folders
# It also assumes the (testing) datasets to be available in the /data/X-rayDatasets/ManyMaterialsTrue/ and /data/RemoteSensingDatasets/PartiallyOverlappingTrue/ folders

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
            RootDataPath = '../../data/X-rayDatasets/' + Data + '/'
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
            NetworkPath = 'RemoteSensing/training/UNet/' + Data + '/RobustnessExperiment/'
            if(DR == 'DRUNet'):
                FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRTrueL_' + str(bins) + '_seed' + str(seed) + '.pth'
                DataReductionList = [bins]
                DataReduction = True
            else:
                FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRFalse' + '_seed' + str(seed) + '.pth'
                DataReductionList = [0]
                DataReduction = False
        else:
            NetworkPath = 'X-ray/training/UNet/' + Data + '/RobustnessExperiment/'
            if(DR == 'DRUNet'):
                FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRTrueL_' + str(bins) + '_seed' + str(seed) + '.pth'
                DataReductionList = [bins]
                DataReduction = True
            else:
                FullNetworkPath = NetworkPath + 'UNetsegm_params_' + subname + '_DRFalse' + '_seed' + str(seed) + '.pth'
                DataReductionList = [0]
                DataReduction = False

        #Load the datanames
        flsin = sorted(os.listdir(FullDataPath))[90:100] #[1:2] for quick testing
        flstg = sorted(os.listdir(FullGTPath))[90:100] #[1:2] for quick testing
        flsin = [FullDataPath + s for s in flsin]
        flstg = [FullGTPath + s for s in flstg]        
        
        #Load the network
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        
        spectralDim = tifffile.imread(FullDataPath + '/00000Data.tiff').shape[0]
        num_class = len(np.unique(tifffile.imread(FullGTPath + '/00000GT.tiff')))
        model = DRUNet(num_class, DataReduction, DataReductionList, spectralDim)
        model.load_state_dict(torch.load(FullNetworkPath))
        model.eval()
        model.to(device)

        #Initialize
        TotalPixErr = 0
        TotalPixErrNorm = 0
        TotalTPrate = 0 

        for i in range(len(flsin)):

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
Configs = [['RemoteSensing', 'PartiallyOverlappingTrue', 'DRUNet', 1, 0, 'DataSourceNoise', 7],
           ['RemoteSensing', 'PartiallyOverlappingTrue', 'LDA',    1, 6, 'DataSourceNoise', 7],
           ['X-ray',         'ManyMaterialsTrue',        'DRUNet', 2, 0, 'DataNoise', 7],
           ['X-ray',         'ManyMaterialsTrue',        'LDA',    2, 6, 'DataNoise', 7]] 

#Compute the results for all given configurations
Results = []
for conf in Configs:
    Res = computeRobustnessResults(conf)
    Results.append(Res)

#Write raw results to csv file
with open("../../results/Robustness/RobustnessResultsUNet.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(["ExpType", " Data", " Reduction", " Bins", " Sample", " ACA avg", " ACA std", " ACA min", " ACA max", " ACA median"])  #Write header
    writer.writerows(Results)                                                                                                               #Write data
