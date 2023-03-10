#Plotting script for results from UNet networks on the remote sensing data

# This script plots the results of UNet networks on the remote sensing based testing data
# The plots show how the number data reduction channels influences the UNet segmentation results on various simulated remote sensing datasets for all data reduction methods
# The results are the plots files in the results/RemoteSensing/plots/ folder

# The code assumes that the csv files with UNet network results on testing data are available in the results/RemoteSensing/quantitative/ folder

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os

os.makedirs('../../../results/RemoteSensing/plots/', exist_ok=True)

def PlotResults(inputFile, outputName, plotTitle):

    #Read csv file containing the data
    with open("../../../results/RemoteSensing/quantitative/" + inputFile) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        next(reader, None)  #Skip the headers
        data_read = [row for row in reader]

    #Prepare the data for plotting
    binvalues = list(set([int(x[2]) for x in data_read]))
    binvalues.sort()
    DRMethods = ['DRUNet', 'UNet', 'LDA', 'NMF', 'PCA']
    DRMethodsLabels = ['DRCNN', 'No red.', 'LDA', 'NMF', 'PCA']
    cnt = 0
    DRValueList = []
    for DR in DRMethods:
        valueList = []
        for x in data_read:
            if x[1] == DR:
                if(int(x[2]) == 21):
                    valueList.append((2, float(x[-1])))
                elif(int(x[2]) == 2):
                    valueList.append((3, float(x[-1])))
                elif(int(x[2]) == 10):
                    valueList.append((4, float(x[-1])))
                elif(int(x[2]) == 200):
                    valueList.append((5, float(x[-1])))
                else:
                    valueList.append((int(x[2]), float(x[-1])))
        DRValueList.append([DR, valueList, DRMethodsLabels[cnt]])
        cnt += 1
    print(DRValueList)
    n = len(DRValueList)

    #Plot the data!
    markers = ["s", "p", "o", "D", "*"]
    cnt = 0
    fig, ax1 = plt.subplots()
    for x in DRValueList:
        color=iter(cm.brg(np.linspace(0,1,n)))
        c = next(color)
        for i in range(0,cnt):
            c = next(color)
        plt.plot(*zip(*x[1]), c=c, label=x[2], marker=markers[cnt], alpha=0.7)
        cnt += 1
    plt.ylabel('Average class accuracy (%)')
    plt.xlabel('Reduction feature map channels')
    plt.ylim(ymin = 0, ymax = 102)
    plt.title(plotTitle)
    ax1.set_xticks([1,2,3,4,5])
    ax1.set_xticklabels(['1', '[2,1]', '2', '10', '200'])
    ax1.tick_params(which='minor', bottom=False)
    plt.legend(loc = 'lower right')
    plt.savefig("../../../results/RemoteSensing/plots/" + outputName + ".png", dpi=500)
    plt.savefig("../../../results/RemoteSensing/plots/" + outputName + ".eps", format='eps', dpi=500)
    plt.show()

PlotResults("RemoteSensingDRUNetReductionChannelsCleanResults.csv", "RemoteSensingDRUNetReductionChannelsCleanResults", 'UNet segmentation results on clean dataset')
PlotResults("RemoteSensingDRUNetReductionChannelsCleanOverlappingResults.csv", "RemoteSensingDRUNetReductionChannelsCleanOverlappingResults", 'UNet segmentation results on clean overlapping dataset')
PlotResults("RemoteSensingDRUNetReductionChannelsNoisyResults.csv", "RemoteSensingDRUNetReductionChannelsNoisyResults", 'UNet segmentation results on noisy dataset')
PlotResults("RemoteSensingDRUNetReductionChannelsNoisyOverlappingResults.csv", "RemoteSensingDRUNetReductionChannelsNoisyOverlappingResults", 'UNet segmentation results on noisy overlapping dataset')
