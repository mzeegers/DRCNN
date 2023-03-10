#Plotting script for results from MSD networks on the X-ray data

# This script plots the results of MSD networks on the X-ray based testing data
# The plots show how the number data reduction channels influences the MSD segmentation results on the simulated X-ray dataset for all data reduction methods
# The results are the plots files in the results/X-ray/plots/ folder

# The code assumes that the csv files with MSD network results on testing data are available in the results/X-ray/quantitative/ folder

#Author,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os

os.makedirs('../../../results/X-ray/plots/', exist_ok=True)

#Read csv file containing the data
with open("../../../results/X-ray/quantitative/X-rayDRMSDReductionChannelsResults.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    next(reader, None)  #Skip the headers
    data_read = [row for row in reader]

#Prepare the data for plotting
binvalues = list(set([int(x[2]) for x in data_read]))
binvalues.sort()
DRMethods = ['DRMSD', 'MSD', 'LDA', 'NMF', 'PCA']
DRMethodsLabels = ['DRCNN', 'No red.', 'LDA', 'NMF', 'PCA']
cnt = 0
DRValueList = []
for DR in DRMethods:
    valueList = []
    for x in data_read:
        if x[1] == DR:
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
plt.xscale('log')
plt.title('MSD segmentation results on noisy multi-material dataset')
ax1.set_xticks([1, 2, 10, 60, 200, 300])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.tick_params(which='minor', bottom=False)
plt.legend()
plt.savefig("../../../results/X-ray/plots/X-rayDRMSDReductionChannelsResults.png", dpi=500)
plt.savefig("../../../results/X-ray/plots/X-rayDRMSDReductionChannelsResults.eps", format='eps', dpi=500)
plt.show()
