#Creation of noisy remote sensing based images from clean images

#This scripts converts clean hyperspectral remote sensing based images into images with (sufficiently) realistic noise
#The results are the noisy hyperspectral remote sensing images for the training and test sets in /data/RemoteSensingDatasets/PartiallyOverlappingFalse/DataSourceNoise/ and /data/RemoteSensingDatasets/PartiallyOverlappingTrue/DataSourceNoise folders

#The code assumes that the EnvironmentsGenerator.py and the SpectralRemoteSensingDataGenerator.py scripts have been carried out first to produce the clean images

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import numpy as np
import os
import scipy.interpolate
import operator
import tifffile

def ApplyNoise(PartiallyOverlapping):

    #Set random seed
    np.random.seed(123)

    #Collect and create the solar spectrum
    arrayWavelenghts = []
    with open('../../../data/RemoteSensingDatasets/RemoteSensingSpectra/SolarSpectrumWavelengths.txt') as f:
        for line in f: # read rest of lines
            arrayWavelenghts.append([float(x) for x in line.split()])
    arrayWavelenghts[:] = [x[0]/1000 for x in arrayWavelenghts]

    arrayValues = []
    with open('../../../data/RemoteSensingDatasets/RemoteSensingSpectra/SolarSpectrumValues.txt') as f:
        for line in f: # read rest of lines
            arrayValues.append([float(x) for x in line.split()])
    arrayValues[:] = [x[0] for x in arrayValues]

    solar_spectrum = scipy.interpolate.interp1d(arrayWavelenghts, arrayValues)

    #Set energy setting
    MinEnergy = 0.45
    MaxEnergy = 2.4
    EnergyBins = 200

    #Compute the solar spectrum images
    x = np.linspace(MinEnergy, MaxEnergy, EnergyBins)
    y = np.asarray([solar_spectrum(i) for i in x])

    IntegralBins = 30
    EnergyBounds = np.linspace(MinEnergy, MaxEnergy, num = EnergyBins+1)

    SolarFigures = np.zeros((EnergyBins,512,512))

    for e in range(0, EnergyBins):
        EMin = EnergyBounds[e]
        EMax = EnergyBounds[e+1]
        energies = np.linspace(EMin, EMax, num = IntegralBins+1)
        
        val = 0    
        for b in range(0, IntegralBins):
            val += solar_spectrum((energies[b]+energies[b+1])*0.5)/IntegralBins

        SolarFigures[e,:,:] = val

    #Output folder for the noisy hyperspectral spectral images (also the input folder for the clean hyperspectral images)
    outputPath = '../../../data/RemoteSensingDatasets/PartiallyOverlapping' + str(PartiallyOverlapping) + '/'
    os.makedirs(outputPath + 'DataSourceNoise/', exist_ok=True)

    #Compute the noisy images for all input instances
    for inst in range(0,100):
        print("Instance", inst)
        CleanImg = tifffile.imread(outputPath + 'Data/' + str(inst).zfill(5) + 'Data.tiff') 
        CleanImg2 = CleanImg * SolarFigures
        NoisyImg2 = CleanImg2 + np.random.normal(0,np.max(CleanImg2)/1000,(EnergyBins,512,512))
        TotalImg = NoisyImg2/SolarFigures
        tifffile.imsave(outputPath + 'DataSourceNoise/{:05d}Data.tiff'.format(inst), TotalImg.astype(np.float32))

#Run for non-partial overlapping and overlapping modes        
ApplyNoise(False)        
ApplyNoise(True)
