#Creation of spectral projections from material projections

#This scripts converts material projections into (sufficiently) realistic spectral projections
#The results are the hyperspectral X-ray projections for the training and test sets in /data/X-rayDatasets/ManyMaterialsFalse and /data/X-rayDatasets/ManyMaterialsFalse true folders
#In the ManyMaterialsTrue setting there are 60 different materials (among which 1 has to be recovered by identifying 2 cylinders)
#In the ManyMaterialsFalse setting there are 2 different materials (among which 1 has to be recovered by identifying 2 cylinders, while the others are 1 other material)

#The code assumes that the MaterialProjectionsGenerator.py have been carried out first to produce the material projections

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import os
from SpectralProjectorComputer import *

#GT for LDA reduction
Scheme = [[2, [list(range(0, 17)) + list(range(18, 60)), [17]]],
          [10, [list(range(0, 6)), list(range(6, 12)), list(range(12,18)), list(range(19,25)), list(range(25,32)), list(range(32,39)), list(range(39,46)), list(range(46,53)), list(range(53,60)), [17]]],
          [60, [[x] for x in range(0,17)] + [[x] for x in range(18,60)] + [[17]]]]

Ninstances = 1    #Number of instances to create

def SpectralDataGenerator(ManyMaterials):

    #Set random seed (once)
    np.random.seed(123) 

    #Number of averages to compute the averaged flatfield image from (for noisy projections)
    FFAverages = 50

    #Output folder for the spectral projections
    outputPath = '../../../data/X-rayDatasets/ManyMaterials' + str(ManyMaterials) + '/'
    os.makedirs(outputPath + 'Data', exist_ok=True)
    os.makedirs(outputPath + 'DataNoise', exist_ok=True)
    if(ManyMaterials):
        for List in Scheme:
            os.makedirs(outputPath + '/GTLDA' + str(List[0]), exist_ok=True)

    #Input folder for the material projections
    projsPath = '../../../data/X-rayDatasets/X-rayMatProjs/'

    #Make spectral projector object
    PG = Spectral()

    #Energy settings
    PG.EnergyBounds = None  #Set to None for equidistant bounds
    PG.MinEnergy = 14
    PG.MaxEnergy = 69
    PG.EnergyBins = 300
    PG.IntegralBins = 30
    PG.exposureTime = 0.5

    #Material settings
    PG.AttSpectraInfo = []
    if(ManyMaterials):
        #Set every two cylinders to different materials
        for i in range(0,60):
            PG.AttSpectraInfo.append([i+1, 60-30+i, None])
    else:
        #Set two cylinders to silver (47) and the remaining cylinders to cadmium (48) 
        for i in range(0,17):
            PG.AttSpectraInfo.append([i+1, 48, None])
        PG.AttSpectraInfo.append([18, 47, None])
        for i in range(18,60):
            PG.AttSpectraInfo.append([i+1, 48, None])
    print(PG.AttSpectraInfo)

    PG.Materials = len(PG.AttSpectraInfo)

    #Create flatfield image
    print("Making the flatfield image...")
    PG.PathToProjs = 'Flatfield'
    PG.Instances = 1
    PG.InstancesStart = 1
    FF = PG.MakeSpectralProjection()
    PG.PathToProjs = projsPath
    #Make noisy version
    FFSum = np.random.poisson(FF)
    for i in range(1, FFAverages):
        FFSum += np.random.poisson(FF)
    FFNoise = FFSum/float(FFAverages)
    print("Flatfield images created!")

    #Loop over all material projection instances
    for i in range(0, Ninstances):
        print(i)
        PG.Instances = 1
        PG.InstanceStart = i

        Res = PG.MakeSpectralProjection()

        #Apply noise
        ResNoise = np.random.poisson(Res)

        #Normalize by the flatfield image
        Proj = np.log(FF[0,:,:,:]/Res[0,:,:,:])
        ProjNoise = np.log(FFNoise[0,:,:,:]/ResNoise[0,:,:,:])
        
        #Remove inf and nan values
        Proj[~np.isfinite(Proj)] = 0
        ProjNoise[~np.isfinite(ProjNoise)] = 0

        #Save the projection data to the selected folders
        tifffile.imsave(outputPath + 'Data/{:05d}Data.tiff'.format(i), Proj.astype(np.float32))
        tifffile.imsave(outputPath + 'DataNoise/{:05d}Data.tiff'.format(i), ProjNoise.astype(np.float32))

        #Make GT by taking one of the entries in PG.AttSpectraInfo
        os.makedirs(outputPath + 'GT', exist_ok=True)
        SliceGTStart1 = 34  #Material 47
        SliceGTStop1 = 35 
        
        GT = tifffile.imread(PG.PathToProjs + 'Instance' + str(i).zfill(3) + '.tiff')
        GT = GT[SliceGTStart1:SliceGTStop1+1,:,:].sum(axis=0)
        GT[GT > 0] = 1
        
        tifffile.imsave(outputPath + 'GT/{:05d}GT.tiff'.format(i), GT.astype(np.uint8))

        #Make GT for the data reduction with LDA
        if(ManyMaterials):
            for List in Scheme:
                GT = tifffile.imread(PG.PathToProjs + 'Instance' + str(i).zfill(3) + '.tiff')
                GTSep = np.zeros((60,GT.shape[1], GT.shape[2]))
                for l in range(0,60):
                    GTSep[l,:,:] = GT[2*l:2*l+2,:,:].sum(axis=0)
                GTSep[GTSep > 0] = 1
            
                GTOutput = np.zeros((GTSep.shape[1], GTSep.shape[2]))

                cnt = 1
                for l in List[1]:
                    print(l)
                    for el in l:
                        GTOutput[GTSep[el,:,:] > 0] = cnt
                    cnt += 1
                tifffile.imsave(outputPath + '/GTLDA' + str(List[0]) + '/{:05d}GT.tiff'.format(i), GTOutput.astype(np.uint8))    

        #Save list of materials and other settings
        with open(outputPath + 'materialsettings.txt', 'w') as f:
            f.write("ManyMaterials: %s\n" % ManyMaterials) 
            f.write("FFAverages: %s\n" % FFAverages) 
            f.write("MinEnergy: %s\n" % PG.MinEnergy) 
            f.write("MaxEnergy: %s\n" % PG.MaxEnergy)
            f.write("EnergyBins: %s\n" % PG.EnergyBins)
            f.write("IntegralBins: %s\n" % PG.IntegralBins) 
            f.write("ExposureTime: %s\n" % PG.exposureTime)
            f.write("SliceGTStart1: %s\n" % SliceGTStart1)
            f.write("SliceGTStop1: %s\n" % SliceGTStop1)
            for item in PG.AttSpectraInfo:
                f.write("%s\n" % item)
                
#Run the spectral data generator for both limited number of materials and many materials 
SpectralDataGenerator(False)
SpectralDataGenerator(True)
