#Creation of remote sensing images from material configuration

#This scripts converts material configurations into hyperspectral remote sensings based images
#The results are the hyperspectral remote sensing based images for the training and test sets in /data/RemoteSensingDatasets/PartiallyOverlappingFalse/Data/ and /data/RemoteSensingDatasets/PartiallyOverlappingTrue/Data/ folders
#In the PartiallyOverlappingFalse setting there 20 to-be detected cylinders of 10 different materials without overlapping the other cylinders of 50 different materials (2 cylinders of each material)
#In the PartiallyOverlappingTrue setting there 20 to-be detected cylinders of 10 different materials that can overlap other cylinders (2 cylinders of each material), among 50 different materials

#The code assumes that the EnvironmentsGenerator.py and EnvironmentsGeneratorPartiallyOverlapping.py scripts have been carried out first to produce the material configurations

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import numpy as np
import os
import scipy.interpolate
import csv
import operator
import tifffile


class Spectra(object):
    def __init__(self):
        self.Labels = [(0, 'Void')]
        self.AttenuationSpectra = []    #List containing all relevant attenuation spectrum functions (constructed on the fly)
                                        #Contains tuples of the form (Number, Name, Wavelengths, ReflectanceData, InterpolationFunction) 
    
        self.MaterialTypes = 'ChapterV_Vegetation/'
        self.path = '../../../data/RemoteSensingDatasets/RemoteSensingSpectra/usgs_splib07/ASCIIdata/ASCIIdata_splib07b_cvAVIRISc2014/'
    
    #Read coefficients from a given file
    def ReadCoefficients(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines.pop(0)
            lines = [line.replace(',', '.') for line in lines]  # Comma decimal marker -> dot
            lines.insert(0, 'Wavelength;')                      # Manually add column labels

            dialect = csv.Sniffer().sniff(lines[0])
            reader = csv.DictReader(lines, dialect=dialect)

            Wavelengths = np.empty(len(lines) - 1)
            for i, row in enumerate(reader):
                Wavelengths[i] = float(row['Wavelength']) 

            return Wavelengths
    
    #Get attenuation spectrum for a given material number
    def getAttenuationSpectrum(self, materialno):

        #Open the files with wavelenghts and the spectrum values
        x = self.ReadCoefficients(self.path + 's07_AV14_Wavelengths_in_microns_224_ch_AVIRIS14a.txt')
        Filenames = sorted(os.listdir(self.path + self.MaterialTypes))
        FullFileName = self.path + self.MaterialTypes + Filenames[materialno]
        y = self.ReadCoefficients(FullFileName)

        #Remove all undefined elements
        indices = [i for i,v in enumerate(y) if v > 0]
        x = x[indices]
        y = y[indices]   

        #Create (linear) interpolation function
        spectrum = scipy.interpolate.interp1d(x, y)

        return x, y, spectrum 
    
    #Collect all relevant remote sensing spectra
    def collectAttenuationSpectra(self):
        print("Loading attenuation spectra...")

        for mat in self.Labels:
            if(mat[0] != 0 and mat[1] != "Void"): #Exclude Voids        
                attData = self.getAttenuationSpectrum(mat[1])
                self.AttenuationSpectra.append((mat[0],)+(mat[1],) + attData)        
            elif(mat[0] != 0 and mat[1] == "Void"):
                #Make the zero attenuation spectrum
                x, y = np.arange(0,100), np.zeros(100)
                spectrum = scipy.interpolate.interp1d(x, y)
                self.AttenuationSpectra.append((mat[0],)+('Void',) + (x,y,spectrum))

        self.AttenuationSpectra.sort(key = operator.itemgetter(0)) #Keep sorted on atomic number  
        print("Attenuation spectra fully loaded")

        print(self.AttenuationSpectra)

    #Update the list of material labels
    def updateLabel(self, value, labeling, autoLabel = True):
        self.Labels.append((value, labeling))


class RemoteSensingGenerator(object):
    def __init__(self):

        #Input and output paths
        self.PathToConfs = '../../../data/RemoteSensingConfigurations/' #Path to folder with remote sensing configurations
        self.outputPath = ''            

        #Image characteristics
        self.sizey = 512                                                #Size of the image y
        self.sizex = 512                                                #Size of the image x
        self.Materials = 3                                              #Number of materials, excluding empty area
        self.InstanceStart = 0                                          #Instance to start from
        self.Instances = 2                                              #Number of subsequent instances to load (always >= 1)                                  

        #Material spectra
        self.AttSpectraInfo = [[1,7,None]]                              #List of material properties in the following format [identifier, atomnumber/NIST material name, file/array]
                                                                        #One of the latter two has to be None, otherwise the first argument will go first
             
        #Energy bin partition
        self.EnergyBounds = None                                        #Energy partition, if empty then parition will be made with values below (with equal spacing)
        self.MinEnergy = 0.45                                           #Lower bound energy partition
        self.MaxEnergy = 2.4                                            #Upper bound energy partition
        self.EnergyBins = 200                                           #Number of energy bins

        #Image computation
        self.IntegralBins = 30                                                                      

    def MakeSpectralImage(self):
        
        #Read first instance to get the dimensions
        MatImgs = tifffile.imread(self.PathToConfs + '/Instance' + str(self.InstanceStart).zfill(3) + '.tiff')   

        Sp = Spectra()
        
        #Prepare remote sensing spectrum arrays
        for idx, item in enumerate(self.AttSpectraInfo):
            if item[0] <= self.Materials:
                print("Material identifier and element",item[0],item[1])
                Sp.updateLabel(item[0], item[1], True)
            else:
                print("No material identifier ", item[0], "in the data")

        print(Sp.Labels)
        
        #Get all the relevant reflectance spectra
        Sp.collectAttenuationSpectra()

        #Select the energy range and precision over all images
        if not self.EnergyBounds:
            self.EnergyBounds = np.linspace(self.MinEnergy, self.MaxEnergy, num = self.EnergyBins+1)
        else:
            self.EnergyBins = len(self.EnergyBounds)-1
        print("Energy binning:", self.EnergyBounds)

        #Compute reflectance spectra images for all materials
        MatReflectances = np.zeros((MatImgs.shape[0], self.EnergyBins))
        for e in range(0, self.EnergyBins):
            print(e)
            EMin = self.EnergyBounds[e]
            EMax = self.EnergyBounds[e+1]
            energies = np.linspace(EMin, EMax, num = self.IntegralBins+1)

            for m in range(0, MatReflectances.shape[0]):
                Spec = [x[4] for x in Sp.AttenuationSpectra if x[0] == m+1]
                for b in range(0, self.IntegralBins):
                    MatReflectances[m,e] += Spec[0]((energies[b]+energies[b+1])*0.5)/self.IntegralBins

        #Compute the final hyperspectral reflectance image
        for i in range(self.InstanceStart, self.Instances):
            print("Instance", i)

            MatImgs = tifffile.imread(self.PathToConfs + '/Instance' + str(i).zfill(3) + '.tiff')   
            MatImgsFlat = np.reshape(MatImgs, (MatImgs.shape[0], MatImgs.shape[1]*MatImgs.shape[2]))             
        
            ResImgs = np.matmul(MatImgsFlat.swapaxes(0,1), MatReflectances)
            ResImgs = ResImgs.swapaxes(0,1)
            ResImgs = ResImgs.reshape((self.EnergyBins, MatImgs.shape[1], MatImgs.shape[2]))
            
            #Remove inf and nan values (probably not present)
            ResImgs[~np.isfinite(ResImgs)] = 0
            
            #Save the hyperspectral image to the selected folders
            tifffile.imsave(self.outputPath + 'Data/{:05d}Data.tiff'.format(i), ResImgs.astype(np.float32))
           
            
            #Make ground truth
            GT = np.zeros((self.sizey, self.sizex))
            Zeros = np.all(MatImgs == 0, axis=0)
            
            if(self.PartiallyOverlapping):
                Indices = [x for x in range(0,360) if x not in range(0,360,6)]
                Zeros2 = np.all(MatImgs[range(0,360,6)] == 0, axis = 0)
                MatImgs2 = MatImgs.copy()
                MatImgs2[Indices,:,:] = 0 
                GT = np.argmax(MatImgs2,0)
                GT = GT%60
                GT = GT+1
                GT[Zeros] = 0
                GT[Zeros2] = 0
                
            else:
                GT = np.argmax(MatImgs,0)
                GT = GT%60
                GT = GT+1       
                GT[Zeros] = 0

            GT[(GT-1)%6 != 0] = 0
            GT = np.floor((GT + 5)/6)
        
            #Save the ground truth
            tifffile.imsave(self.outputPath + 'GT10Labels/{:05d}GT.tiff'.format(i), GT.astype(np.uint8))


def SpectralRemoteSensingDataGenerator(PartiallyOverlapping):

    #Make spectral image generator object
    RSG = RemoteSensingGenerator()

    #Energy settings
    RSG.EnergyBounds = None  #Set to None for equidistant bounds
    RSG.MinEnergy = 0.45
    RSG.MaxEnergy = 2.4
    RSG.EnergyBins = 200
    RSG.IntegralBins = 30

    #Material settings
    RSG.AttSpectraInfo = []
    DiffMaterials = 60
    Step = 2
    m = 0
    for i in range(0,360): 
        RSG.AttSpectraInfo.append([i+1, m, None])
        m = m+2
        if(m == DiffMaterials*Step):
            m = 0
    print(RSG.AttSpectraInfo)
    RSG.Materials = len(RSG.AttSpectraInfo)

    #Input folder for the material configurations
    if PartiallyOverlapping:
        RSG.PathToConfs = '../../../data/RemoteSensingDatasets/RemoteSensingConfigurations/'
    else: 
        RSG.PathToConfs = '../../../data/RemoteSensingDatasets/RemoteSensingConfigurationsPartiallyOverlapping/'

    #Output folder for the hyperspectral images
    outputPath = '../../../data/RemoteSensingDatasets/PartiallyOverlapping' + str(PartiallyOverlapping) + '/'
    os.makedirs(outputPath, exist_ok=True)
    os.makedirs(outputPath + 'Data/', exist_ok=True)
    RSG.outputPath = outputPath

    #Make GT folders
    os.makedirs(outputPath + 'GT10Labels/', exist_ok=True)

    #Settings for the hyperspectral image creation
    RSG.Instances = 100
    RSG.InstanceStart = 0
    RSG.PartiallyOverlapping = PartiallyOverlapping  

    #Create and save the hyperspectral images
    RSG.MakeSpectralImage()
        
    #Save list of materials and other settings
    with open(outputPath + 'materialsettings.txt', 'w') as f:
        f.write("MinEnergy: %s\n" % RSG.MinEnergy) 
        f.write("MaxEnergy: %s\n" % RSG.MaxEnergy)
        f.write("EnergyBins: %s\n" % RSG.EnergyBins)
        f.write("IntegralBins: %s\n" % RSG.IntegralBins) 
        for item in RSG.AttSpectraInfo:
            f.write("%s\n" % item)
            
#Run the spectral data generator for both non overlapping cylinders and overlapping cylinders
SpectralRemoteSensingDataGenerator(False)
SpectralRemoteSensingDataGenerator(True)
