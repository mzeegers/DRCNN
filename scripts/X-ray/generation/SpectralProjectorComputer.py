import numpy as np
import os
import scipy.interpolate
import csv
import operator
import tifffile
import pyqtgraph as pq
#import physdata.xray
import matplotlib.pyplot as plt
from ElementaryData import *

NIST = False    #If True, real attenuation data is taken from NIST website, otherwise from local copy

###Additional code for parallelizing CPU operations###
import ctypes
lib = ctypes.CDLL('./operations2.so')
aslong = ctypes.c_uint64
asfloat = ctypes.c_float
asuint = ctypes.c_uint
cfloatp = ctypes.POINTER(ctypes.c_float)
def asfloatp(arr):
    return arr.ctypes.data_as(cfloatp)
# Try to set number of threads to number of physical cores
try:
    import psutil
    ncpu = psutil.cpu_count(logical=False)
    naff = len(psutil.Process().cpu_affinity())
    if naff < ncpu:
        ncpu = naff
    lib.set_threads(asuint(ncpu))
except ImportError:
    pass
###


#Converts a given material name to its atomic number
def elementToAtomic(materialname):
    Candidates = [x for x in ElementaryData if x[1] == materialname]
    if not Candidates:
        return 0
    else:
        return next(x for x in ElementaryData if x[1] == materialname)[0]


class Spectra(object):
    def __init__(self):
        self.Labels = [(0, 'Void')]
        self.SourceSpectrum = None #Scipy function that determines the photon source intensity given an energy (constructed using linear interpolation)
        self.AttenuationSpectra = []    #List containing all relevant attenuation spectrum functions (constructed on the fly)
                                        #Contains tuples of the form (AtomicNumber, Name, EnergyData, AttenuationData, InterpolationFunction) 

    #Initialize source spectrum from file, array or default file (in files the entries are separated by commas)
    def InitializeSourceSpectrum(self, path = None):
        if path is not None:
            #Read the given file
            with open(path) as f:
                lines = f.readlines()
                lines = [line.replace(',', '.') for line in lines]  # Comma decimal marker -> dot
                lines.insert(0, 'keV;photon_density\n')             # Manually add column labels
            dialect = csv.Sniffer().sniff(lines[0], delimiters=';')
            reader = csv.DictReader(lines, dialect=dialect)
            energies = np.empty(len(lines) - 1)
            photon_dens = np.empty(len(lines) - 1)
            for i, row in enumerate(reader):
                energies[i] = float(row['keV']) 
                photon_dens[i] = float(row['photon_density'])       # [mm^(-2)]
        else:
            energies = [0,1000]
            photon_dens = [1,1]
        #Create an interpolation function for the given energt-photon density tuples
        source_spectrum = scipy.interpolate.interp1d(energies, photon_dens)
        self.SourceSpectrum = source_spectrum

    #Collect all relevant attenation spectra
    def collectAttenuationSpectra(self, Rootdatapath):
        print("Loading attenuation spectra...")
        
        for mat in self.Labels:
            if(mat[0] != 0 and mat[1] != "Void"): #Exclude Voids                
                if (mat[1] in [i[1] for i in ElementaryData]):                          #Elementary material
                    AtNo = elementToAtomic(mat[1])
                    if (AtNo > 0):
                        attData = self.getAttenuationSpectrum(AtNo, Rootdatapath)
                        self.AttenuationSpectra.append((mat[0],)+(mat[1],) + attData)
                else:
                    attData = self.getAttenuationSpectrum(mat[1], Rootdatapath)         #Mixture material
                    self.AttenuationSpectra.append((mat[0],)+(mat[1],) + attData)
            elif(mat[0] != 0 and mat[1] == "Void"):
                #Make the zero attenuation spectrum
                attData = getAttenuationSpectrum(0, Rootdatapath)
                x, y = np.arange(0,100), np.zeros(100)
                spectrum = scipy.interpolate.interp1d(x, y)
                self.AttenuationSpectra.append((mat[0],)+('Void',) + (x,y,spectrum))

        self.AttenuationSpectra.sort(key = operator.itemgetter(0)) #Keep sorted on atomic number  
        print("Attenuation spectra fully loaded")

        print(self.AttenuationSpectra)

    #Add labels for material index (value) to list of labels
    def updateLabel(self, value, labeling):
        if isinstance(labeling, int):
            label = ElementaryData[labeling][1]
        else:
            label = labeling
        self.Labels.append((value, label))

    #Get attenuation spectrum for a given material number
    def getAttenuationSpectrum(self, materialno, rootdatapath):
        if NIST == True:    #Take data from online NIST website
            data = np.array(physdata.xray.fetch_coefficients(materialno)) #Density taken from array
        else:               #Take data from local copy of the NIST website
            data = self.fetchCoefficientsCustom(materialno, rootdatapath) #Density taken from array
        data[:, 0] *= 1000  #Convert from MV to kV
       
        #Create (linear) interpolation function
        x, y = data[:, 0], data[:, 1]
        spectrum = scipy.interpolate.interp1d(x, y)

        return data[:,0], data[:,1], spectrum

    #Get attenuation spectrum data from local copy (in case NIST is unreachable)
    def fetchCoefficientsCustom(self, arg, Rootdatapath):
        #Helper function for finding files with prefixes
        def findPrefixFile(prefix, path):
            for i in [f for f in sorted(os.listdir(path)) if not f.endswith('~')]:
                if os.path.isfile(os.path.join(path,i)) and prefix in i:
                    return i
        #Find the data file
        if type(arg) is int or type(arg) is np.uint8:
            path = Rootdatapath + 'DataElemental/'
            pathtofile = path + findPrefixFile(str(arg).zfill(2) +'-', path)
        elif type(arg) is str:
            path = Rootdatapath + 'DataMixture/'
            pathtofile = path + findPrefixFile(arg + ' - ', path)
        #Open the data file and read and parse contents
        with open(pathtofile) as f:
            content = f.readlines()
        content = np.array(content)
        content2 = np.zeros(((content.size),3))
        for ind, i in enumerate(content):
            dataline = i.split(' ')
            dataline = [x for x in dataline if (x != '' and x != '\n')]
            content2[ind,:] = [float(x) for x in dataline[-3:]]
        #Return nx3 table
        return content2


class ProjectionGenerator(object):
    def __init__(self):
        #Material projections
        self.PathToProjs = '../../../data/X-rayDatasets/X-rayMatProjs/'                                             #Path to folder with material projections

        self.sizey = 512                                                                                            #Size of the material projections y
        self.sizex = 512                                                                                            #Size of the material projections x
        self.Materials = 3                                                                                          #Number of materials, excluding empty area
        self.InstanceStart = 1                                                                                      #Instance to start from
        self.Instances = 2                                                                                          #Number of subsequent instances to load (always >= 1)

        #Source spectrum
        self.fileSource = '../../../data/X-rayDatasets/SourceSpectra/Radiology_Source_Thungsten_70kV_NoFilter.csv'  #Path to source spectrum
        self.exposureTime = 0.5                                                                                     #Exposure time (this includes the current)      
        self.detPixelSize = 0.11                                                                                    #mm
        self.scaling = 13976000                                                                                     #Look in spectrumscaling.txt for the right values
        self.SourceSpectrumScaling = self.scaling*self.exposureTime*self.detPixelSize*self.detPixelSize                                                   

        #Material spectra
        self.RootDataPath = self.Rootdatapath = '../../../data/X-rayDatasets/NIST/RawData/'                         #Location of the NIST data
        self.AttSpectraInfo = [[1,7,None],[2,None,np.array([[1,40],[20,20]])],[3,'vinyl',None]]                     #List of material properties in the following format [identifier, atomnumber/NIST material name, file/array]
        self.Materials = len(self.AttSpectraInfo)                                                                   #One of the latter two has to be None, otherwise the first argument will go first
             
        #Energy bin partition
        self.EnergyBounds = None                                                                                    #Energy partition, if empty then parition will be made with values below (with equal spacing)
        self.MinEnergy = 14                                                                                         #Lower bound energy partition
        self.MaxEnergy = 69                                                                                         #Upper bound energy partition
        self.EnergyBins = 300                                                                                       #Number of energy bins

        #Projection computation
        self.VoxelSize = 0.011                                                                                      #cm
        self.IntegralBins = 30                                                                                  
        
        #Setup spectra
        self.Sp = Spectra() 
        
        #Prepare source spectrum array
        self.Sp.InitializeSourceSpectrum(path = self.fileSource)
    
        
    def MakeSpectralProjection(self):
    
        #Reading material projections
        def readMatProjection(inst, mat, path, Materials):
            Res = tifffile.imread(path + '/Instance' + str(inst).zfill(3) + '.tiff')
            if(Materials < Res.shape[0]):
                Res = Res[mat*int(Res.shape[0]/Materials):(mat+1)*int(Res.shape[0]/Materials),:,:].sum(axis=0)
            return Res

        def readAllMatProjections(inst, Materials, path, sizey, sizex):
            Res = np.zeros((Materials, sizey, sizex), dtype=np.float32)
            for mat in range(1, Materials + 1):
                Res[mat-1,:,:] = readMatProjection(inst, mat-1, path, Materials)
            return Res

        def readAllInstancesAndMatProjections(InstanceStart, Instances, Materials, path, sizey, sizex):
            Res = np.zeros((Instances, Materials, sizey, sizex), dtype=np.float32)
            for inst in range(0, Instances):
                Res[inst-1,:,:,:] = readAllMatProjections(InstanceStart+inst, Materials, path, sizey, sizex)
            return Res

        if self.PathToProjs == 'Flatfield':
            MatProjs = np.zeros((self.Instances, self.Materials, self.sizey, self.sizex), dtype=np.float32)
        else:
            MatProjs = readAllInstancesAndMatProjections(self.InstanceStart, self.Instances, self.Materials, self.PathToProjs, self.sizey, self.sizex)   #Output has shape (#self.instances, #materials, self.sizey, self.sizex)
        
        #Prepare attenuation spectrum arrays
        for idx, item in enumerate(self.AttSpectraInfo):
            if item[0] <= self.Materials:
                print("Material identifier and element",item[0],item[1])      #Elemental material or mixture
                self.Sp.updateLabel(item[0], item[1])
            else:
                print("No material identifier ", item[0], "in the data")

        #Collect the attenuation spectra of all involved materials
        self.Sp.collectAttenuationSpectra(self.RootDataPath)

        #Make special plastic attenuation spectrum
        attData = self.Sp.getAttenuationSpectrum('polyethylene', self.RootDataPath)
        PlasticSpectrum = attData[2]

        #Select the energy range and precision over all projections
        if self.EnergyBounds is None:
            self.EnergyBounds = np.linspace(self.MinEnergy, self.MaxEnergy, num = self.EnergyBins+1)
        else:
            self.EnergyBins = len(self.EnergyBounds)-1
        print("Energy binning:", self.EnergyBounds)

        #Create projection for all instances
        AllInstProj = np.zeros((self.Instances, self.EnergyBins, self.sizey, self.sizex), dtype=np.float32)
        for inst in range(0, self.Instances):
            print("Instance", inst)

            MatProjsInst = MatProjs[inst,:,:,:]

            for i in range(0,len(self.EnergyBounds)-1):
                print("Bin [", self.EnergyBounds[i], ",", self.EnergyBounds[i+1], "]")
                
                #Make full integral energy projection - using midpoint rule
                EMin = self.EnergyBounds[i]
                EMax = self.EnergyBounds[i+1]
                energies = np.linspace(EMin, EMax, num = self.IntegralBins+1)
                binWidth = (EMax - EMin)/self.IntegralBins

                #Compute material summation in the exponent 
                atts = []
                for m in range(1, self.Materials+1): 
                    atts.append([x[4] for x in self.Sp.AttenuationSpectra if x[0] == m])                
                TotalProjection = np.zeros((self.sizey, self.sizex), dtype=np.float32)
                for e in range(0, self.IntegralBins):
                    lib.zero(asfloatp(TotalProjection.ravel()), aslong(TotalProjection.size))
                    for m in range(1, self.Materials+1): 
                        #Find the spectrum related to material m
                        Sp = atts[m-1]
                        att = (Sp[0]((energies[e]+energies[e+1])*0.5)*0.01 + PlasticSpectrum((energies[e]+energies[e+1])*0.5)*0.99)*self.VoxelSize
                        lib.cplusab(asfloatp(MatProjsInst[m-1].ravel()),asfloat(att), asfloatp(TotalProjection.ravel()), aslong(TotalProjection.size))
                    lib.cplusexpab(asfloatp(TotalProjection.ravel()),asfloat(binWidth*self.Sp.SourceSpectrum((energies[e]+energies[e+1])*0.5)*self.SourceSpectrumScaling), asfloatp(AllInstProj[inst, i]), aslong(TotalProjection.size))

        return AllInstProj
