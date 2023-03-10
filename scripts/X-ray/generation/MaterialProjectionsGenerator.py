#Generation of X-ray material projections

#This script carries out the generation of X-ray configurations which consist of 2D material projections of 3D cylinders of various shapes that do not overlap each other
#The results are a stack of images with one cylinder, which can be found in data/X-rayDatasets/X-rayMatProjs/

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import astra
import numpy as np
import os
import pyqtgraph as pq      #pyqtgraph always slices through the first axis of the 3D array
import random as rd
import scipy.ndimage
import tifffile
import random
from ElementaryData import *

np.set_printoptions(threshold=np.inf)

#Produce a 3D cylinder given with the provided parameters
def makeCylinder(l, a, ang1, ang2, Tops = True, show = False):
    #l:     length of the tube
    #a:     radius of the tube
    #ang1:  tilting in yx-plane (in degrees)
    #ang2   tilting in zx-plane (in degrees)

    #Set sizes
    if(Tops == True):
        maxHalfSize = np.amax([a,a+int((l+1)/2)])
        maxHalfSizeTube = int((l+1)/2)
    else:
        maxHalfSize = np.amax([a,int((l+1)/2)])   
    Frame = np.zeros((2*maxHalfSize+1,2*maxHalfSize+1,2*maxHalfSize+1))

    zz,yy,xx = np.mgrid[-maxHalfSize:maxHalfSize+1,-maxHalfSize:maxHalfSize+1,-maxHalfSize:maxHalfSize+1]

    #Make cylinder
    Dist = (zz**2 + yy**2)

    Frame[Dist < a**2] = 1
    if(Tops == True):
        Frame[xx > maxHalfSizeTube] = 0
        Frame[xx < -maxHalfSizeTube] = 0

    #Make cones on top
    if(Tops == True):
        Dist1 = (zz**2+yy**2+(xx-maxHalfSizeTube)**2)
        Frame[Dist1 < a**2] = 1
        Dist2 = (zz**2+yy**2+(xx+maxHalfSizeTube)**2)
        Frame[Dist2 < a**2] = 1

    #Rotate according to given angles
    Frame = scipy.ndimage.rotate(Frame, ang1, axes=(1,2), reshape = False)
    Frame = scipy.ndimage.rotate(Frame, ang2, axes=(0,2), reshape = False)

    Frame[Frame >= 0.5] = 1
    Frame[Frame <= 0.5] = 0

    if(show):
        pq.image(Frame, title ='Z-slices')
        pq.image(np.swapaxes(Frame,0,1), title ='Y-slices')
        pq.image(np.swapaxes(np.swapaxes(Frame,0,2),1,2), title ='X-slices')

    if(Tops == True):
        print("Made cylinder with length l =", l, ", radius a =", a, "and angles (yx) and (zx) =", ang1, ang2, "with tops.")
    else:
        print("Made cylinder with length l =", l, ", radius a =", a, "and angles (yx) and (zx) =", ang1, ang2, "without tops.")

    return(Frame, maxHalfSize)


#Insert given object Obj in given array Field with a given value (Assumption: Field is square (SIZE is size of the world))
def insertObject(Obj, centerz, centery, centerx, radius, value, Field, SIZE, checkFrame = False, AllowList = [0]):
    Obj[Obj == 1] = value

    #Determine copy ranges
    minzField = max(0, centerz-radius)
    maxzField = min(SIZE, centerz+radius+1)
    minyField = max(0, centery-radius)
    maxyField = min(SIZE, centery+radius+1)
    minxField = max(0, centerx-radius)
    maxxField = min(SIZE, centerx+radius+1)

    minzObj = max(0, radius - centerz)
    maxzObj = min(2*radius+1, radius - centerz + SIZE)
    minyObj = max(0, radius - centery)
    maxyObj = min(2*radius+1, radius - centery + SIZE)
    minxObj = max(0, radius - centerx)
    maxxObj = min(2*radius+1, radius - centerx + SIZE)

    ObjSlice = Obj[minzObj:maxzObj,minyObj:maxyObj,minxObj:maxxObj]    

    FieldRanges = Field[minzField:maxzField,minyField:maxyField,minxField:maxxField]

    #Check whether position is already occupied
    if checkFrame == True:
        if np.all(np.in1d(FieldRanges[ObjSlice!=0], AllowList)) == False:
            print("Object cannot be placed: space already occupied by other object")
            return False

    #Insert the object
    Field[minzField:maxzField,minyField:maxyField,minxField:maxxField][ObjSlice > 0] = ObjSlice[ObjSlice>0]

    print("Inserted object at location = (z,y,x)", centerz, centery, centerx)
    return True


class Phantom(object):
    def __init__(self):
        self.defaultValues()

    #Default values of source and detector properties
    def defaultValues(self):

        #Phantom array and environment
        self.Ph = None
        self.vol_geom = None

        #Geometric default variables
        self.radiusSrc = 4000
        self.radiusDet = 1000

        self.zSrc = 0
        self.ySrc = -self.radiusSrc
        self.xSrc = 0

        self.zDet = 0
        self.yDet = self.radiusDet            
        self.xDet = 0

        #Detector sizes
        self.detSizey = 1536
        self.detSizex = 1536

        self.Labels = [(0, 'Void')]

    def clear(self):
        self.Labels.clear()
        self.Labels.append((0, 'Void'))

    #Forward project a given array with ASTRA using a vector geometry
    def forwardProject(self, World = None):

        if World is None:
            World = self.Ph

        self.vol_geom = astra.create_vol_geom(World.shape)   
        Proj = np.zeros((self.detSizey,self.detSizex))
        vectors = np.zeros((1, 12))
        
        #Source position
        vectors[0,0] = self.zSrc
        vectors[0,1] = self.ySrc  #at the start, this is the source-detector axis
        vectors[0,2] = self.xSrc

        #Center of detector
        vectors[0,3] = self.zDet
        vectors[0,4] = self.yDet
        vectors[0,5] = self.xDet
                
        #Vector from detector pixel (0,0) to (0,1)
        vectors[0,6] = 1
        vectors[0,7] = 0
        vectors[0,8] = 0
                
        #Vector from detector pixel (0,0) to (1,0)
        vectors[0,9] = 0
        vectors[0,10] = (self.xSrc - self.xDet)/(np.sqrt(np.power(self.xDet - self.xSrc,2) + np.power(self.ySrc - self.yDet,2)))
        vectors[0,11] = (self.yDet - self.ySrc)/(np.sqrt(np.power(self.xDet - self.xSrc,2) + np.power(self.ySrc - self.yDet,2)))

        proj_geom = astra.create_proj_geom('cone_vec', self.detSizey, self.detSizex, vectors)

        proj_id, proj_data = astra.create_sino3d_gpu(World, proj_geom, self.vol_geom)

        Proj = proj_data[:,0,:]

        return Proj
    
    #Create material projections - collect labels in GroundTruth and include these in the forward projection    
    def groundTruthProjection(self, GroundTruth = None, Binary = False):
       
        print(GroundTruth)
        #Backup the world
        newWorld = np.copy(self.Ph)

        for i in GroundTruth:
            newWorld[newWorld != i] = 0
            newWorld[newWorld == i] = 1

        Proj = self.showView(newWorld, Binary)

        newWorld = None

        return Proj
    
    #Carry out projection and make it (possibly) binary
    def showView(self, World = None, Binary = False):

        if World is None:
            World = self.Ph

        Res = np.zeros((1,self.detSizey,self.detSizex))
        Res[0,:,:] = self.forwardProject(World=World)
            
        if(Binary is True):
            Res[Res > 0] = 1

        return Res

    #Update material labels
    def updateLabel(self, value):
        label = None
        if(value > 120):
            label = 'DummyLabel' + str(value)
        else:
            label = ElementaryData[value][1]
        self.Labels.append((value, label))

    # Main part: Generating cylinder material projections #
    def ExampleMaker(self):
        
        Ninstances = 100    #Number of instances to create
        NCyl = 120          #Number of cylinders in the phantom
        Rsf = 3             #Factor to resize with
        
        #Create output directory
        os.makedirs('../../../data/X-rayDatasets/X-rayMatProjs/', exist_ok=True)
        
        for inst in range(0, Ninstances):

            print("INSTANCE", inst)

            np.random.seed(124+inst)
            random.seed(124+inst)
            
            self.clear()

            #Make cube
            WSIZE = 1024                #size of the world
            CSIZE = 1024                #size of the cube 
            HWSIZE = int(WSIZE/2)       #half the size of the world
            HCSIZE = int(CSIZE/2)       #half the size of the cube
            
            self.Ph = np.zeros((WSIZE,WSIZE,WSIZE))

            #Make other cylinders
            i = 1
            while (i < NCyl+1):
                print(i)
                #l:     length of the tube
                #a:     radius of the tube
                #ang1:  tilting in yx-plane (in degrees)
                #ang2   tilting in zx-plane (in degrees)
                l = rd.randint(39,390)
                a = rd.randint(12,30)
                ang1 = rd.randint(0,90)
                ang2 = rd.randint(0,90)

                #Make cylinder
                cyl, hsize = makeCylinder(l,a,ang1,ang2, Tops = True, show = False)

                #Select random location
                pz = rd.randint(HWSIZE-HCSIZE+hsize,HWSIZE+HCSIZE-hsize)
                py = rd.randint(HWSIZE-HCSIZE+hsize,HWSIZE+HCSIZE-hsize)
                px = rd.randint(HWSIZE-HCSIZE+hsize,HWSIZE+HCSIZE-hsize)

                #Try to insert the cylinder without overlapping other objects
                Inserted = insertObject(cyl, pz , py , px, hsize, i, self.Ph, WSIZE, True, [0])
                if(Inserted == True):
                    self.updateLabel(i)
                    i = i+1

            #Make and save material projections individually
            MatProj = np.zeros((NCyl, self.detSizey, self.detSizex), dtype = np.float32)
            for i in self.Labels:
                if i[0] != 0:
                    MatProj[i[0]-1,:,:] = self.groundTruthProjection([i[0]], Binary = False)            

            ###Resizing the data by factor Rsf
            RsS = int(MatProj.shape[-1]/Rsf)    #New size of the projection          
            ProjResize = np.zeros((MatProj.shape[0],RsS,RsS))
            for resi in range(Rsf):
                for resj in range(Rsf):
                    ProjResize += MatProj[:,resi::Rsf,resj::Rsf]
            MatProjCROP = (ProjResize/(Rsf*Rsf*Rsf)).astype(np.float32)     #3D normalization

            #Save the result
            tifffile.imsave('../../../data/X-rayDatasets/X-rayMatProjs/Instance' + str(inst).zfill(3) + '.tiff', MatProjCROP[:,:,:].astype(np.float32))

def main():
    Ph = Phantom()
    Ph.ExampleMaker()

if __name__ == "__main__":
    main()
