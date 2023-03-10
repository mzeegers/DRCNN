#Generation of remote sensing configurations without overlapping elements

#This script carries out the generation of remote sensing configurations which consist of cylinders of various shapes that do not overlap each other
#The results are a stack of images with one cylinder each, which can be found in data/RemoteSensingDatasets/RemoteSensingConfigurations/

#Authors,
#   Mathé Zeegers, 
#       Centrum Wiskunde & Informatica, Amsterdam (m.t.zeegers@cwi.nl)

import numpy as np
import os
import random as rd
import scipy.ndimage
import tifffile
import random

np.set_printoptions(threshold=np.inf)

os.makedirs('../../../data/RemoteSensingDatasets/', exist_ok=True)
os.makedirs('../../../data/RemoteSensingDatasets/RemoteSensingConfigurations/', exist_ok=True)

#Insert given object Obj in given array Field with a given value (Assumption: Field is square (SIZE is size of the Field))
def insertObject(Obj, centery, centerx, radius, value, Field, SIZE, checkFrame = False, AllowList = [0]):
    Obj[Obj > 0] = value

    #Determine copy ranges
    minyField = max(0, centery-radius)
    maxyField = min(SIZE, centery+radius+1)
    minxField = max(0, centerx-radius)
    maxxField = min(SIZE, centerx+radius+1)

    minyObj = max(0, radius - centery)
    maxyObj = min(2*radius+1, radius - centery + SIZE)
    minxObj = max(0, radius - centerx)
    maxxObj = min(2*radius+1, radius - centerx + SIZE)

    ObjSlice = Obj[minyObj:maxyObj,minxObj:maxxObj]    

    FieldRange = Field[minyField:maxyField,minxField:maxxField]
    
    #Check whether position is already occupied
    if checkFrame == True:
        if np.all(np.in1d(FieldRange[ObjSlice!=0], AllowList)) == False:
            print("Object cannot be placed: space already occupied by other object")
            return False

    #Insert the object
    Field[minyField:maxyField,minxField:maxxField][ObjSlice > 0] = ObjSlice[ObjSlice>0]

    print("Inserted object at location = (y,x)", centery, centerx)
    return True


WSize = 3072        #Size of the image
Rsf = 6             #Factor to downsize the final image with
NObj = 360          #Number of cylinders for each instance
Ninstances = 100    #Number of instances to create

for inst in range(0, Ninstances):

    World = np.zeros((NObj,WSize,WSize))
    WorldSum = np.zeros((WSize,WSize))

    np.random.seed(inst)
    random.seed(inst)

    Obj = 0
    while Obj < NObj:
        print(Obj)

        #l:     length of the tube
        #a:     radius of the tube
        #ang:  tilting (in degrees)
        l = rd.randint(46,468)
        a = rd.randint(15,46)
        ang = rd.randint(0,180)

        maxHalfSize = np.amax([a,a+int((l+1)/2)])
        maxHalfSizeTube = int((l+1)/2)
        Frame = np.zeros((2*maxHalfSize+1,2*maxHalfSize+1))

        yy,xx = np.mgrid[-maxHalfSize:maxHalfSize+1,-maxHalfSize:maxHalfSize+1]

        #Make cylinder
        Dist = yy**2

        Frame[Dist < a**2] = 1

        Frame[xx > maxHalfSizeTube] = 0
        Frame[xx < -maxHalfSizeTube] = 0

        #Make cones on top
        Dist1 = (yy**2+(xx-maxHalfSizeTube)**2)
        Frame[Dist1 < a**2] = 1
        Dist2 = (yy**2+(xx+maxHalfSizeTube)**2)
        Frame[Dist2 < a**2] = 1

        #Rotate according to given angles
        Frame = scipy.ndimage.rotate(Frame, ang, axes=(0,1), reshape = False)

        Frame[Frame >= 0.5] = 1
        Frame[Frame <= 0.5] = 0

        #Select random location
        py = rd.randint(maxHalfSize,WSize-maxHalfSize)
        px = rd.randint(maxHalfSize,WSize-maxHalfSize)
        
        #Try to insert the cylinder without overlapping other objects
        Inserted = insertObject(Frame, py, px, maxHalfSize, 1, WorldSum, WSize, True)
        if(Inserted == True):
            insertObject(Frame, py, px, maxHalfSize, 1, World[Obj,:,:], WSize, True)
            Obj = Obj+1

    #Resizing the data by factor Rsf
    RsS = int(World.shape[-1]/Rsf)
    Resize = np.zeros((World.shape[0],RsS,RsS))
    for resi in range(Rsf):
        for resj in range(Rsf):
            Resize += World[:,resi::Rsf,resj::Rsf]
    Result = (Resize/(Rsf*Rsf)).astype(np.float32)

    #Save the result
    tifffile.imsave('../../../data/RemoteSensingDatasets/RemoteSensingConfigurations/Instance' + str(inst).zfill(3) + '.tiff', Result.astype(np.float32))
