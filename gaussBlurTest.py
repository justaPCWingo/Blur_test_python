#! /usr/bin/python


'''Description:
   Script that uses a brute force method for performing a gaussian blur on the provided image.
   The blurring can either be carried out serially or in parallel (using the multiprocess module).
Usage:
   gaussBlurTest.py mode image [noshow]
   Where:
       mode    : The method of blurring. Options are:
               :    serial      Sequentially blur pixels in single process.
               :    multiproc   Blur pixel in a series of processes that can execute in parallel.
               :
       image   : The path to an image file.
               :
       noshow  : Optional argument; if present, image will not be displayed after blurring.
       '''

import sys
import numpy as np
from scipy import misc as msp
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from multiprocessing import sharedctypes,Pool,cpu_count


#5x5 blur kernel - must have odd, square dimensions for proper sampling
kernel= np.array([[0.01, 0.02, 0.04, 0.02, 0.01],
                  [0.02, 0.04, 0.08, 0.04, 0.02],
                  [0.04, 0.08, 0.16, 0.08, 0.04],
                  [0.02, 0.04, 0.08, 0.04, 0.02],
                  [0.01, 0.02, 0.04, 0.02, 0.01]])
'''Gaussian kernel used for blurring.'''

xLim=0
'''XLimit used for loops.'''
yLim=0
'''YLimit used for loops.'''

class PixJob:
    '''
    Container for individual attributes for each job.
    
    Attributes:
        x: Integer for pixel x-coordinate.
        y: Integer for pixel y-coordinate.
    '''
    
    def __init__(self,x,y):
        '''Simple Constructor for initializing all arguments.'''
        self.x=x
        self.y=y
        


#def blurPixel(inBuff, outBuff,posX,posY):
def blurPixel(pj):
    '''Blurs a single pixel using the kernel.
    
    Args:
        pj: The values to use to process a single pixel.
    '''
    
    #assume sampling region is fully in image
    xStep=(kernel.shape[0]-1)/2
    yStep=(kernel.shape[1]-1)/2
    
    #find kernel probe region
    xStart=pj.x-xStep
    xFinish=pj.x+xStep
    yStart=pj.y-yStep
    yFinish=pj.y+yStep
    
    #initialize kernal indexes
    kXStart=0
    kYStart=0
    
    #compensate for image boundaries
    while xStart<0:
        xStart+=1
        kXStart+=1
    while yStart<0:
        yStart+=1
        kYStart+=1
    while xFinish>=xLim:
        xFinish-=1
    while yFinish>=yLim:
        yFinish-=1
        
    #perform blur
    #assume outbuff is all zeros
    tot=0
    kx=kXStart
    for x in range(xStart,xFinish+1):
        ky=kYStart
        for y in range(yStart,yFinish+1):
            contrib=gInImage[x,y]*kernel[kx,ky]
            gOutImage[pj.x,pj.y]+=contrib
            tot+=kernel[kx,ky]
            ky+=1
        kx+=1

    #normalize (just in case the kernel was clipped by edge)
    gOutImage[pj.x,pj.y]/=tot
    
def serialBlur(workSet,inImage,outImage):
    '''Blur one pixel at at time, in order.
    Args:
        workSet: The list of working pixels to operate on.
        inImage: The image to retrieve pixel value from.
        outImage: The image to write new pixels to.
    '''
    
    initGlobals(inImage,outImage)
    
    for job in workSet:
        blurPixel(job)
    
def multiProcBlur(workSet,inImage,outImage):
    '''Blur pixels in parallel, using the multiprocess module.
    Args:
        workSet: The list of working pixels to operate on.
        inImage: The image to retrieve pixel value from.
        outImage: The image to write new pixels to.
    '''
    
    #1-2 processes per logical core (ie 2 per CPU core if hyperthreading) 
    #is a reasonable estimate for balance. Modify to test other counts.
    cpuCount=cpu_count()*2
    mPool=Pool(initializer=initGlobals,initargs=(inImage,outImage,),processes=cpuCount)
    
    mPool.map(blurPixel,workSet)
    
def npToShared(npa):
    '''Convert a numpy array to a shared ctype.
    
    Based on code found here:
    http://briansimulator.org/sharing-numpy-arrays-between-processes/
    
    Args:
        npa: The numpy array to convert.
        
    Returns:
        A tuple containing the ctype raw array, the shape of npa, and the 
        repackaged numpy array.
    '''
    size=npa.size
    shape=npa.shape
    npa.shape=size
    npa_ctypes=sharedctypes.RawArray('B',npa)
    npa=np.frombuffer(npa_ctypes,dtype=np.uint8,count=size)
    npa.shape=shape
    
    return npa_ctypes,shape,npa

def sharedToNp(npa_ctypes,shape):
    '''Convert a shared ctype to a numpy array.
    
    Based on code found here:
    http://briansimulator.org/sharing-numpy-arrays-between-processes/
    
    Args:
        npa_ctypes: The ctype array to convert to a numpy array.
        shape: The shape to apply to the numpy array.
    
    Returns:
        The newly converted numpy array.
    '''
    npa=np.ctypeslib.as_array(npa_ctypes)
    npa.shape=shape
    return npa
    
def buildWorkSet(shape):
    '''Build a list of tasks used to carry out pixel transformations.
    
    Args:
        shape: The shape which defines the dimensions of the pixel images.
        
    Returns:
        A list of PixJob objects.
    '''
    ws=[]
    for x in range(shape[0]):
        for y in range(shape[1]):
            ws.append(PixJob(x,y))
    return ws

def initGlobals(inImage,outImage):
    '''Initalize globals that are shared across pixel processes.
    
    Args:
        inImage: The input image to mark as global.
        outImage: The output image to mark as global.
    '''
    global gInImage
    global gOutImage
    gInImage=inImage
    gOutImage=outImage

####################################################

if __name__=="__main__":
    argv=sys.argv
    if argv.__len__()>2:
        mode=argv[1]
        inFile=argv[2]
        inFace=msp.imread(inFile)
        theType=inFace.dtype
        inCTypes,inShape,inImage=npToShared(inFace)
        inFace=None #for safety
        outCTypes,outShape,outImage=npToShared(np.full(inShape,0,dtype=theType))
        xLim=inShape[0]
        yLim=inShape[1]
        print("Image '"+inFile+"' loaded.")
        
        workSet=buildWorkSet(inShape)
        print("Workset built with "+str(workSet.__len__())+" jobs.")
        
        if mode=="multiproc":
            print("Processing as multiprocess...")
            timeStart=timer()
            multiProcBlur(workSet,inImage,outImage)
            timeEnd=timer()
        else: # mode=="serial":
            print("Processing as serial...")
            timeStart=timer()
            serialBlur(workSet,inImage,outImage)
            timeEnd=timer()
        
        #display blur
        totTime=str(timeEnd-timeStart)
        if argv.__len__()<=3 or argv[3]!="noshow":
            plt.subplot(1,2,1)
            plt.title("Total time: "+totTime+" s")
            plt.imshow(sharedToNp(inCTypes,inShape))
            plt.subplot(1,2,2)
            plt.imshow(sharedToNp(outCTypes,outShape))
            plt.show()
        else:
            print("Total conversion time: "+totTime+" seconds")
    else:
        print(__doc__)
    