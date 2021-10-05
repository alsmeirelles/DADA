#!/usr/bin/env python3
#-*- coding: utf-8

import openslide
from .SegImage import SegImage

class SVSImage( SegImage ):
    """
    SVS images are not handled by OpenCV and need openslide to be handled
    """
    def __init__(self,path,keepImg=False,verbose=0):
        """
        @param path <str>: path to image
        @param keepImg <bool>: keep image data in memory (not recomended for SVS images as they are too big)
        """
        super().__init__(path,keepImg,verbose)
        self._oslide = None

    def __checkOpen(self):
        
        if self._oslide is None:
            self._oslide = openslide.OpenSlide(self._path)

    def __checkClose(self):

        if not self._keep and not self._oslide is None:
            self._oslide.close()
            self._oslide = None
            
    def readImage(self):
        """
        Returns a low resolution version of the hole image. Currently, returned image has size
        equal to the highest resolution adjusted by a scale factor of 64.
        """

        self.__checkOpen()
            
        data = self._oslide.get_thumbnail((self._oslide.dimensions[0]//64,self._oslide.dimensions[1]//64)).convert('RGB')
        #Convert data to numpy array
        data = np.array(data)
        
        if self._keep:
            self._data = data
        else:
            self._oslide.close()
            self._oslide = None
        
        return data
    
    def readImageRegion(self,x,y,dx,dy):
        data = None
        
        self.__checkOpen()

        data = self._oslide.read_region((x, y), 0, (dx, dy)).convert('RGB')
        data = np.array(data)

        self.__checkClose()
        
        return data
    
    def getImgDim(self):
        """
        Dimensions of the highest magnification level
        """
        self.__checkOpen()

        dim = self._oslide.dimensions

        self.__checkClose()
        
        return dim
    
    def setKeepImg(self,keep):
        """
        If we don't want to keep anything anymore, delete what may be stored.
        """
        
        if not keep:
            if not self._oslide is None:
                self._oslide.close()
            self._oslide = None
            self._data = None

        self._keep = keep
        
