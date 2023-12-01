#!/usr/bin/env python3
#-*- coding: utf-8

import cv2
import os
import numpy as np

from .SegImage import SegImage

class CVImage(SegImage):
    """
    Represents any image handled by OpenCV.
    """
    def __init__(self,path,keepImg=False,origin=None,coord=None,verbose=0,):
        """
        @param path <str>: path to image
        @param keepImg <bool>: keep image data in memory
        @param origin <str>: current image is originated from origin
        @param coord <tuple>: coordinates in original image
        """
        super().__init__(path,keepImg,verbose)
        self._coord = coord
        self._origin = origin

    def __str__(self):
        """
        String representation is (coord)-origin if exists, else, file name
        """
        if not (self._coord is None and self._origin is None):
            return "{0}-{1}".format(self._coord,self._origin)
        else:
            return os.path.basename(self.path)

    def __repr__(self):
        return self.__str__()
    
    def readImage(self,keepImg=False):
        
        data = None

        #Change seting if we are going to keep the image in memory now
        if keepImg:
            self.setKeepImg(keepImg)
            
        if self._data is None:
            if self._verbose > 1:
                print("Reading image: {0}".format(self._path))
            data = cv2.imread(self._path)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            h,w,c = data.shape
            self._dim = (w,h,c)
            
            if self._keep:
                self._data = data
                
        else:
            if self._verbose > 1:
                print("Data already loaded:\n - {0}".format(self._path))
            data = self._data
            
        return data
    
    def readImageRegion(self,x,y,dx,dy):
        data = None
        
        if self._data is None:
            data = self.readImage()
        else:
            data = self._data
            
        return data[y:(y+dy), x:(x+dx)]

    def setKeepImg(self,keep):
        """
        If image should not be held anymore, delete data
        """
        if keep is None:
            return
        
        if not keep:
            del self._data
            self._data = None

        self._keep = keep
        
    def getImgDim(self):
        """
        Implements abstract method of SegImage
        """
        h,w,c = 0,0,0

        if not self._dim is None:
            return self._dim
        elif not self._data is None:
            h,w,c = self._data.shape
        else:
            data = cv2.imread(self._path)
            h,w,c = data.shape
            
            if self._keep:
                self._data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        self._dim = (w,h,c)
        return self._dim
