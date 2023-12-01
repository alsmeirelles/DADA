#!/usr/bin/env python3
#-*- coding: utf-8

import os
import numpy as np

from .SegImage import SegImage

class NPImage(SegImage):
    """
    Represents any image already stored as Numpy arrays.
    """

    def __init__(self,path,data=None,keepImg=False,origin=None,coord=None,verbose=0):
        """
        @param path <str>: path to image
        @param data <ndarray>: image data in a Numpy array
        @param keepImg <bool>: keep image data in memory
        @param origin <str>: current image is originated from origin: x_train, x_val or x_test
        @param coord <int>: coordinates in original image: index
        """
        super().__init__(path,keepImg,verbose)
        self._coord = coord
        self._origin = origin
        if not data is None and isinstance(data,np.ndarray):
            self._data = data

    def __str__(self):
        """
        String representation is (coord)-origin if exists, else, file name
        """
        if not (self._coord is None and self._origin is None):
            return "{0}-{1}".format(self._coord,self._origin)
        else:
            return "{0}-{1}".format(os.path.basename(self._path),self._coord)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # Hashes current dir and file name
        return hash((os.path.basename(self._path),self._origin,self._coord))

    def readImage(self,keepImg=None,size=None,verbose=None):

        if not self._data is None:
            return self._data

        data = None
        with np.load(self._path, allow_pickle=True) as f:
            if self._origin in f:
                data = f[self._origin][self._coord]

        if keepImg is None:
            keepImg = self._keep
        elif keepImg:
            #Change seting if we are going to keep the image in memory now
            self.setKeepImg(keepImg)
        if not verbose is None:
            self._verbose = verbose

        if self._keep:
            self._data = data

        return data

    def getImgDim(self):
        """
        Implements abstract method of SegImage
        """

        if not self._dim is None:
            return self._dim
        elif not self._data is None:
            self._dim = self._data.shape
        else:
            data = self.readImage()
            self._dim = data.shape
                
        return self._dim

    def readImageRegion(self,x,y,dx,dy):
        data = None
        
        if self._data is None:
            data = self.readImage()
        else:
            data = self._data
            
        return data[y:(y+dy), x:(x+dx)]

    def __getstate__(self):
          """
          Prepares for pickling.
          """
          state = self.__dict__.copy()
          if not self._keep:
            del state['_data']
            state['_data'] = None

          return state
