#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import imghdr
import copy
from Preprocessing import PImage

class ImageSource(object):
    """
    Used for datasets that don't have a predefined structure.
    """
    def __init__(self,data,path,img_type,verbose=0):
        """
        @param path <str>: Path do data directory
        @param img_type <list>: list of supported image types
        """
        self.img_type = img_type
        self.verbose = verbose
        
        if not data is None and isinstance(data,tuple) and len(data[0]) > 0 and len(data[1]) > 0:
            self._data = data[0]
            self._labels = data[1]
            self.path = path
        elif isinstance(path,str) and os.path.isdir(path):
            self.path = path
            self._labels = None
            self._data = None
            self._run()
        else:
            print("[ImageSource] No data to load and no path to process")
            sys.exit(1)
            
    def _run(self):
        """
        Reads dir and builds the data dir with information about images found
        """
        def _checkType(impath):
            if not self.img_type is None:
                fields = os.path.basename(impath).split('.')

                #Even if file has no extension, check to see if its an image of given type. This may be slow!!
                if len(fields) == 2 and fields[1] in self.img_type:
                    return impath
                
                elif imghdr.what(impath) in self.img_type:
                    return impath
                
                return None
            else:
                return impath

        self._data = []
        self._labels = []
        for d in os.listdir(self.path):
            subdir = os.path.join(self.path,d)
            if os.path.isdir(subdir):
                if os.path.isfile(os.path.join(subdir,'label.txt')):
                    self._labels.append(os.path.join(subdir,'label.txt'))
                full_p = [os.path.join(subdir,img) for img in os.listdir(subdir)]
                self._data.extend(list(filter(_checkType,full_p)))

    def getData(self):

        return copy.copy(self._data)
    
    def getImgList(self):
        """
        Returns a list of GenericImage objects (ones tractable by OpenCV).
        """
        return [PImage(p) for p in self._data]

    def getLabelsList(self):
        """
        Returns a copy of the labels list
        """
        return copy.copy(self._labels)

