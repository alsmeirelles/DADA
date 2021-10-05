#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os
import random
import re
import concurrent.futures

#Local modules
from Datasources import GenericDatasource as gd
from Preprocessing import PImage
from Utils import CacheManager

class LDir(gd.GenericDS):
    """
    Class that parses labels from file names according to the following examples:
    TCGA-63-A5MW-01Z-00-DX1_42058_8792_422_89_299_0.png    
    TCGA-63-A5MW-01Z-00-DX1_50550_14686_507_148_299_1.png

    A _0.png in the end means nonlymphocite patch, _1.png is a lymphocite patch
    """
    _rex = r'(?P<tcga>TCGA)-(?P<tss>[\w]{2})-(?P<part>[\w]{4})-(?P<sample>[\d]{2}[A-Z]{0,1})-(?P<ddigit>[\w]{2})-(?P<plate>[\w]{2}[0-9]{0,1})_(?P<xcoord>[\d]+)_(?P<ycoord>[\d]+)_(?P<unk1>[\d]+)_(?P<unk2>[\d]+)_(?P<unk3>[\d]+)_(?P<label>[\d]{1}).png'
    
    def __init__(self,data_path,keepImg=False,config=None):
        """
        @param data_path <str>: path to directory where image patches are stored
        @param config <argparse>: configuration object
        @param keepImg <boolean>: keep image data in memory
        """
        super().__init__(data_path,keepImg,config,name='LDir')
        self.nclasses = 2
        self.rcomp = re.compile(self._rex)
        self._executor = None

    def _load_metadata_from_dir(self,d):
        """
        Create SegImages from a directory.
        d is a directory corresponding to a cancer type. Inside it there should be a directory for each WSI from where patches were extracted.
        """
        class_set = set()

        t_x,t_y = ([],[])
        wsi_list = os.listdir(d)

        #Start thread pool if not already started
        if self._executor is None:
            workers = round((self._config.cpu_count/3 + (self._config.cpu_count%3>0) +0.5))
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

        futures = []
        for w in wsi_list:
            t_path = os.path.join(d,w)
            if not os.path.isdir(t_path):
                continue
            
            futures.append(self._executor.submit(self._run_threaded,t_path,w))

        for i in range(len(futures)):
            # add point to x_batch and diagnoses to y
            r_tx,r_ty,r_class = futures[i].result()
            t_x.extend(r_tx)
            t_y.extend(r_ty)
            class_set.update(r_class)

        del(futures)
            
        if self._verbose > 1:
            print("On directory {2}:\n - Number of classes: {0};\n - Classes: {1}".format(len(class_set),class_set,os.path.basename(d)))
        
        return t_x,t_y

    def _run_threaded(self,t_path,w):
        patches = os.listdir(t_path)
        class_set = set()
        t_x,t_y = ([],[])
        
        for f in patches:
            m = self.rcomp.match(f)
            if m is None:
                if self._verbose > 1:
                    print('[LDir] File does not match pattern: {0}'.format(f))
                continue
            coord = (m.group('xcoord'),m.group('ycoord'))
            seg = PImage(os.path.join(t_path,f),keepImg=self._keep,origin=w,coord=coord,verbose=self._verbose)
            label = int(m.group('label'))
            if label < 1:
                label = 0
            t_x.append(seg)
            t_y.append(label)
            class_set.add(label)
            
        return(t_x,t_y,class_set)
