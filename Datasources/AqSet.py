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
    TCGA-63-A5MW-01Z-00-DX1_42058_8792_422_89_299_[0,1].png    
    UN-0-TCGA-63-A5MW-01Z-00-DX1_50550_14686_507_148_299_vN_1.png

    A _0.png in the end means nonlymphocite patch, _1.png is a lymphocite patch
    """
    _rex = r'(?P<un>UN-[\d]+)?-(?P<tcga>TCGA)-(?P<tss>[\w]{2})-(?P<part>[\w]{4})-(?P<sample>[\d]{2}[A-Z]{0,1})-(?P<ddigit>[\w]{2})-(?P<plate>[\w]{2}[0-9]{0,1})-(?P<xcoord>[\d]+)-(?P<ycoord>[\d]+)-(?P<ps1>[\d]+)-(?P<ps2>[\d]+)(?P<version>-v[\d]+)?_(?P<label>[\d]{1}).png'
    
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
        d is a directory corresponding to an acquisition iteration number. The patches acquired in each iteration are in d.
        """
        patches = os.listdir(d)
        t_x,t_y = ([],[])
        
        for f in patches:
            m = self.rcomp.match(f)
            if m is None:
                if self._verbose > 1:
                    print('[LDir] File does not match pattern: {0}'.format(f))
                continue
            w = "-".join([m.group(g) for g in ('tcga','tss','part','sample','ddigit','plate')])
            coord = (m.group('xcoord'),m.group('ycoord'))
            seg = PImage(os.path.join(t_path,f),keepImg=self._keep,origin=w,coord=coord,verbose=self._verbose)
            label = int(m.group('label'))
            if label < 1:
                label = 0
            t_x.append(seg)
            t_y.append(label)
            
        return (t_x,t_y)


    def _run_threaded(self,t_path,w):
