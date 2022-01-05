import numpy as np
import sys
import os
import time
from PIL import Image
import argparse
import multiprocessing
import importlib
import imgaug as ia
from imgaug import augmenters as iaa

#Local imports
from import_module import import_parents

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)

if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Augment a given dataset with \
        specified operations.')

    parser.add_argument('-ds', dest='ds', type=str,default='WSI', 
        help='Path original patches (directory containing the dataset).')        
    parser.add_argument('-od', dest='out_dir', type=str, default='Patches', 
        help='Save extracted patches to this location.')
    parser.add_argument('-data',dest='data',type=str,default='CellRep',
        help='Dataset name to train model.\n Check documentation for available datasets.')
    parser.add_argument('-mag', dest='mag', type=int, 
        help='Number of new patches each original image will generate', default=5,required=False)    
