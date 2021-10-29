#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import sys
import os
import argparse
import random
import shutil
import concurrent.futures
import tqdm
import numpy as np

#Local imports
from import_module import import_parents
from ParallelUtils import multiprocess_run

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)

from Preprocessing import white_ratio,PImage

def run_imgs(imgs,pw,workers,verbose=0):
    def _load_calc(pi,pw):
        img = PImage(pi,keepImg=False,verbose=verbose).readImage(toFloat=False)
        return white_ratio(img,pw)
        
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    futures = {}
    whites = np.zeros(shape=len(imgs),dtype=np.float32)
    for i in range(len(imgs)):
        futures[executor.submit(_load_calc,imgs[i],pw)] = i

    for f in concurrent.futures.as_completed(futures):
        whites[futures[f]] = f.result()

    return whites

def run_acquisition(acdir,sdir,pw,threads,verbose):

    acdir = acdir[0]
    base_dir = os.path.join(sdir,acdir)
    imgs = list(filter(lambda k: k.endswith('.png'),os.listdir(base_dir)))
    im_paths = [os.path.join(base_dir,k) for k in imgs]

    return (acdir,run_imgs(im_paths,pw,threads,verbose))
        
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Dataset manipulation.')
    
    parser.add_argument('-sd', dest='sdir', type=str,default=None, required=True,
        help='Source directory (path to patches).')
    parser.add_argument('-ac', dest='ac_n', nargs='+', type=str,
        help='Acquisitions to obtain images.', default=None, required=False)
    parser.add_argument('-pb', action='store_true', dest='pbar', default=False, 
        help='Print progress bars of processing execution.')
    parser.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)
    parser.add_argument('-threads', dest='th_count', type=int, 
        help='Number of threads/process (Default: 4).', default=4)
    parser.add_argument('-pw', dest='pw', type=int, 
        help='Whiteness region (Default: 50).', default=50)
    parser.add_argument('-np', dest='np', type=int, 
        help='Number of patches per acquisition (Default: 200).', default=200)    
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    
    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.sdir):
        print("No such directory: {}".format(config.sdir))
        sys.exit(1)

    dirs = list(filter(lambda k: os.path.isdir(os.path.join(config.sdir,k)),os.listdir(config.sdir)))

    if config.ac_n is None:
        ac_dirs = dirs
    else:
        ac_dirs = [d for d in config.ac_n if d in dirs]

    wt = multiprocess_run(run_acquisition,(config.sdir,config.pw,config.th_count,config.verbose),ac_dirs,config.cpu_count,
                              config.pbar,1,split_output=True,txt_label='Acquisition steps',verbose=config.verbose)

    data = None
    ordered_k = list(wt.keys())
    ordered_k.sort(key=lambda x:int(wt[x][0]))
    for k in ordered_k:
        print("Acquisition {}:\n - mean whiteness: {};\n - standard dev: {}".format(wt[k][0],np.mean(wt[k][1]),np.std(wt[k][1])))
        if data is None:
            data = np.zeros(shape=(len(wt),config.np))

        if data[k].shape[0] > wt[k][1].shape[0]:
            data[k,:wt[k][1].shape[0]] = wt[k][1]
        else:
            data[k] = wt[k][1]
        print("***************************")

    print("Mean patch whiteness in experiment: {}; STD Dev: {}".format(np.mean(data),np.std(data)))

        
