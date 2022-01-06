import numpy as np
import sys
import os
import time
from PIL import Image
import argparse
import importlib
import imgaug as ia
from imgaug import augmenters as iaa

#Local imports
from import_module import import_parents
from ParallelUtils import multiprocess_run

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)

def _run_multiprocess(data,dst,mag):
    """Generate derived patches and saves them to destination"""
    for p in data:
        dst_dir = os.path.join(dst,os.path.basename(os.path.dirname(p.getPath())))
        name = p.getImgName()
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        for k in range(mag):
            pass
        
    
def generate_augmentation(data,dst,mag):

    if not os.path.isdir(dst):
        os.mkdir(dst)
        
    X,Y = multiprocess_run(_run_multiprocess,(dst,mag),data,
                               pbar=True,
                               cpu_count=4,
                               output_dim=2,
                               step_size=int(len(data)/4),
                               txt_label='Acquisition')

    print("Done generating.\n - New patches: {}".format(len(X)))

if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Augment a given dataset with \
        specified operations.')

    parser.add_argument('-ds', dest='ds', type=str,default=None, required=True,
        help='Path to original patches (directory containing the dataset).')        
    parser.add_argument('-od', dest='out_dir', type=str, default=None, 
        help='Save extracted patches to this location.')
    parser.add_argument('-data',dest='data',type=str,default='AqSet',
        help='Dataset name to train model.\n Check documentation for available datasets.')
    parser.add_argument('-mag', dest='mag', type=int, 
        help='Number of new patches each original image will generate', default=5,required=False)    
    config, unparsed = parser.parse_known_args()

    dsm = importlib.import_module('Datasources',config.data)
    ds = getattr(dsm,config.data)(config.ds,False,config)

    files = {
        'datatree.pik':os.path.join(config.cache,'{}-datatree.pik'.format(config.data)),
        'tcga.pik':os.path.join(config.cache,'tcga.pik'),
        'metadata.pik':os.path.join(config.cache,'{0}-{1}-metadata.pik'.format(config.data,os.path.basename(config.ds))),
        'un_metadata.pik':os.path.join(config.cache,'{0}-{1}-un_metadata.pik'.format(config.data,os.path.basename(config.ds))),
        'sampled_metadata.pik':os.path.join(config.cache,'{0}-sampled_metadata.pik'.format(config.data)),
        'testset.pik':os.path.join(config.cache,'{0}-testset.pik'.format(config.data)),
        'initial_train.pik':os.path.join(config.cache,'{0}-inittrain.pik'.format(config.data))}

    cache = CacheManager(locations=files)
    
    metadata = ds.load_metadata()

    if config.out_dir is None:
        config.out_dir = config.ds
        
    generate_augmentation(metadata,config.out_dir,config.mag)
