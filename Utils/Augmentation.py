import numpy as np
import sys
import os
import time
import argparse
import importlib
import imgaug as ia
from imgaug import augmenters as iaa
from import_module import import_parents

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)
    
#Local imports
from Preprocessing import PImage
from Utils.ParallelUtils import multiprocess_run
from Utils.CacheManager import CacheManager

def _run_multiprocess(data,dst,mag,dim):
    """Generate derived patches and saves them to destination"""
    
    aug = iaa.SomeOf(3,[iaa.AddToBrightness((-30,30)),
                                          iaa.AddToSaturation((-50,50)),
                                          iaa.LinearContrast((0.8,1.8)),
                                          iaa.AddToHue((-10, 10)),
                                          #iaa.Rotate((0,22.5)),
                                          iaa.Fliplr(),
                                          iaa.Flipud(),
                                          iaa.KeepSizeByResize(iaa.CenterCropToFixedSize(width=dim[0]-20,height=dim[1]-20))
                                            ])
    data = tuple(zip(*data))
    X,Y = ([],[])
    for pk in range(len(data[0])):
        p = data[0][pk]
        dst_dir = os.path.join(dst,os.path.basename(os.path.dirname(p.getPath())))
        name = p.getImgName()
        try:
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)
        except FileExistsError:
            continue
        patch = p.readImage(size=dim,toFloat=False)
        X.append(p)
        Y.append(data[1][pk])
        p.saveImg(dst=os.path.join(dst_dir,os.path.basename(p.getPath())),arr=patch)
        for k in range(mag):
            pa_data = aug(images=[patch])[0]
            pa_name = "{}-v{}_{}.png".format(p.getImgName()[:-2],k,p.getImgName()[-1:])
            pa = PImage(os.path.join(dst_dir,pa_name),pa_data,keepImg=False,origin=p.getOrigin(),coord=p.getCoord())
            pa.saveImg()
            X.append(pa)
            Y.append(data[1][pk])

    return X,Y
    
def generate_augmentation(data,dst,mag,dim,cpu):

    if not os.path.isdir(dst):
        os.mkdir(dst)

    z1 = data[0]
    z2 = data[1]
    z = list(zip(z1,z2))
    X,Y = multiprocess_run(_run_multiprocess,(dst,mag,dim),z,
                               pbar=True,
                               cpu_count=cpu,
                               output_dim=2,
                               step_size=int(len(data[0])/cpu),
                               txt_label='augmentations')

    print("Done generating.\n - Total patches: {}".format(len(X)))

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
    parser.add_argument('-sample', dest='sample', type=int, 
        help='Get a sample of the dataset to be augmented', default=1.0,required=False)
    parser.add_argument('-tdim', dest='tdim', nargs='+', type=int, 
        help='Patch width and heigth, optionally inform the number of channels.', 
        default=None, metavar=('Width', 'Height'))
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False, 
        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-logdir', dest='logdir', type=str,default='logs', 
        help='Keep logs of current execution instance in dir.')
    parser.add_argument('-pb', action='store_true', dest='progressbar', default=True, 
        help='Print progress bars of processing execution.')
    parser.add_argument('-cache', dest='cache', type=str,default='cache', 
        help='Keeps caches in this directory',required=False)
    parser.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)

    config, unparsed = parser.parse_known_args()

    config.spool = 0
    config.pred_size = 0
    config.split = (1.0,0.0,0.0)
    config.cpu_count = 4

    files = {
        'datatree.pik':os.path.join(config.cache,'{}-datatree.pik'.format(config.data)),
        'tcga.pik':os.path.join(config.cache,'tcga.pik'),
        'metadata.pik':os.path.join(config.cache,'{0}-{1}-metadata.pik'.format(config.data,os.path.basename(config.ds))),
        'un_metadata.pik':os.path.join(config.cache,'{0}-{1}-un_metadata.pik'.format(config.data,os.path.basename(config.ds))),
        'sampled_metadata.pik':os.path.join(config.cache,'{0}-sampled_metadata.pik'.format(config.data)),
        'testset.pik':os.path.join(config.cache,'{0}-testset.pik'.format(config.data)),
        'split_ratio.pik':os.path.join(config.cache,'{0}-split_ratio.pik'.format(config.data))}

    cache = CacheManager(locations=files)

    dsm = importlib.import_module('Datasources',config.data)
    ds = getattr(dsm,config.data)(config.ds,False,config)
    
    metadata = ds.load_metadata()
    sampled = ds.sample_metadata(config.sample)

    if config.out_dir is None:
        config.out_dir = config.ds
        
    generate_augmentation(sampled[:2],config.out_dir,config.mag,config.tdim,config.cpu_count)
