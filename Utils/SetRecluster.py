#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os,sys
import copy
import shutil
import math
import argparse
import pickle
import importlib
from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn.decomposition import PCA

from AL.Common import load_model_weights
from Utils import CacheManager

def load_modules(config):
    net_name = config.network
    if net_name is None or net_name == '':
        print("A network should be specified")
        return None

    ds = None
    if config.data:
        dsm = importlib.import_module('Datasources',config.data)
        ds = getattr(dsm,config.data)(config.predst,config.keepimg,config)
    else:
        ds = CellRep(config.predst,config.keepimg,config)

    net_module = importlib.import_module('Models',net_name)
    net_model = getattr(net_module,net_name)(config,ds)

    return net_model,ds

def restore_last_train(logdir,acq):
    """
    Restore the last training set used in a previous experiment
    """
    
    files = filter(lambda f:f.startswith('al-metadata'),os.listdir(logdir))
    metadata = {}
    for f in files:
        ac_id = int(f.split('.')[0].split('-')[3][1:])
        metadata[ac_id] = os.path.join(config.logdir,f)
    if acq > 0 and acq in metadata:
        last = acq
    else:
        last = max(metadata.keys())
    name = os.path.basename(metadata[last]).split('.')[0].split('-')[2]
    with open(metadata[last],'rb') as fd:
        train,_,_ = pickle.load(fd)

    return train[0],train[1],name,last

def run_clustering(config,data,net_model,nclasses):
    from Trainers import ThreadedGenerator
    from keras.preprocessing.image import ImageDataGenerator

    if hasattr(net_model,'build_extractor'):
        single_m,parallel_m = net_model.build_extractor(training=False,feature=True,parallel=False)
    else:
        if config.info:
            print("Model is not prepared to produce features. No feature extractor")
        return None    

    train_prep = ImageDataGenerator(
                samplewise_center=config.batch_norm,
                samplewise_std_normalization=config.batch_norm)    
    generator = ThreadedGenerator(dps=data,
                                      classes=nclasses,
                                      dim=config.tdim,
                                      batch_size=config.batch_size,
                                      image_generator=train_prep,
                                      extra_aug=False,
                                      shuffle=False,
                                      verbose=config.verbose)

    if not net_model.is_ensemble():
        pred_model = load_model_weights(config,model,single_m,parallel_m)
    else:
        generator.set_input_n(config.emodels)
        if not parallel_m is None:
            pred_model = parallel_m
        else:
            pred_model = single_m
            
    #Extract features for all images in the pool
    if config.info:
        print("Starting feature extraction ({} batches)...".format(len(generator)))
    features = pred_model.predict_generator(generator,
                                            workers=4*config.cpu_count,
                                            max_queue_size=100*config.gpu_count,
                                            verbose=0)

    features = features.reshape(features.shape[0],np.prod(features.shape[1:]))

    if config.info:
        print("Feature vector shape: {}".format(features.shape))
            
    if config.pca > 0:
        if config.info:
            print("Starting PCA decomposition ({} features)...".format(config.pca))

        pca = PCA(n_components = config.pca)
        features = pca.fit_transform(features)

    if config.bandwidth == 0:
        bw = estimate_bandwidth(features, quantile=0.3)
        print("Estimated bandwidth: {}".format(bw))
    else:
        bw = config.bandwidth
    ms = MeanShift(bandwidth = bw, cluster_all=True,n_jobs=max(int(config.cpu_count/2),1)).fit(features)

    unique,count = np.unique(ms.labels_,return_counts=True)
    print("Number of clusters: {}".format(unique.shape[0]))
    cl = list(zip(unique,count))
    cl.sort(key=lambda k:k[1],reverse=True)
    count = len(data[0])
    print("Cluster sizes:")
    print("\n".join(["Cluster # {0}: {1} items ({2:2.2f}% of total)" .format(key[0],key[1],100*(key[1]/count)) for key in cl]))

    #Cumulative patch sum
    for k in range(1,len(cl)+1):
        first = sum([j[1] for j in cl[:k]])
        print("Patches grouped in {} denser clusters: {} ({:2.2f} % of total)".format(k,first,100*first/count))
    
if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    parser.add_argument('--rec', dest='rec', action='store_true', 
        help='Recluster training data with MeanShift clustering.', default=False)
    parser.add_argument('-strategy',dest='strategy',type=str,
       help='Which strategy to use: ALTrainer, EnsembleTrainer, etc.',default='ALTrainer')
    parser.add_argument('-net',dest='network',type=str,default='',help='Network name which should be trained.\n \
    Check documentation for available models.')    
    parser.add_argument('-tdim', dest='tdim', nargs='+', type=int, 
        help='Tile width and heigth, optionally inform the number of channels (Use: 200 200 for SVS 50 um).', 
        default=[240,240], metavar=('Width', 'Height'))
    parser.add_argument('-predst', dest='predst', type=str,default='tiles', 
        help='Output tiles go to this directory')
    parser.add_argument('-b', dest='batch_size', type=int, 
        help='Batch size (Default: 8).', default=8)
    parser.add_argument('-emodels', dest='emodels', type=int, 
        help='Number of ensemble submodels (Default: 3).', default=3)    
    parser.add_argument('-bw', dest='bandwidth', type=float, 
        help='Mean-shift bandwidth. Zero means use default estimator. (Default: 0).', default=0)
    parser.add_argument('-pca', dest='pca', type=int, 
        help='Apply PCA to extracted features before clustering (Default: 0 (not used)).',default=50)    
    parser.add_argument('-data',dest='data',type=str,help='Dataset name to train model.\n \
    Check documentation for available datasets.',default='')
    parser.add_argument('-lr', dest='learn_r', type=float, 
        help='Learning rate (Default: 0.00005).', default=0.00005)    
    parser.add_argument('-split', dest='split', nargs=3, type=float, 
        help='Split data in as much as 3 sets (Default: 80%% train, 10%% validation, 10%% test). If AL experiment, test set can be defined as integer.',
        default=(0.8, 0.1,0.1), metavar=('Train', 'Validation','Test'))
    parser.add_argument('-gpu', dest='gpu_count', type=int, 
        help='Number of GPUs available (Default: 0).', default=0)
    parser.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False, 
        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-k', action='store_true', dest='keepimg', default=False, 
        help='Keep loaded images in memory.')
    parser.add_argument('-tnorm', action='store_true', dest='batch_norm',
        help='Applies batch normalization during training.',default=False)
    parser.add_argument('-pb', action='store_true', dest='progressbar', default=False, 
        help='Print progress bars of processing execution.')    
    parser.add_argument('-logdir', dest='logdir', type=str,default='logs', 
        help='Keep logs of current execution instance in dir.')    
    parser.add_argument('-model_dir', dest='model_path',
        help='Save trained models in dir (Default: TrainedModels).',
        default='')
    parser.add_argument('-acq', dest='acq', type=int, 
        help='Use trainset from this acquisition (Default: -1, means last one).', default=-1)

    config, unparsed = parser.parse_known_args()

    config.model_path = config.logdir
    config.cache = config.logdir
    config.weights_path = config.logdir
    
    files = {
        'datatree.pik':os.path.join(config.cache,'{}-datatree.pik'.format(config.data)),
        'tcga.pik':os.path.join(config.cache,'tcga.pik'),
        'metadata.pik':os.path.join(config.cache,'{0}-metadata.pik'.format(config.data)),
        'sampled_metadata.pik':os.path.join(config.cache,'{0}-sampled_metadata.pik'.format(config.data)),
        'testset.pik':os.path.join(config.cache,'{0}-testset.pik'.format(config.data)),
        'initial_train.pik':os.path.join(config.cache,'{0}-inittrain.pik'.format(config.data)),
        'split_ratio.pik':os.path.join(config.cache,'{0}-split_ratio.pik'.format(config.data)),
        'clusters.pik':os.path.join(config.cache,'{0}-clusters.pik'.format(config.data)),
        'data_dims.pik':os.path.join(config.cache,'{0}-data_dims.pik'.format(config.data)),
        'tiles.pik':os.path.join(config.predst,'tiles.pik'),
        'test_pred.pik':os.path.join(config.logdir,'test_pred.pik'),
        'cae_model.h5':os.path.join(config.model_path,'cae_model.h5'),
        'vgg16_weights_notop.h5':os.path.join('PretrainedModels','vgg16_weights_notop.h5')}

    cache_m = CacheManager(locations=files)
    
    net_model,ds = load_modules(config)
    train_x,train_y,_,_ = restore_last_train(config.logdir,config.acq)
    run_clustering(config,(train_x,train_y),net_model,ds.nclasses)
