#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os,sys
import argparse
import importlib
import math
from scipy.spatial import distance

from keras.preprocessing.image import ImageDataGenerator

#Local imports
from import_module import import_parents

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)
    
from AL.Common import load_model_weights
from Trainers import ThreadedGenerator
from Trainers.GenericTrainer import Trainer
from Utils.CacheManager import CacheManager

def load_metadata(config,ds):
    #Test set is extracted from the last items of the full DS or from a test dir and is not changed for the whole run
    fX,fY = ds.load_metadata()

    if config.train_set + config.val_set + config.un_set > len(fX):
        raise ValueError("Not enough patches to satisfy set sizes")
    
    #Define training data, as a random sample of all data or as the hole set
    train_idx = np.random.choice(len(fX),config.train_set,replace=False)
    tX = fX[train_idx]
    tY = fY[train_idx]
    fX = np.delete(fX,train_idx)
    fY = np.delete(fY,train_idx)
    
    #Define validation data
    vX,vY = None,None
    if config.val_set > 0:
        val_idx = np.random.choice(len(fX),config.val_set,replace=False)
        vX = fX[val_idx]
        vY = fY[val_idx]
        fX = np.delete(fX,val_idx)
        fY = np.delete(fY,val_idx)

    #Define the set to have uncertainties calculated uppon
    un_idx = np.random.choice(len(fX),config.un_set,replace=False)
    uX = fX[un_idx]
    uY = fY[un_idx]
    del fX,fY

    return ((tX,tY),(vX,vY),(uX,uY))

def _make_un_generator(**kwargs):
    """
    Mandatory kwargs:
    - config
    - classes: number of classes
    - fix_fim: image dimensions
    - dps: data points
    """

    if not 'config' in kwargs:
        return None
    else:
        kwargs['config'].save_var = True

    pool_prep = ImageDataGenerator(
        samplewise_center=kwargs['config'].batch_norm,
        samplewise_std_normalization=kwargs['config'].batch_norm)
    
    generator_params = {
        'dps':kwargs['dps'],
        'classes':kwargs['classes'],
        'dim':kwargs['fix_dim'],
        'keep':False,#self._config.keepimg,
        'batch_size':kwargs['config'].gpu_count * kwargs['config'].batch_size if kwargs['config'].gpu_count > 0 else kwargs['config'].batch_size,
        'image_generator':pool_prep,
        'shuffle':False, #DO NOT SET TRUE!
        'verbose':kwargs['config'].verbose}

    generator = ThreadedGenerator(**generator_params)

    return generator

    
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Jensen-Shannon calculation over uncertainties \
        applied to CNNs.')

    ##Training options
    train_args = parser.add_argument_group('Training','Common network training options')
    arg_groups.append(train_args)

    train_args.add_argument('--train', action='store_true', dest='train', default=False, 
        help='Train models and calculate divergence.')    
    train_args.add_argument('-nets', dest='nets', type=str,default=None, nargs='+', required=True,
        help='Network names for divergence calculation.')
    train_args.add_argument('-tnet',dest='tnet',type=str,default=None,help='Target network for divergence calculation.\n \
    Check documentation for available models.')
    train_args.add_argument('-data',dest='data',type=str,help='Dataset name to train model.\n \
    Check documentation for available datasets.',default='CellRep')
    train_args.add_argument('-predst', dest='predst', type=str,default='tiles', 
        help='Tiles directory')    
    train_args.add_argument('-b', dest='batch_size', type=int, 
        help='Batch size (Default: 8).', default=8)
    train_args.add_argument('-lr', dest='learn_r', type=float, 
        help='Learning rate (Default: 0.00005).', default=0.00005)
    train_args.add_argument('-e', dest='epochs', type=int, 
        help='Number of epochs (Default: 1).', default=1)
    train_args.add_argument('-split', dest='split', nargs=3, type=float, 
        help='Split data in as much as 3 sets (Default: 80%% train, 10%% validation, 10%% test). If AL experiment, test set can be defined as integer.',
        default=(0.8, 0.1,0.1), metavar=('Train', 'Validation','Test'))
    train_args.add_argument('-f1', dest='f1period', type=int, 
        help='Execute F1 and ROC AUC calculations every X epochs (Default: 0).', default=0)
    train_args.add_argument('-train_set', dest='train_set', type=int, 
        help='Initial training set size (Default: 1000).', default=1000)
    train_args.add_argument('-val_set', dest='val_set', type=int, 
        help='Initial training set size (Default: 100).', default=100)
    train_args.add_argument('-un_set', dest='un_set', type=int,
        help='Initial training set size (Default: 1000).', default=1000)
    train_args.add_argument('-phis', dest='phis', type=int, nargs='+',
        help='Phi defines network architecture reduction. Values bigger than 1 reduce nets by 1/phi. Default = 1 (use original sizes).',default=1)
    train_args.add_argument('-tnphi', dest='tnphi', type=int, 
        help='Phi defines network architecture reduction. Values bigger than 1 reduce nets by 1/phi. Default = 1 (use original sizes).',default=1)
    train_args.add_argument('-tdim', dest='tdim', nargs='+', type=int, 
        help='Tile width and heigth, optionally inform the number of channels (Use: 200 200 for SVS 50 um).', 
        default=None, metavar=('Width', 'Height'))
    train_args.add_argument('-strategy',dest='strategy',type=str,
       help='Which strategy to use: ALTrainer, EnsembleTrainer, etc.',default='ALTrainer')
    train_args.add_argument('-wpath', dest='weights_path',
        help='Use weights file contained in path - usefull for sequential training (Default: None).',
        default='ModelWeights')
    train_args.add_argument('-aug', action='store_true', dest='augment',
        help='Applies data augmentation during training.',default=False)
    train_args.add_argument('-plw', action='store_true', dest='plw',
        help='Preload Imagenet weights after single model build.',default=False)
    train_args.add_argument('-lyf', dest='lyf', type=int, 
        help='Freeze this number of layers for training (Default=0).', default=0)
    train_args.add_argument('-tnorm', action='store_true', dest='batch_norm',
        help='Applies batch normalization during training.',default=False)
    train_args.add_argument('-tn', action='store_true', dest='new_net',
        help='Do not use older weights file.',default=False)    

    
    ##Divergence calculation options
    un_args = parser.add_argument_group('Divergence','Divergence calculation options')
    arg_groups.append(un_args)

    un_args.add_argument('--calc', action='store_true', dest='calc', default=False, 
        help='Calculate divergence using savend uncertainties.')    
    un_args.add_argument('-ac_function',dest='ac_function',type=str,
       help='Acquisition function. Check documentation for available functions.',default=None)
    un_args.add_argument('-un_function',dest='un_function',type=str,
       help='Uncertainty function to be used with KM. Check documentation for available functions.',default='bayesian_varratios')
    un_args.add_argument('-emodels', dest='emodels', type=int, 
        help='Number of ensemble submodels (Default: 3).', default=3)
    un_args.add_argument('-dropout_steps', dest='dropout_steps', type=int, 
        help='For Bayesian CNNs, sample the network this many times (Default: 100).', default=100)
    
    ##Load trained models
    load_args = parser.add_argument_group('Load','Load trained models from experiments')
    arg_groups.append(load_args)

    load_args.add_argument('-sd', dest='sdir', type=str,default=None, required=False,
        help='Experiment result path (should contain an slurm file).')    
    load_args.add_argument('-ids', dest='ids', nargs='+', type=int, 
        help='Experiment IDs to plot.', default=None,required=False)
    load_args.add_argument('-type', dest='tmode', type=str, nargs='+',
        help='Experiment type: \n \
        AL - General active learning experiment; \n \
        MN - MNIST dataset experiment.',
       choices=['AL','MN','DB','OR','KM','EN','TMP'],default='AL')    

    ##Hardware configurations
    hd_args = parser.add_argument_group('Hardware')
    arg_groups.append(hd_args)

    hd_args.add_argument('-gpu', dest='gpu_count', type=int, 
        help='Number of GPUs available (Default: 0).', default=0)
    hd_args.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)

    ##Runtime options
    parser.add_argument('-out', dest='bdir', type=str,default='', 
        help='Base dir to store all temporary data and general output',required=True)
    parser.add_argument('-cache', dest='cache', type=str,default='cache', 
        help='Keeps caches in this directory',required=False)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False, 
        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-logdir', dest='logdir', type=str,default='logs', 
        help='Keep logs of current execution instance in dir.')
    parser.add_argument('-mp', action='store_true', dest='multiprocess', default=False, 
        help='[TODO] Preprocess multiple images at a time (memory consuming - multiple processes).')
    parser.add_argument('-pb', action='store_true', dest='progressbar', default=False, 
        help='Print progress bars of processing execution.')
    parser.add_argument('-k', action='store_true', dest='keepimg', default=False, 
        help='Keep loaded images in memory.')
    parser.add_argument('-d', action='store_true', dest='delay_load', default=False, 
        help='Delay the loading of images to the latest moment possible (memory efficiency).')
    parser.add_argument('-db', action='store_true', dest='debug',
        help='Runs debugging procedures.',default=False)
    parser.add_argument('-model_dir', dest='model_path',
        help='Save trained models in dir (Default: TrainedModels).',
        default='TrainedModels')
    parser.add_argument('-save_dt', action='store_true', dest='save_dt',
        help='Save data (train, val, uncertainty).',default=False)    

    config, unparsed = parser.parse_known_args()

    files = {
        'datatree.pik':os.path.join(config.cache,'{}-datatree.pik'.format(config.data)),
        'tcga.pik':os.path.join(config.cache,'tcga.pik'),
        'metadata.pik':os.path.join(config.cache,'{0}-{1}-metadata.pik'.format(config.data,os.path.basename(config.predst))),
        'un_metadata.pik':os.path.join(config.cache,'{0}-{1}-un_metadata.pik'.format(config.data,os.path.basename(config.predst))),
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

    #Configuration options used by Acquisition functions but not relevant to JS convergence
    config.split = tuple(config.split)
    config.spool = 0
    config.pred_size = 0
    config.save_w = False
    config.acquire = 100
    config.clusters = 20
    config.ffeat = None
    config.recluster = 0

    #Uncertainty function
    function = None
    if not config.ac_function is None:
        acq = importlib.import_module('AL','AcquisitionFunctions')
        function = getattr(acq,config.ac_function)
    else:
        print("You should specify an acquisition function")
        sys.exit(Exitcodes.RUNTIME_ERROR)

    dsm = importlib.import_module('Datasources',config.data)
    ds = getattr(dsm,config.data)(config.predst,config.keepimg,config)        

    #Target net uncertainty file
    tnun = None
    un_file = 'al-uncertainty-{0}-r{1}.pik'
    tn_file = un_file.format(config.ac_function,"{}-PHI-{}".format(config.tnet,config.tnphi))
    cache_m.registerFile(os.path.join(config.logdir,tn_file),tn_file)
    
    #Train all models and generate uncertainties
    if config.train and not config.calc:
        def _common_train(model,trainer,data,ds,config,kwargs):
            trainer.train_x,trainer.train_y = data[0]
            trainer.val_x,trainer.val_y = data[1]
            tmodel,sw_thread,_ = trainer._target_net_train(model)
            un_generator = _make_un_generator(config=config,classes=ds.nclasses,fix_dim=model.check_input_shape(),dps=data[2])
            kwargs['sw_thread'] = sw_thread[0] if len(sw_thread) == 1 else sw_thread
            kwargs['config'] = config
            tmodel = tmodel[0] if len(tmodel) == 1 else tmodel
            pooled_idx = function(tmodel,un_generator,len(data[2][0]),**kwargs)

            return sw_thread,tmodel
        
        print("*************\nStart base model training\n*************\n")
        #Configurations and loading
        #config.new_net = True
        data = load_metadata(config,ds)
        if config.save_dt:
            cache_m.dump(data,'{0}-{1}-un_metadata.pik'.format(config.data,os.path.basename(config.predst)))
        ts = importlib.import_module('Trainers',config.strategy)
        trainer = getattr(ts,config.strategy)(config)
        kwargs = None
        #Begin network training
        for m in range(len(config.nets)):
            print("Current model: {} PHI={}".format(config.nets[m],config.phis[m]))
            config.phi = config.phis[m] #Set current PHI parameter
            model = trainer.load_modules(config.nets[m],ds)
            model.setName("{}-PHI-{}".format(model.getName(),config.phis[m]))
            on_file = un_file.format(config.ac_function,"{}-PHI-{}".format(config.nets[m],config.phis[m]))
            cache_m.registerFile(os.path.join(config.logdir,on_file),on_file)

            if cache_m.checkFileExistence(on_file) and not config.new_net:
                print("An uncertainty file already exists for {} PHI={}. Use -tn config option to regenerate".format(config.nets[m],config.phis[m]))
                continue
            
            if kwargs is None:
                kwargs = {}
            kwargs['model'] = model
            kwargs['acquisition'] = model.getName()

            sw_thread,tmodel = _common_train(model,trainer,data,ds,config,kwargs)
            
            if not sw_thread is None:
                tl = len(sw_thread)
                for k in range(tl):
                    if sw_thread[k].is_alive():
                        print("Waiting model {} weights' to become available ({}/{})...".format(config.nets[m],k,tl))
                        sw_thread[k].join()

        if not cache_m.checkFileExistence(tn_file) or config.new_net:
            print("\n *Start target network training*\n")
            model = trainer.load_modules(config.tnet,ds)
            model.setPhi(config.tnphi)
            model.setName("{}-PHI-{}".format(model.getName(),config.tnphi))
            kwargs['model'] = model
            kwargs['acquisition'] = model.getName()
            sw_thread = _common_train(model,trainer,data,ds,config,kwargs)
        else:
            print("An uncertainty file already exists for target {}. Use -tn config option to regenerate".format(config.tnet))

    #Calculate divergence
    if os.path.isfile(os.path.join(config.logdir,tn_file)):
        _,tnun = cache_m.load(tn_file)
        print("Target net uncertainties count: {}".format(tnun.shape[0]))
    else:
        print("Could not find uncertainties for target: {}".format(cache_m.fileLocation(tn_file)))
        sys.exit(1)

    for n in range(len(config.nets)):
        on_file = un_file.format(config.ac_function,"{}-PHI-{}".format(config.nets[n],config.phis[n]))
        if not cache_m.checkFileExistence(on_file):
            print("No uncertainty file generated for {} ({})".format(config.nets[n],cache_m.fileLocation(on_file)))
            continue
        _,onun = cache_m.load(on_file)

        print("Calculating divergence between {} and {}".format(config.nets[n],config.tnet))
        jsdist = distance.jensenshannon(onun,tnun,base=2)
        jsdiv = math.pow(jsdist,2)

        print("*******\n - JS Divergence: {}\n - JS Distance: {}\n*******".format(jsdiv,jsdist))
