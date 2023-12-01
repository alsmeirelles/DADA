#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os,sys
from tqdm import tqdm
from scipy.stats import mode

from .Common import load_model_weights

__doc__ = """
All acquisition functions should receive:
1 - numpy array of items
2 - numpy array of labels
3 - number of items to query
4 - keyword arguments specific for each function (if needed)

Returns: numpy array of element indexes
"""

def ensemble_varratios(pred_model,generator,data_size,**kwargs):
    """
    Calculation as defined in paper:
    Bayesian convolutional neural networks with Bernoulli approximate variational inference

    Function needs to extract the following configuration parameters:
    pred_model <dict>: keys (int) -> keras.Model model to use for predictions
    model <GenericModel>: instance responsible for model construction
    generator <keras.Sequence>: data generator for predictions
    data_size <int>: number of data samples
    mc_dp <int>: number of dropout iterations
    cpu_count <int>: number of cpu cores (used to define number of generator workers)
    gpu_count <int>: number of gpus available
    verbose <int>: verbosity level
    pbar <boolean>: user progress bars
    """
    from Utils import CacheManager
    cache_m = CacheManager()
    
    if 'config' in kwargs:
        config = kwargs['config']
        emodels = config.emodels
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
        verbose = config.verbose
        pbar = config.progressbar
        query = config.acquire
        save_var = config.save_var
    else:
        return None        

    r = kwargs.get('acquisition',0)

    if 'model' in kwargs:
        model = kwargs['model']
    else:
        print("[ensemble_varratios] GenericModel is needed by ensemble_varratios. Set model kw argument")
        return None

    if 'sw_thread' in kwargs:
        sw_thread = kwargs['sw_thread']
    else:
        sw_thread = None

    fidp = None
    if save_var:
        fid = 'al-uncertainty-{1}-r{0}.pik'.format(r,config.ac_function)
        cache_m.registerFile(os.path.join(config.logdir,fid),fid)
        if config.debug:
            fidp = 'al-probs-{1}-r{0}.pik'.format(r,config.ac_function)
            cache_m.registerFile(os.path.join(config.logdir,fidp),fidp)
        
    All_Dropout_Classes = np.zeros(shape=(data_size,1),dtype=np.float32)

    #If sw_thread was provided, we should check the availability of model weights
    if not sw_thread is None:
        for k in range(len(sw_thread)):
            if sw_thread[k].is_alive():
                print("Waiting ensemble model {} weights' to become available...".format(k))
                sw_thread[k].join()
                
    if pbar:
        l = tqdm(range(emodels), desc="Ensemble member predictions",position=0)
    else:
        if config.info:
            print("Starting Ensemble sampling...")
        l = range(emodels)

    #Keep probabilities for analysis
    all_probs = None
    if config.debug:
        all_probs = np.zeros(shape=(emodels,data_size,generator.classes),dtype=np.float32)

    #single,parallel = model.build(preload_w=False)
    for d in l:
        if not pbar and config.info:
            print("Step {0}/{1}".format(d+1,emodels))

        model.register_ensemble(d)
            
        curmodel = pred_model[d]
        curmodel = load_model_weights(config,model,curmodel,sw_thread)
        
        #Keep verbosity in 0 to gain speed 
        proba = curmodel.predict_generator(generator,
                                               workers=5*cpu_count,
                                               max_queue_size=100*gpu_count,
                                               verbose=0)

        if config.debug:
            all_probs[d] = proba
            
        dropout_classes = proba.argmax(axis=-1)    
        dropout_classes = dropout_classes.reshape(-1,1)
        All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)
        del(dropout_classes)

    if verbose > 1:
        print("Variation array {0}:".format(All_Dropout_Classes.shape))
        for i in np.random.choice(All_Dropout_Classes.shape[0],100,replace=False):
            print("Predictions for image ({0}): {1}".format(i,All_Dropout_Classes[i]))
    
    Variation = np.zeros(shape=(data_size),dtype=np.float32)

    for t in range(data_size):
        L = np.array([0])
        for d_iter in range(emodels):
            L = np.append(L, All_Dropout_Classes[t, d_iter+1])
        Predicted_Class, Mode = mode(L[1:])
        v = np.array(  [1 - Mode/float(emodels)])
        Variation[t] = v
    
    if verbose > 1:
        print("Variation {0}:".format(data_size))
        for i in np.random.choice(data_size,100,replace=False):
            print("Variation for image ({0}): {1}".format(i,Variation[i]))
        
    a_1d = Variation.flatten()
    x_pool_index = a_1d.argsort()[-query:][::-1]

    if config.debug:
        from .Common import debug_acquisition
        s_expected = generator.returnLabelsFromIndex(x_pool_index)
        #After transposition shape will be (classes,items,mc_dp)
        s_probs = all_probs[:emodels,x_pool_index].T
        debug_acquisition(s_expected,s_probs,generator.classes,cache_m,config,fidp)
            
    if save_var:
        cache_m.dump((x_pool_index,a_1d),fid)
        
    if verbose > 0:
        #print("Selected item indexes: {0}".format(x_pool_index))
        print("Selected item's variation: {0}".format(a_1d[x_pool_index]))
        print("Maximum variation in pool: {0}".format(a_1d.max()))
    
    return x_pool_index

def ensemble_bald(pred_model,generator,data_size,**kwargs):
    """
    Calculation as defined in paper:
    Bayesian convolutional neural networks with Bernoulli approximate variational inference

    Function needs to extract the following configuration parameters:
    pred_model <dict>: keys (int) -> keras.Model model to use for predictions
    model <GenericModel>: instance responsible for model construction
    generator <keras.Sequence>: data generator for predictions
    data_size <int>: number of data samples
    mc_dp <int>: number of dropout iterations
    cpu_count <int>: number of cpu cores (used to define number of generator workers)
    gpu_count <int>: number of gpus available
    verbose <int>: verbosity level
    pbar <boolean>: user progress bars
    """
    from Utils import CacheManager
    cache_m = CacheManager()
    
    if 'config' in kwargs:
        config = kwargs['config']
        emodels = config.emodels
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
        verbose = config.verbose
        pbar = config.progressbar
        query = config.acquire
        save_var = config.save_var
    else:
        return None

    if 'acquisition' in kwargs:
        r = kwargs['acquisition']
        
    if 'sw_thread' in kwargs:
        sw_thread = kwargs['sw_thread']
    else:
        sw_thread = None

    if 'model' in kwargs:
        model = kwargs['model']
    else:
        print("[ensemble_bald] GenericModel is needed by ensemble_bald. Set model kw argument")
        return None
    
    #If sw_thread was provided, we should check the availability of model weights
    if not sw_thread is None:
        for k in range(len(sw_thread)):
            if sw_thread[k].is_alive():
                print("Waiting ensemble model {} weights' to become available...".format(k))
                sw_thread[k].join()

    fidp = None
    if save_var:
        fid = 'al-uncertainty-{1}-r{0}.pik'.format(r,config.ac_function)
        cache_m.registerFile(os.path.join(config.logdir,fid),fid)
        if config.debug:
            fidp = 'al-probs-{1}-r{0}.pik'.format(r,config.ac_function)
            cache_m.registerFile(os.path.join(config.logdir,fidp),fidp)
            
    All_Entropy_Dropout = np.zeros(shape=data_size,dtype=np.float32)
    score_All = np.zeros(shape=(data_size, generator.classes),dtype=np.float32)

    #Keep probabilities for analysis
    all_probs = None
    if config.debug:
        all_probs = np.zeros(shape=(emodels,data_size,generator.classes),dtype=np.float32)
        
    if pbar:
        l = tqdm(range(emodels), desc="Ensemble member predictions",position=0)
    else:
        if config.info:
            print("Starting ensemble sampling...")
        l = range(emodels)

    #single,parallel = model.build(preload_w=False)
    
    for d in l:
        if not pbar and config.info:
            print("Step {0}/{1}".format(d+1,emodels))
            sys.stdout.flush()
            
        model.register_ensemble(d)

        curmodel = pred_model[d]
        curmodel = load_model_weights(config,model,curmodel,sw_thread)
        
        proba = curmodel.predict_generator(generator,
                                                workers=5*cpu_count,
                                                max_queue_size=100*gpu_count,
                                                verbose=0)
        if config.debug:
            all_probs[d] = proba
            
        #computing G_X
        score_All = np.add(score_All,proba,out=score_All)

        #computing F_X
        dropout_score_log = np.log2(proba)
        Entropy_Compute = - np.multiply(proba, dropout_score_log,out=dropout_score_log)
        Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)
        All_Entropy_Dropout = np.add(All_Entropy_Dropout, Entropy_Per_Dropout,out=All_Entropy_Dropout)
        
        del(proba)
        del(dropout_score_log)
        del(Entropy_Compute)
        del(curmodel)

    Avg_Pi = np.divide(score_All, emodels)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    G_X = Entropy_Average_Pi

    #Average entropy
    F_X = np.divide(All_Entropy_Dropout, emodels)

    #F_X = Average_Entropy
    U_X = G_X - F_X
    #Prevent nan values
    U_X[np.isnan(U_X)] = 0.0
    a_1d = U_X.flatten()
    x_pool_index = a_1d.argsort()[-query:][::-1]    

    #Release memory - there's a leak somewhere
    del(score_All)
    del(All_Entropy_Dropout)
    del(U_X)
    del(G_X)
    del(F_X)
    
    if save_var:
        cache_m.dump((x_pool_index,a_1d),fid)

    if config.debug:
        from .Common import debug_acquisition
        s_expected = generator.returnLabelsFromIndex(x_pool_index)
        #After transposition shape will be (classes,items,mc_dp)
        s_probs = all_probs[:emodels,x_pool_index].T
        debug_acquisition(s_expected,s_probs,generator.classes,cache_m,config,fidp)
        
    if verbose > 0:
        #print("Selected item indexes: {0}".format(x_pool_index))
        print("Selected item's average entropy: {0}".format(a_1d[x_pool_index]))
        print("Maximum entropy in pool: {0}".format(a_1d.max()))
    
    return x_pool_index
