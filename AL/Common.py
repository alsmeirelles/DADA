#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os

__doc__ = """
Utility functions for acquisition functions and independent functions
"""

def load_model_weights(config,genmodel,tmodel,sw_thread=None):

    """
    genmodel: GenericModel
    tmodel: tuple (single_model,parallel_model) or a Keras.Model instance
    """
    import time
    from datetime import timedelta
    
    npfile = False
    checkpath = None
    stime = None
    
    if not sw_thread is None:
        last_thread = None
        if isinstance(sw_thread,list):
            last_thread = sw_thread[-1]
        else:
            last_thread = sw_thread
        if config.ffeat is None and last_thread.is_alive():
            if config.info:
                print("[load_model_weights] Waiting for model weights to become available...")
            last_thread.join()
    
    if hasattr(genmodel,'get_npweights_cache'):
        checkpath = genmodel.get_npweights_cache(add_ext=True)
        spath = checkpath
        npfile = True
        
    if npfile and not os.path.isfile(checkpath):
        spath = genmodel.get_weights_cache()
        npfile = False

    if config.info:
        stime = time.time()
            
    #Model can be loaded from previous acquisition train or from a fixed final model
    if config.gpu_count > 1:
        if hasattr(genmodel,'get_npmgpu_weights_cache'):
            checkpath = genmodel.get_npmgpu_weights_cache(add_ext=True)
            ppath = checkpath
            npfile = True
        
        if npfile and not os.path.isfile(checkpath):
            ppath = genmodel.get_mgpu_weights_cache()
            npfile = False

        if isinstance(tmodel,tuple):
            pred_model = tmodel[1]
        else:
            pred_model = tmodel
        if not config.ffeat is None and os.path.isfile(config.ffeat):
            pred_model.load_weights(config.ffeat,by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(config.ffeat))
        elif npfile:
            pred_model.set_weights(np.load(ppath,allow_pickle=True))
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(ppath))
        elif os.path.isfile(ppath):
            pred_model.load_weights(ppath,by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(ppath))
        else:
            if self._config.info:
                print("No trained model or weights file found")
            pred_model = None
    else:
        if isinstance(tmodel,tuple):
            pred_model = tmodel[0]
        else:
            pred_model = tmodel
        if not config.ffeat is None and os.path.isfile(config.ffeat):
            pred_model.load_weights(config.ffeat,by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(config.ffeat))
        elif npfile:
            pred_model.set_weights(np.load(spath,allow_pickle=True))
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(spath))                
        elif os.path.isfile(spath):
            pred_model.load_weights(spath,by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(spath))
        else:
            if config.info:
                print("No trained model or weights file found")
            pred_model = None
        
    if config.info:
        etime = time.time()
        td = timedelta(seconds=(etime-stime))
        print("Weights loading took: {}".format(td))

    return pred_model

def random_sample(pred_model,generator,data_size,**kwargs):
    """
    Returns a random list of indexes from the given dataset
    """
    if 'config' in kwargs:
        k = kwargs['config'].acquire
    else:
        return None
    
    return np.random.choice(range(data_size),k,replace=False)

def oracle_sample(pred_model,generator,data_size,**kwargs):
    """
    Returns the indexes of images that the current network missclassified with high probability.
    """

    if 'config' in kwargs:
        acquire = kwargs['config'].acquire
        cpu_count = kwargs['config'].cpu_count
        gpu_count = kwargs['config'].gpu_count
    else:
        return None

    if kwargs['config'].info:
        print("Oracle prediction starting...")
        
    #Keep verbosity in 0 to gain speed
    proba = pred_model.predict_generator(generator,
                                             workers=4*cpu_count,
                                             max_queue_size=100*gpu_count,
                                             verbose=0)
            
    pred_classes = proba.argmax(axis=-1)    
    expected = generator.returnLabelsFromIndex()
    miss = np.where(pred_classes != expected)[0]
    miss_prob = np.zeros(shape=expected.shape)
    for k in range(miss.shape[0]):
        miss_prob[miss[k]] = proba[miss[k]][pred_classes[miss[k]]]

    x_pool_idx = np.argsort(miss_prob)[-acquire:]
    
    if kwargs['config'].verbose > 0:
        print('Misses ({}): {}'.format(miss.shape[0]/expected.shape[0],miss))
        print("Probabilities for selected items:\n {}".format(miss_prob[x_pool_idx]))
        print("Selected item's prediction/true label:\n Prediction: {}\n True label: {}".format(pred_classes[x_pool_idx],
                                                                                                   expected[x_pool_idx]))

    return x_pool_idx


def debug_acquisition(s_expected,s_probs,classes,cache_m,config,fidp):
    from Utils import PrintConfusionMatrix
    
    if config.verbose > 0:
        r_class = np.random.randint(0,classes)
        print("Selected item's probabilities for class {2} ({1}): {0}".format(s_probs[r_class],s_probs.shape,r_class))
        prob_mean = np.mean(np.mean(s_probs,axis=-1),axis=-1)
        print("\n".join(["Selected item's mean probabilities for class {}:{}".format(k,prob_mean[k]) for k in range(prob_mean.shape[0])]))
        
    s_pred_all = s_probs[:,:].argmax(axis=0)

    if config.verbose > 0:
        print("Votes: {}".format(s_pred_all))
    #s_pred holds the predictions for each item after a vote
    s_pred = np.asarray([np.bincount(s_pred_all[i]).argmax(axis=0) for i in range(0,s_pred_all.shape[0])])
    if config.verbose > 0:
        print("Classification after vote: {}".format(s_pred))
    PrintConfusionMatrix(s_pred,s_expected,classes,config,"Selected images (AL)")
    if config.save_var:
        cache_m.dump((s_expected,s_probs),fidp)


def extract_feature_from_function(function,generator):

    data_size = generator.returnDataSize()
    bsize = generator.batch_size
    stp = int(np.ceil(data_size / bsize))
    features = None
    for i in range(stp):
        start_idx = i*bsize
        inp = generator.next()[0]
        if not isinstance(inp,list):
            inp = [inp]
        ff = function(inp)[0] #Considering the model has a single output
        if features is None:
            features = np.zeros(tuple([data_size]+list(ff.shape[1:])),dtype=np.float32)
        features[start_idx:start_idx+bsize] = ff

    return features
