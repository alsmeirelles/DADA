#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import numpy as np
import random

from Utils import CacheManager

def _split_origins(config,x_data,t_idx):
    """
    Separates patches of a predefined number of WSIs to be used as test set
    """

    cache_m = CacheManager()
    if cache_m.checkFileExistence('testset.pik'):
        full_id,samples = cache_m.load('testset.pik')
        if not samples is None and config.info:
            print("[DataSetup] Using cached TEST SET. This is DANGEROUS. Use the metadata correspondent to the set.")
        return full_id,samples
            
    wsis = set()

    for k in x_data:
        wsis.add(k.getOrigin())

    #Defines slides to provide test set patches
    wsis = list(wsis)
    if config.wsilist is None:
        selected = set(random.choices(wsis,k=config.wsi_split))
    else:
        selected = set(config.wsilist)
    selected_idx = []

    if config.info:
        print("[DataSetup] WSIs selected to provide test patches:\n{}".format("\n".join(selected)))

    patch_count = {}
        
    for i in range(len(x_data)):
        w = x_data[i].getOrigin()
        if w in selected:
            patch_count.setdefault(w,[])
            patch_count[w].append(i)

    if config.wsimax is None or len(config.wsimax) != len(config.wsilist):
        for w in patch_count:
            if config.info:
                print("[Datasetup] Using all {} patches from slide {}".format(len(patch_count[w]),w))
            selected_idx.extend(patch_count[w])
    else:
        for i in range(len(config.wsilist)):
            w = config.wsilist[i]
            pc = int(config.wsimax[i] * len(patch_count[w]))
            pc = min(pc,len(patch_count[w]))
            selected_idx.extend(patch_count[w][:pc])
            if config.info:
                print("[Datasetup] Using {} ({:.2f}%) patches from slide {}".format(pc,100*pc/len(patch_count[w]),w))
                
    
    t_idx = min(len(selected_idx),t_idx)
    samples = np.random.choice(selected_idx,t_idx,replace=False)
    full_id = np.asarray(selected_idx,dtype=np.int32)
    cache_m.dump((full_id,samples),'testset.pik')
        
    return full_id,samples
    
def split_test(config,ds):

    #Test set is extracted from the last items of the full DS or from a test dir and is not changed for the whole run
    fX,fY = ds.load_metadata()
    test_x = None
    test_y = None
    
    tsp = config.split[-1:][0]
    t_idx = 0
    if tsp > 1.0:
        t_idx = int(tsp)
    elif tsp > 0.0:
        t_idx = int(tsp * len(fX))
    else:
        t_idx = np.inf

    #Configuration option that limits test set size
    t_idx = min(config.pred_size,t_idx) if config.pred_size > 0 else t_idx

    if config.testdir is None or not os.path.isdir(config.testdir):
        if config.wsi_split > 0 or not config.wsilist is None:
            full_id,samples = _split_origins(config,fX,t_idx)
            test_x = fX[samples]
            test_y = fY[samples]
            X = np.delete(fX,full_id)
            Y = np.delete(fY,full_id)
        else:
            test_x = fX[- t_idx:]
            test_y = fY[- t_idx:]
            X,Y = fX[:-t_idx],fY[:-t_idx]
        ds.check_paths(test_x,config.predst)
    else:
        x_test,y_test = ds.run_dir(config.testdir)
        t_idx = min(len(x_test),t_idx)
        samples = np.random.choice(len(x_test),t_idx,replace=False)
        test_x = [x_test[s] for s in samples]
        test_y = [y_test[s] for s in samples]
        del(x_test)
        del(y_test)
        del(samples)
        X,Y = fX,fY
        
    return test_x,test_y,X,Y

def csregen(superp,pool_size,generator_params,kwargs):
    """
    Regenerates the pool extracting pool_size elements from superpool.
    
    Uses CoreSet to select the samples that will be in the pool.

    Uses KMedian to select initial members, based on a limited subsample of the superpool 

    Arguments:
    - superp <tuple>: (superpool X, superpool Y)
    - pool_size <int>: size of the returned set
    - generator_params <dict>: parameters to be passed to a ThreadedGenerator

    Optional arguments:
    - space <int>: sets the observation space as space*pool_size. Defaul is 3.
    - clusters <int>: clusters space in this many groups
    """
    from sklearn.cluster import KMeans,MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances_argmin_min
    from datetime import timedelta
    import time
    from Trainers import ThreadedGenerator
    from AL.Common import extract_feature_from_function,load_model_weights
    from AL import cs_select_batch

    
    #Checks params
    model = None
    tmodels = None
    if not 'model' in kwargs or not 'emodels' in kwargs:
        print('[csregen] Generic model or trained models unavailable, falling back to random selection.')
        return np.random.choice(superp.shape[0],pool_size,replace=False)
    else:
        model = kwargs['model']
        tmodels = kwargs['emodels']

    sw_thread = kwargs.get('sw_thread',None)
    config = None
    if 'config' in kwargs:
        config = kwargs['config']
        clusters = 3*config.clusters
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
    else:
        cpu_count = 1
        gpu_count = 0
        clusters = 20
    
    #First - select space from superpool and setup generator
    if pool_size <= 1000:
        sp_size = kwargs.get('space',10) * pool_size
    else:
        sp_size = kwargs.get('space',5) * pool_size
    space_idx = np.random.choice(superp[0].shape[0],sp_size,replace=False)
    space = (superp[0][space_idx],superp[1][space_idx])
    generator_params['dps'] = space

    generator = ThreadedGenerator(**generator_params)
    
    #Second - extract features from space, apply PCA
    stime = time.time()
    if hasattr(model,'build_extractor'):
        pred_model = model.build_extractor(model=tmodels,parallel=gpu_count>1,sw_thread=sw_thread,new=True)
    else:
        if config.info:
            print("[DataSetup] Model is not prepared to produce features. No feature extractor")
        return None

    if model.is_ensemble():
        generator.set_input_n(config.emodels)
            
    #Extract features for all images in the pool
    #Some model variables should be reinitialized
    for m in tmodels:
        model.register_ensemble(m)
        load_model_weights(config,model,tmodels[m],sw_thread)
        
    features = extract_feature_from_function(pred_model,generator)

    del(pred_model)
    del(generator)
    
    if config.pca > 0:
        if config.info:
            print("[csregen] Starting PCA decomposition ({} features)...".format(config.pca))

        pca = PCA(n_components = config.pca)
        features = pca.fit_transform(features)
        
    #Third - Runs KMeans and divides feature space in k clusters (same as given by config - default 20)
    if pool_size < 10000:
        km = KMeans(n_clusters = clusters, init='k-means++',n_jobs=max(int(cpu_count/2),1)).fit(features)
    else:
        km = MiniBatchKMeans(n_clusters = clusters, init='k-means++',batch_size=500).fit(features)

    centers, _ = pairwise_distances_argmin_min(km.cluster_centers_, features)

    if config.verbose > 0:
        print("[csregen] Selected centers cluster #: {}".format(km.labels_[centers]))
    
    #Fourth - CoreSet extracts pool_size samples from space (pass space features and cluster center features - already selected)
    mask = np.ones(features.shape[0],dtype=bool)
    mask[centers] = False
    acquired = cs_select_batch(features[centers],features[mask],pool_size,cluster_centers=centers)
    del(features)
    
    if config.verbose > 0:
        etime = time.time()
        td = timedelta(seconds=(etime-stime))
        print("Pool setup step took: {}".format(td))
        
    #Fifth - return the index for selected samples, relative to superpool
    return space_idx[acquired]
