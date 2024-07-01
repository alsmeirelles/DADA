# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from sklearn.metrics import pairwise_distances

from .Common import extract_feature_from_function

def __flatten_X(X):
    shape = X.shape
    flat_X = X
    if len(shape) > 2:
      flat_X = np.reshape(X, (shape[0],np.product(shape[1:])))
    return flat_X


def __update_distances(cluster_centers,
                           min_distances,
                           pool_features,
                           sel_features,
                           already_selected,
                           metric='l2',
                           only_new=True,
                           reset_dist=False,
                           initialize = False):
    """
    Update min distances given cluster centers.

    Args:
      cluster_centers: indices of new cluster centers 
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in already_selected]
    if len(cluster_centers) > 0 or initialize:
      # Update min_distances for all examples given new cluster center.
      if only_new:
        x = pool_features[cluster_centers]
      else:
        x = sel_features
      dist = pairwise_distances(pool_features, x, metric=metric)

      if min_distances is None:
        min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        min_distances = np.minimum(min_distances, dist)

    return min_distances

def cs_select_batch(train_features, pool_features, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.
    
    Args:
      train_features: features of data already selected
      pool_features: features of data not selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """
        
    min_distances = None

    cluster_centers = kwargs.get('cluster_centers',[])
    
    if not train_features is None:
        if len(train_features.shape) != 2:
            print("[core-set] Train features should be a 2D array")
            return None
        min_distances = __update_distances(cluster_centers,
                                               min_distances=min_distances,
                                               pool_features = pool_features,
                                               sel_features = train_features,
                                               already_selected = [],
                                               only_new=False,
                                               reset_dist=True,
                                               initialize=True)


    new_batch = []

    for _ in range(N):
        if train_features is None:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(pool_features.shape[0]))
        else:
            ind = np.argmax(min_distances)
        # New examples should not be already selected since those points
        # should have min_distance of zero to a cluster center.
        if ind in new_batch:
            continue

        min_distances = __update_distances([ind],
                                               min_distances=min_distances,
                                               pool_features = pool_features,
                                               sel_features = train_features,
                                               already_selected = new_batch,
                                               only_new=True,
                                               reset_dist=False)
        new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.4f'
              % max(min_distances))

    return np.asarray(new_batch)


def core_set(bayesian_model,generator,data_size,**kwargs):
    """
    Cluster data in K clusters and returns their centroids

    Function needs to extract the following configuration parameters:
    model <None>: not used. Model used to extract features will be created here
    generator <keras.Sequence>: data generator for predictions
    data_size <int>: number of data samples
    config <argparse>: configuration object
    train_gen <Iterator>: batch data generator
    sw_threads <thread Object>: if a thread object is passed, you must wait its conclusion before loading weights
    """
    from sklearn.decomposition import PCA
    import importlib
    import copy
    import time
    from datetime import timedelta
    from Utils import CacheManager

    cache_m = CacheManager()
    
    if 'config' in kwargs:
        config = kwargs['config']
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
        verbose = config.verbose
        pbar = config.progressbar
        query = config.acquire
        clusters = config.clusters
    else:
        return None
    
    if 'acquisition' in kwargs:
        acq = kwargs['acquisition']
    else:
        acq = config.acquisition_steps
        
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        print("[core_set] GenericModel is needed by core_set. Set model kw argument")
        return None

    if 'train_gen' in kwargs:
        train_gen = kwargs['train_gen']
    else:
        print("[core_set] Training data is needed by core_set. Set train_gen kw argument")
        return None
    
    #Models that take to long to save weights might not have finished
    if 'sw_thread' in kwargs:
        last_thread = None
        if isinstance(kwargs['sw_thread'],list):
            last_thread = kwargs['sw_thread'][-1]
        else:
            last_thread = kwargs['sw_thread']
        if config.ffeat is None and last_thread.is_alive():
            if config.info:
                print("[core_set] Waiting for model weights to become available...")
            last_thread.join()
    elif config.info:
        print("[core_set] Weights thread not available...trying to load weights")

    if not model.is_ensemble() and not (os.path.isfile(model.get_weights_cache()) or not os.path.isfile(model.get_mgpu_weights_cache())):
        if config.info:
            print("[core_set] No trained model or weights file found (H5).")
        return None    

    #Run feature extraction and clustering
    if hasattr(model,'build_extractor'):
        pred_model = model.build_extractor(model=bayesian_model,training=False,feature=True,parallel=True)
    else:
        if config.info:
            print("[core_set] Model is not prepared to produce features. No feature extractor")
        return None

    if model.is_ensemble():
        generator.set_input_n(config.emodels)
        train_gen.set_input_n(config.emodels)

            
    #Extract features for all images in the pool
    if config.info:
        print("Starting feature extraction ({} batches)...".format(len(generator)))
    pool_features = extract_feature_from_function(pred_model,generator)

    train_features = extract_feature_from_function(pred_model,train_gen)

    del(pred_model)
    del(generator)
    del(train_gen)
    
    pool_features = pool_features.reshape(pool_features.shape[0],np.prod(pool_features.shape[1:]))
    train_features = train_features.reshape(train_features.shape[0],np.prod(train_features.shape[1:]))
    
    if config.info:
        print("Pool feature vector shape: {}".format(pool_features.shape))
        print("Train data feature vector shape: {}".format(train_features.shape))
        
    if config.pca > 0:
        if config.info:
            print("Starting PCA decomposition ({} features)...".format(config.pca))

        pca = PCA(n_components = config.pca)
        pool_features = pca.fit_transform(pool_features)
        train_features = pca.fit_transform(train_features)

    stime = None
    etime = None
    if config.verbose > 0:
        print("Done extraction...starting CoreSet")
        stime = time.time()

    acquired = cs_select_batch(train_features, pool_features, query)

    print("Acquired ({}): {}".format(acquired.shape,acquired))

    return acquired
