#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
import numpy as np

#Preparing migration to TF 2.0
import tensorflow as tf
if tf.__version__ >= '1.14.0':
    tf = tf.compat.v1
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
from keras.layers import Average,Concatenate, Flatten
from keras import backend as K
from keras.models import Model

#Locals
from Utils import CacheManager
from Models.GenericModel import GenericModel
from AL.Common import load_model_weights

class GenericEnsemble(GenericModel):
    """
    Defines common model attributes and methods

    Some models may implement a feature extractor or an ensemble build. If they do, the following methods
    will be available:
    - build_extractor
    - build_ensemble
    """
    def __init__(self,config,ds,name=None,nclasses=2):
        super().__init__(config,ds,name=name,nclasses=nclasses)
        self.tmodels = None
        self.tmids = None

    def reset(self):
        if hasattr(self,'_s_ensemble'):
            del(self._s_ensemble)
        if hasattr(self,'_p_ensemble'):
            del(self._p_ensemble)

        self.tmodels = None
        self.tmids = None

    def get_npweights_cache(self,add_ext=False):
        """
        Returns path to model cache.

        @param add_ext <boolean>: add numpy file extension to file name.
        """
        fp = os.path.split(self.cache_m.fileLocation(self._weightsCache))
        if add_ext:
            return "{}.npy".format(os.path.join(fp[0],fp[1].split('.')[0]))
        else:
            return os.path.join(fp[0],fp[1].split('.')[0])

    def get_npmgpu_weights_cache(self,add_ext=False):
        """
        Returns path to model cache

        @param add_ext <boolean>: add numpy file extension to file name.
        """
        fp = os.path.split(self.cache_m.fileLocation(self._mgpu_weightsCache))
        if add_ext:
            return "{}.npy".format(os.path.join(fp[0],fp[1].split('.')[0]))
        else:
            return os.path.join(fp[0],fp[1].split('.')[0])

    def adjustWeightCache(self):
            
        self.cache_m.registerFile(os.path.join(self._config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(self._config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)
        self.cache_m.registerFile(os.path.join(self._config.model_path,self._modelCache),self._modelCache)
        
    def setName(self,name):
        """
        Override GenericModel's method to redefine cache file names
        """
        self.name = name
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
        self._modelCache = "{0}-model.h5".format(self.name)

        self.adjustWeightCache()
    
    def register_ensemble(self,m):
        self._model_n = m
        self._weightsCache = "{0}-EM{1}-weights.h5".format(self.name,m)
        self._mgpu_weightsCache = "{0}-EM{1}-mgpu-weights.h5".format(self.name,m)
        self._modelCache = "{0}-EM{1}-model.h5".format(self.name,m)        

        self.adjustWeightCache()
        
    def return_model_n(self):
        if hasattr(self,'_model_n'):
            return self._model_n
        else:
            return -1

    def build_extractor(self,**kwargs):
        """
        Builds a feature extractor from an already built model.

        Returns: Keras backend function

        Key word arguments:
        model: constructs the extractor from this model, build and ensemble if a dict of models is given
        preload_w: return model with weights already loaded? True -> Yes
        parallel: return parallel model (overrides gpu_count avaliation)? True -> Yes
        """
        #Weight loading for the feature extraction is done latter by requesting party
        model = None
        parallel = kwargs.get('parallel',False)
        
        if not 'model' in kwargs:
            print("[GenericEnsemble] A compiled model should be passed as the model argument")
            return None
        else:
            model = kwargs['model']

        if isinstance(model,dict):
            if self._config.info:
                print("Building ensemble from trained models: {}".format([hex(id(em)) for em in model.values()]))
            kwargs['emodels'] = model
            s_model,p_model = self.build_ensemble(**kwargs)
            if parallel and not p_model is None:
                model = p_model
            else:
                model = s_model

        x = None
        if self.is_ensemble():
            if parallel:
                 p_features = [model.get_layer('EM{}-{}'.format(e,self.name)) for e in range(self._config.emodels)]
                 layers = [imodel.get_layer('feature').output for imodel in p_features]
            else:
                 layers = [model.get_layer('EM{}-{}'.format(e,'feature')).output for e in range(self._config.emodels)]
            x = Concatenate()(layers)
        else:
            if parallel:
                feature_l  = model.get_layer(self.name).get_layer('feature')
                x = feature_l.output
            else:
                x = model.get_layer('feature').output

        if K.ndim(x) > 2:
            x = Flatten()(x)
        f = K.function(model.inputs, [x])

        return f
        
    def build_ensemble(self,**kwargs):
        """
        Builds an ensemble of M models.

        Weights are loaded here when new is set True because of the way ensembles should be built.

        Default build: avareges the output of the corresponding softmaxes

        @param npfile <boolean>: loads weights from numpy files
        @param rbuild <boolean>: build a new ensemble body or use the last built
        @param new <boolean>: create a new ensemble from emodels or use one already available
        @param emodels <dict>: dictionary  keys (int - model number) -> ensemble tower
        @param sw_thread <list/thread object>: wait on thread to load weights
        @param allocated_gpus <int>: define the number of GPUs to distribute the model
        @param load_weights <boolean>: load ensemble model weights during build
        """
        
        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']

        #Optional parameters
        npfile = kwargs.get('npfile',False)
        rbuild = kwargs.get('rbuild',False)
        new = kwargs.get('new',False)
        load_weights = kwargs.get('load_weights',False)        
        emodels = kwargs.get('emodels',None)
        sw_thread = kwargs.get('sw_thread',None)
        allocated_gpus = kwargs.get('allocated_gpus',self._config.gpu_count)
        feature = False
        
        inputs = None
        s_models = None
        p_models = None

        #Use cached models if needed
        if not emodels is None:
            ids = tuple([id(emodels[m]) for m in range(len(emodels))])
            if ids != self.tmids:
                self.tmids = ids
                self.tmodels = emodels
            else:
                new = False
                
        if emodels is None and not self.tmodels is None:
            emodels = self.tmodels
        
        if rbuild or (emodels is None and not (hasattr(self,'_s_ensemble') or hasattr(self,'_p_ensemble'))):
            if self._config.info and not new:
                print("[{}] No previous ensemble models stored, building new ones".format(self.name))
            s_models,p_models,inputs = self._build_ensemble_body(npfile,allocated_gpus,sw_thread)
            p_models = list(filter(lambda x: not x is None,p_models))
        elif not emodels is None and (new or not (hasattr(self,'_s_ensemble') and hasattr(self,'_p_ensemble'))):
            #Build from trained models
            inputs = []
            outputs = []
            for e in emodels:
                if load_weights:
                    self.register_ensemble(e)
                    emodels[e] = load_model_weights(self._config,self,emodels[e],sw_thread)
                for l in emodels[e].layers:
                    l._name = 'EM{}-{}'.format(e,l.name)
                inputs.extend(emodels[e].inputs)
                outputs.append(emodels[e].layers[-1].output)
            if allocated_gpus > 1:
                p_models = list(emodels.values())
                p_outputs = outputs
            else:
                s_models = list(emodels.values())
                s_outputs = outputs
        else:
            return (self._s_ensemble,self._p_ensemble)

        s_model,p_model = None,None            
        if not s_models is None and len(s_models) > 0:
            s_outputs = [out for s in s_models for out in s.outputs]
            x = Average()(s_outputs)
            s_model = Model(inputs = inputs, outputs=x)
            
        if not p_models is None and len(p_models) > 0:
            p_outputs = [out for p in p_models for out in p.outputs]
            x = Average()(p_outputs)
            p_model = Model(inputs=inputs,outputs=x)

        if hasattr(self,'_s_ensemble'):
            del(self._s_ensemble)
        if hasattr(self,'_p_ensemble'):
            del(self._p_ensemble)
        
        self._s_ensemble = s_model
        self._p_ensemble = p_model
        
        return s_model,p_model

    def _build_ensemble_body(self,npfile,allocated_gpus,sw_thread=None):
        s_models = []
        p_models = []
        inputs = []
        
        width,height,channels = self.check_input_shape()
        if K.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        for m in range(self._config.emodels):
            self.register_ensemble(m)
            model,inp = self._build_architecture(input_shape=input_shape,training=False,
                                                     preload=False,ensemble=True)

            inputs.append(inp)
            
            #Updates all layer names to avoid repeated name error
            for layer in model.layers:
                layer.name = 'EM{}-{}'.format(m,layer.name)
                
            single,parallel = self._configure_compile(model,allocated_gpus)
            
            single,parallel = self._load_weights(single,parallel,npfile,m,sw_thread)
            
            s_models.append(single)
            p_models.append(parallel)        

        return s_models,p_models,inputs    


    def _load_weights(self,single,parallel,npfile,m='',sw_thread=None):

        if not sw_thread is None:
            last_thread = None
            if isinstance(sw_thread,list):
                last_thread = sw_thread[-1]
            else:
                last_thread = sw_thread
            if last_thread.is_alive():
                if self._config.info:
                    print("[GenericEnsemble] Waiting for model weights to become available...")
                last_thread.join()
            
        if not parallel is None:
            #Updates all layer names to avoid repeated name error
            for layer in parallel.layers:
                layer.name = 'EM{}-{}'.format(m,layer.name)
            if npfile:
                parallel.set_weights(np.load(self.get_npmgpu_weights_cache(add_ext=True),allow_pickle=True))
                if self._config.info:
                    print("[{}] loaded ensemble weights: {}".format(self.name,self.get_npmgpu_weights_cache(add_ext=True)))
            elif os.path.isfile(self.get_mgpu_weights_cache()):                    
                parallel.load_weights(self.get_mgpu_weights_cache(),by_name=True)
                if self._config.info:
                    print("[{}] loaded ensemble weights: {}".format(self.name,self.get_mgpu_weights_cache()))
                    
            return single,parallel


        if npfile:
            single.set_weights(np.load(self.get_npweights_cache(add_ext=True),allow_pickle=True))
            if self._config.info:
                print("[{}] loaded ensemble weights: {}".format(self.name,self.get_npweights_cache(add_ext=True)))
        elif os.path.isfile(self.get_weights_cache()):
            single.load_weights(self.get_weights_cache(),by_name=True)
        else:
            if self._config.info:
                print("[{}] Could not load ensemble weights (model {})".format(self.name,m))
            single = None

        return single,parallel    
