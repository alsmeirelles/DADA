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
    tf.logging.set_verbosity(tf.logging.ERROR)
    #tf.disable_v2_behavior()

#Network
from keras.models import Sequential,Model
from keras.layers import Input
from keras import optimizers
from keras.utils import multi_gpu_model
from keras import backend as K

#Locals
from Utils import CacheManager
from Models.GenericEnsemble import GenericEnsemble

class Inception(GenericEnsemble):
    """
    Implements abstract methods from GenericModel.
    Model is the same as in: https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py
    Addapted to provide a Bayesian model
    """
    def __init__(self,config,ds,name=None):
        super().__init__(config,ds,name=name)
        if name is None:
            self.name = "Inception"
        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
 
        self.cache_m = CacheManager()
        self.cache_m.registerFile(os.path.join(config.model_path,self._modelCache),self._modelCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)

        self.single = None
        self.parallel = None
        
    def get_model_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._modelCache)
    
    def get_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._weightsCache)

    def get_mgpu_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._mgpu_weightsCache)
        
    def build(self,**kwargs):
        """
        Optional params:
        @param pre_trained <boolean>: returned model should be pre-trained or not
        @param data_size <int>: size of the training dataset
        @param feature <boolean>: return features instead of softmax classification
        """
        model,parallel_model = self._build(**kwargs)
        
        self.single = model
        self.parallel = parallel_model
        
        return (model,parallel_model)
    
    def _build(self,**kwargs):

        width,height,channels = self._check_input_shape()

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']

        if 'training' in kwargs:
            training = kwargs['training']
        else:
            training = True
            
        if 'feature' in kwargs:
            feature = kwargs['feature']
        else:
            feature = False

        if 'preload_w' in kwargs:
            preload = kwargs['preload_w']
        else:
            preload = False

        if 'allocated_gpus' in kwargs and not kwargs['allocated_gpus'] is None:
            allocated_gpus = kwargs['allocated_gpus']
        else:
            allocated_gpus = self._config.gpu_count
            
        if K.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)
        
        model = self._build_architecture(input_shape,training,feature,preload)

        return self._configure_compile(model,allocated_gpus)

    def _configure_compile(self,model,allocated_gpus):
        """
        Configures, compiles, generates parallel model if needed

        @param model <Keras.Model>
        """
        #Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.registerFile(os.path.join(self._config.cache,lr_cache),lr_cache)
        l_rate = self._config.learn_r
        if os.path.isfile(self.cache_m.fileLocation(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))
        
        #opt = optimizers.SGD(lr=l_rate, decay=1.5e-4, momentum=0.9, nesterov=True)
        opt = optimizers.Adam(lr = l_rate)
        #opt = optimizers.Adadelta(lr=l_rate)

        #Return parallel model if multiple GPUs are available
        parallel_model = None
       
        if allocated_gpus > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
            parallel_model = multi_gpu_model(model,gpus=allocated_gpus)
            parallel_model.compile(loss='categorical_crossentropy',
                                       optimizer=opt,
                                       metrics=['accuracy'],
                                       #options=p_opt, 
                                       #run_metadata=p_mtd
                                       )
        else:
            model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'],
                #options=p_opt, 
                #run_metadata=p_mtd
                )

        return (model,parallel_model)        


    def _build_architecture(self,input_shape,training=None,feature=False,preload=True,ensemble=False):

        """
        Parameters:
        - training <boolean>: sets network to training mode, wich enables dropout if there are DP layers
        - feature <boolean>: build a feature extractor
        - preload <boolean>: preload Imagenet weights
        - ensemble <boolean>: builds an ensemble of networks from the Inception architecture

        OBS: self.is_ensemble() returns if the ensemble strategy is in use
        """
        from . import inception_resnet_v2

        kwargs = {'training':training,
                    'feature':feature,
                    'custom_top':False,
                    'preload':preload,
                    'batch_n':True if self._config.gpu_count <= 1 else False,
                    'use_dp': True } #False if self.is_ensemble() else True}

        inp = Input(shape=input_shape)
                
        inception_body = inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                                weights='imagenet',
                                                                input_tensor=inp,
                                                                input_shape=input_shape,
                                                                pooling='avg',
                                                                classes=self._ds.nclasses,
                                                                **kwargs)
        

        if ensemble:
            return (inception_body,inp)
        else:
            return inception_body
                                


