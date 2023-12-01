#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

import numpy as np
import tensorflow as tf

#Keras stuff
#from keras.utils import multi_gpu_model
from keras import utils as keras_utils
from keras_contrib.layers import GroupNormalization
from keras import backend
from keras import layers
from keras import models
from keras import optimizers

#Locals
from Utils import CacheManager
from Models.GenericEnsemble import GenericEnsemble

TF_WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels.h5')
TF_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

class Xception(GenericEnsemble):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Note that the default input image size for this model is 299x299.

    Reference: Xception: Deep Learning with Depthwise Separable Convolutions (CVPR 2017)


    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    def __init__(self,config,ds,name=None,nclasses=2):
        super().__init__(config,ds,name=name,nclasses=nclasses)
        if name is None:
            self.name = "Xception"        

        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
        self.cache_m = CacheManager()
        self.cache_m.registerFile(os.path.join(config.model_path,self._modelCache),self._modelCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)


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

    def _build(self,width,height,channels,**kwargs):
        """
        Custom build process
        """
        training = kwargs.get('training',None)
        preload = kwargs.get('preload_w')
        lf = kwargs.get('layer_freeze')
        allocated_gpus = kwargs.get('allocated_gpus')
        
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        model = self._build_architecture(input_shape,training,preload,layer_freeze=lf)

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

    def _build_architecture(self,input_shape,training=None,preload=True,ensemble=False,**kwargs):
        """
        Parameters:
        - training <boolean>: sets network to training mode, wich enables dropout if there are DP layers
        - preload <boolean>: preload Imagenet weights
        - ensemble <boolean>: builds an ensemble of networks from the Inception architecture

        KWARGS:

        - weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        - use_dp: use Dropout
        - pooling: final pooling type ('avg','max')
        - include_top: include top layers in model
        OBS: self.is_ensemble() returns if the ensemble strategy is in use
        """
        weights = kwargs.get('weights',None)
        use_dp = kwargs.get('use_dp',True) #False if self.is_ensemble() else True
        pooling = kwargs.get('pooling','avg')
        include_top = kwargs.get('include_top',True)
        classes = self.nclasses
        batch_n = True if self._config.gpu_count <= 1 else False
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
        
        # Determine proper input shape
        if input_shape is None:
            input_shape = (299,299,3)

        if not (weights in {'imagenet', None} or os.path.exists(weights)):
            raise ValueError('The `weights` argument should be either '
                            '`None` (random initialization), `imagenet` '
                            '(pre-training on ImageNet), '
                            'or the path to the weights file to be loaded.')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                            ' as true, `classes` should be 1000')        
        
        filters = {32:self.rescale('width',32),
                    48:self.rescale('width',48),
                    64:self.rescale('width',64),
                    128:self.rescale('width',128),
                    160:self.rescale('width',160),
                    256:self.rescale('width',256),
                    512:self.rescale('width',512),
                    728:self.rescale('width',728),
                    1024:self.rescale('width',1024),
                    1536:self.rescale('width',1536),
                    2048:self.rescale('width',2048)}        
        
        inp = layers.Input(shape=input_shape)

        x = layers.Conv2D(filters.get(32,32), (3, 3),
                        strides=(2, 2),
                        use_bias=False,
                        name='block1_conv1')(inp)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = layers.Activation('relu', name='block1_conv1_act')(x)
        x = layers.Conv2D(filters.get(64,64), (3, 3), use_bias=False, name='block1_conv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = layers.Activation('relu', name='block1_conv2_act')(x)

        #Needed for bayesian version
        if use_dp or training:
            x = layers.Dropout(0.1)(x,training=training)        

        residual = layers.Conv2D(filters.get(128,128), (1, 1),
                                strides=(2, 2),
                                padding='same',
                                use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.SeparableConv2D(filters.get(128,128), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block2_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block2_sepconv2_act')(x)
        
        if use_dp or training:
            x = layers.Dropout(0.1)(x,training=training)
            
        x = layers.SeparableConv2D(filters.get(128,128), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block2_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(filters.get(256,256), (1, 1), strides=(2, 2),
                                padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block3_sepconv1_act')(x)
        x = layers.SeparableConv2D(filters.get(256,256), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block3_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block3_sepconv2_act')(x)
        if use_dp or training:
            x = layers.Dropout(0.1)(x,training=training)
            
        x = layers.SeparableConv2D(filters.get(256,256), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block3_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(filters.get(728,728), (1, 1),
                                strides=(2, 2),
                                padding='same',
                                use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block4_sepconv1_act')(x)
        x = layers.SeparableConv2D(filters.get(728,728), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block4_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block4_sepconv2_act')(x)
        if use_dp or training:
            x = layers.Dropout(0.1)(x,training=training)
            
        x = layers.SeparableConv2D(filters.get(728,728), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block4_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(self.rescale('depth',8)):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = layers.SeparableConv2D(filters.get(728,728), (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv1_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = layers.SeparableConv2D(filters.get(728,728), (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv2_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = layers.SeparableConv2D(filters.get(728,728), (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv3')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                        name=prefix + '_sepconv3_bn')(x)
            if use_dp or training:
                x = layers.Dropout(0.1)(x,training=training)
            
            x = layers.add([x, residual])

        residual = layers.Conv2D(filters.get(1024,1024), (1, 1), strides=(2, 2),
                                padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = layers.SeparableConv2D(filters.get(728,728), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block13_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block13_sepconv2_act')(x)
        if use_dp or training:
                x = layers.Dropout(0.1)(x,training=training)
        x = layers.SeparableConv2D(filters.get(1024,1024), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block13_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block13_pool')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(filters.get(1536,1536), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block14_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
        if use_dp or training:
                x = layers.Dropout(0.1)(x,training=training)
        x = layers.Activation('relu', name='block14_sepconv1_act')(x)

        x = layers.SeparableConv2D(filters.get(2048,2048), (3, 3),
                                padding='same',
                                use_bias=False,
                                name='block14_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv2_act')(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name='feature')(x)
            x = layers.Dense(classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D(name='feature')(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D(name='feature')(x)

        # Create model.
        model = models.Model(inp, x, name='xception')

        # Load weights.
        if weights == 'imagenet':
            if include_top:
                weights_path = keras_utils.get_file(
                    'xception_weights_tf_dim_ordering_tf_kernels.h5',
                    TF_WEIGHTS_PATH,
                    cache_subdir='models',
                    file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
            else:
                weights_path = keras_utils.get_file(
                    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    TF_WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b0042744bf5b25fce3cb969f33bebb97')
            model.load_weights(weights_path)
            if backend.backend() == 'theano':
                keras_utils.convert_all_kernels_in_model(model)
        elif weights is not None:
            model.load_weights(weights)

        return model
        

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
        l_rate = self.rescale('lr',l_rate)
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

    def rescaleEnabled(self):
        """
        Returns if the network is rescalable
        """
        return True
        


