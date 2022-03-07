#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

import numpy as np
import tensorflow as tf

#Keras stuff
from keras.utils import multi_gpu_model
from keras import utils as keras_utils
from keras_contrib.layers import GroupNormalization
from keras import backend
from keras import layers
from keras import models
from keras import optimizers

#Locals
from Utils import CacheManager
from Models.GenericEnsemble import GenericEnsemble


BASE_WEIGHTS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/resnet/')
WEIGHTS_HASHES = {
    'ResNet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'ResNet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'ResNet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'ResNet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'ResNet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'ResNet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'ResNext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'ResNext101': ('34fb605428fcc7aa4d62f44404c11509',
                   '0f678c91647380debd923963594981b3')
}
    
def block1(x, filters, kernel_size=3, stride=1, use_dp=True, training=None,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    if use_dp or training:
        x = layers.Dropout(0.1)(x,training=training)    

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    if use_dp or training:
        x = layers.Dropout(0.1)(x,training=training)
        
    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    if use_dp or training:
        x = layers.Dropout(0.1)(x,training=training)
        
    return x


def stack1(x, filters, blocks, stride1=2, use_dp=True, training=None,name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, use_dp=use_dp, training=training, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, use_dp=use_dp, training=training, name=name + '_block' + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, use_dp=True, training=None,name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    if use_dp or training:
        x = layers.Dropout(0.1)(x,training=training)    

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    if use_dp or training:
        x = layers.Dropout(0.1)(x,training=training)    

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, use_dp=True, training=None, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, use_dp=use_dp, training=training,name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, use_dp=use_dp, training=training,name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, use_dp=use_dp, training=training, name=name + '_block' + str(blocks))
    return x

class ResNet50(GenericEnsemble):
    """
    Instantiates a ResNet50 model:

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Note that the default input image size for this model is 224x224x3.

    Reference: Deep residual learning for image recognition (CVPR - 2016)


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
            self.name = "ResNet50"        

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
    
    def _stack_fn(self,x,filters,use_dp,training):
        x = stack1(x, filters.get(64,64), self.rescale('depth',3), stride1=1, use_dp=use_dp, training=training, name='conv2')
        x = stack1(x, filters.get(128,128), self.rescale('depth',4), use_dp=use_dp, training=training, name='conv3')
        x = stack1(x, filters.get(256,256), self.rescale('depth',6), use_dp=use_dp, training=training, name='conv4')
        x = stack1(x, filters.get(512,512), self.rescale('depth',3), use_dp=use_dp, training=training, name='conv5')
        return x

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
        - preact: whether to use pre-activation or not (True for ResNetV2, False for ResNet and ResNeXt).
        - weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        - use_dp: use Dropout
        - pooling: final pooling type ('avg','max')
        - include_top: include top layers in model
        - use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        OBS: self.is_ensemble() returns if the ensemble strategy is in use
        """
        kwargs['preact'] = False
        kwargs['use_bias'] = True
        return self._basic_resnet(input_shape,training,preload,ensemble,**kwargs)
            
    def _basic_resnet(self,input_shape,training=None,preload=True,ensemble=False,**kwargs):

        weights = kwargs.get('weights',None)
        use_dp = kwargs.get('use_dp',True) #False if self.is_ensemble() else True
        pooling = kwargs.get('pooling','avg')
        include_top = kwargs.get('include_top',True)
        preact = kwargs.get('preact',False)
        use_bias = kwargs.get('use_bias',True)
        classes = self.nclasses
        batch_n = True if self._config.gpu_count <= 1 else False
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        
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
            
        # Determine proper input shape
        if input_shape is None:
            input_shape = (224,224,3)

        inp = layers.Input(shape=input_shape)

        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inp)
        x = layers.Conv2D(filters.get(64,64), 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

        if preact is False:
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        name='conv1_bn')(x)
            x = layers.Activation('relu', name='conv1_relu')(x)

        if use_dp or training:
            x = layers.Dropout(0.1)(x,training=training)

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        x = self._stack_fn(x,filters,use_dp,training)

        if preact is True:
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        name='post_bn')(x)
            x = layers.Activation('relu', name='post_relu')(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name='feature')(x)
            x = layers.Dense(classes, activation='softmax', name='probs')(x)
        else:
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D(name='feature')(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D(name='max_pool')(x)

        # Create model.
        model = models.Model(inp, x, name=self.name)

        # Load weights.
        if (weights == 'imagenet') and (self.name in WEIGHTS_HASHES):
            if include_top:
                file_name = self.name.lower() + '_weights_tf_dim_ordering_tf_kernels.h5'
                file_hash = WEIGHTS_HASHES[self.name][0]
            else:
                file_name = self.name.lower() + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
                file_hash = WEIGHTS_HASHES[self.name][1]
            weights_path = keras_utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path)
        elif weights is not None:
            model.load_weights(weights)

        return model        

    def rescaleEnabled(self):
        """
        Returns if the network is rescalable
        """
        return True    


class ResNet101(ResNet50):
    """
    Instantiates a ResNet 101 model

    Reference: Deep residual learning for image recognition (CVPR - 2016)
    """
    
    def __init__(self,config,ds,name=None,nclasses=2):
        super().__init__(config,ds,name=name,nclasses=nclasses)
        if name is None:
            self.name = "ResNet101"        

        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
        self.cache_m = CacheManager()
        self.cache_m.registerFile(os.path.join(config.model_path,self._modelCache),self._modelCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)    


    def _stack_fn(self,x,filters,use_dp,training):
        x = stack1(x, filters.get(64,64), self.rescale('depth',3), stride1=1, use_dp=use_dp, training=training, name='conv2')
        x = stack1(x, filters.get(128,128), self.rescale('depth',4), use_dp=use_dp, training=training, name='conv3')
        x = stack1(x, filters.get(256,256), self.rescale('depth',23), use_dp=use_dp, training=training, name='conv4')
        x = stack1(x, filters.get(512,512), self.rescale('depth',3), use_dp=use_dp, training=training, name='conv5')
        return x


class ResNet50V2(ResNet50):
    """
    Instantiates a ResNet 50 V2 model

    Reference: Deep residual learning for image recognition (CVPR - 2016)
    """
    
    def __init__(self,config,ds,name=None,nclasses=2):
        super().__init__(config,ds,name=name,nclasses=nclasses)
        if name is None:
            self.name = "ResNet50V2"        

        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
        self.cache_m = CacheManager()
        self.cache_m.registerFile(os.path.join(config.model_path,self._modelCache),self._modelCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)

    def _stack_fn(self,x,filters,use_dp,training):
        x = stack2(x, filters.get(64,64), self.rescale('depth',3), stride1=1, use_dp=use_dp, training=training, name='conv2')
        x = stack2(x, filters.get(128,128), self.rescale('depth',4), use_dp=use_dp, training=training, name='conv3')
        x = stack2(x, filters.get(256,256), self.rescale('depth',6), use_dp=use_dp, training=training, name='conv4')
        x = stack2(x, filters.get(512,512), self.rescale('depth',3), use_dp=use_dp, training=training, name='conv5')
        return x    

    def _build_architecture(self,input_shape,training=None,preload=True,ensemble=False,**kwargs):    
        """
        Parameters:
        - training <boolean>: sets network to training mode, wich enables dropout if there are DP layers
        - preload <boolean>: preload Imagenet weights
        - ensemble <boolean>: builds an ensemble of networks from the Inception architecture

        KWARGS:
        - preact: whether to use pre-activation or not (True for ResNetV2, False for ResNet and ResNeXt).
        - weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        - use_dp: use Dropout
        - pooling: final pooling type ('avg','max')
        - include_top: include top layers in model
        - use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        OBS: self.is_ensemble() returns if the ensemble strategy is in use
        """
        kwargs['preact'] = True
        kwargs['use_bias'] = True
        return self._basic_resnet(input_shape,training,preload,ensemble,**kwargs)    
