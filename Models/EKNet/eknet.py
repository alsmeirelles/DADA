#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

import numpy as np
import tensorflow as tf

#Network
from keras.models import Sequential,Model
from keras.layers import Input,Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D,Convolution2D, MaxPooling2D
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras.applications import vgg16
from keras import regularizers
from keras_contrib.layers import GroupNormalization
from keras import backend as K

#Locals
from Utils import CacheManager
from Models.GenericModel import GenericModel

class EKNet(GenericModel):
    """
    Implements abstract methods from GenericModel.
    Model is the same as in: https://keras.io/examples/mnist_cnn/
    """
    def __init__(self,config,ds,name=None):
        super().__init__(config,ds,name=name)
        if name is None:
            self.name = "ExtendedKerasNet"
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

        model,parallel_model = self._build(**kwargs)
        
        self.single = model
        self.parallel = parallel_model
        
        return (model,parallel_model)
    
    def _build(self,**kwargs):
        """
        @param pre_trained <boolean>: returned model should be pre-trained or not
        @param data_size <int>: size of the training dataset
        """
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
            
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        self.cache_m = CacheManager()
        
        model = self._build_architecture(input_shape,training,feature)
        
        #Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.registerFile(os.path.join(self._config.cache,lr_cache),lr_cache)
        l_rate = 0.0005
        if os.path.isfile(self.cache_m.fileLocation(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))
        
        #opt = optimizers.SGD(lr=l_rate, decay=1.5e-4, momentum=0.9, nesterov=True)
        #opt = optimizers.Adam(lr = l_rate)
        opt = optimizers.Adadelta()

        #Return parallel model if multiple GPUs are available
        parallel_model = None
        
        if self._config.gpu_count > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

            parallel_model = multi_gpu_model(model,gpus=self._config.gpu_count)
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

    def _build_architecture(self,input_shape,training=None,feature=False):
        original_vgg16 = vgg16.VGG16(weights=self.cache_m.fileLocation('vgg16_weights_notop.h5'),
                                         include_top=False,
                                         input_shape=input_shape)
        model = Sequential()
        for k in range(4):
            model.add(original_vgg16.layers[k])

        model.add(Convolution2D(128, (5, 5),strides=2,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))        
        model.add(Convolution2D(2048, (7, 7),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2048, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(self._ds.nclasses, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(self._ds.nclasses))
        model.add(Activation('softmax'))

        return model

class BayesEKNet(EKNet):
    """
    Bayesian model for the KNet
    """
    def __init__(self,config,ds):
        super(BayesEKNet,self).__init__(config=config,ds=ds,name = "BayesEKNet")

    def build_extractor(self,**kwargs):
        """
        Builds a feature extractor
        """

        return self._build(**kwargs)

    def _build_architecture(self,input_shape,training,feature):
        original_vgg16 = vgg16.VGG16(weights=self.cache_m.fileLocation('vgg16_weights_notop.h5'),
                                         include_top=False,
                                         input_shape=input_shape)

        layer_dict = dict([(layer.name, layer) for layer in original_vgg16.layers])

        del(original_vgg16)
        
        if hasattr(self,'data_size'):
            weight_decay = 2.5/float(self.data_size)
            if self._config.verbose > 1:
                print("Setting weight decay to: {0}".format(weight_decay))
        else:
            weight_decay = 0.01
            
        inp = Input(shape=input_shape)

        #First Layer
        x = Convolution2D(64, (3, 3),input_shape=input_shape,
                    strides=1,
                    padding='valid',
                    name='block1_conv1',
                    weights=layer_dict['block1_conv1'].get_weights(),
                    kernel_regularizer=regularizers.l2(weight_decay))(inp)
        x = GroupNormalization(groups=4,axis=-1)(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x,training=training)
 
        #Second layer
        x = ZeroPadding2D(padding=1)(x)
        x = Convolution2D(64, (3, 3),strides=1,
                    padding='valid',
                    name='block1_conv2',
                    weights=layer_dict['block1_conv2'].get_weights(),
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = GroupNormalization(groups=4,axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=2)(x)
        x = Dropout(0.1)(x,training=training)
 
        #Third layer
        x = ZeroPadding2D(padding=1)(x)
        x = Convolution2D(128, (3, 3),strides=1,
                    padding='valid',
                    name='block2_conv1',
                    weights=layer_dict['block2_conv1'].get_weights(),
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = GroupNormalization(groups=4,axis=-1)(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x,training=training)
 
        #Fourth layer
        x = ZeroPadding2D(padding=1)(x)
        x = Convolution2D(128, (3, 3),strides=1,
                padding='valid',
                name='block2_conv2',
                weights=layer_dict['block2_conv2'].get_weights(),
                kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = GroupNormalization(groups=4,axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=2)(x)
        x = Dropout(0.1)(x,training=training)

        #Fifth layer
        x = ZeroPadding2D(padding=1)(x)
        x = Convolution2D(128, (5, 5),strides=2,
                padding='valid',
                name='block2_conv3',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = GroupNormalization(groups=4,axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=2)(x)
        x = Dropout(0.1)(x,training=training)
        
        if feature:
            return Model(inp,x)
        
        x = Convolution2D(2048, (7, 7),strides=1,padding='valid',kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x,training=training)
        x = Convolution2D(2048, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x,training=training)
        x = Convolution2D(self._ds.nclasses, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal')(x)
        x = Flatten()(x)
        x = Dense(self._ds.nclasses)(x)
        output = Activation('softmax')(x)
        
        return Model(inp,output)
