#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

import numpy as np
import tensorflow as tf

#Network
from keras.models import Sequential,Model
from keras.layers import Input,Activation
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import ZeroPadding2D,Convolution2D, MaxPooling2D
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras.applications import vgg16
from keras import regularizers
from keras_contrib.layers import GroupNormalization
from keras import backend as K

#Locals
from Utils import CacheManager
from Models.GenericEnsemble import GenericEnsemble

class VGG16(GenericEnsemble):
    """
    Implements abstract methods from GenericModel.
    Producess a VGG16 model as implemented by Keras, with convolutional layers
    FC layers are substituted by Conv2D, as defined in:
    https://github.com/ALSM-PhD/quip_classification/blob/master/NNFramework_TF/sa_networks/vgg.py
    """
    def __init__(self,config,ds,name=None):
        super().__init__(config,ds,name=name)
        if name is None:
            self.name = "VGG16_A1"
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
        Returns a VGG 16 model instance, final fully-connected layers are substituted by Conv2Ds
        
        @param pre_trained <boolean>: returned model should be pre-trained or not
        """
        
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']
            
        self.cache_m = CacheManager()
        
        model = self._build_architecture(input_shape)
        
        #Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.registerFile(os.path.join(self._config.cache,lr_cache),lr_cache)
        l_rate = self._config.learn_r
        if os.path.isfile(self.cache_m.fileLocation(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))

        l_rate = self.rescale('lr',l_rate)
        sgd = optimizers.SGD(lr=l_rate, decay=1.5e-4, momentum=0.9, nesterov=True)
        #adam = optimizers.Adam(lr = l_rate)
        
        #Return parallel model if multiple GPUs are available
        parallel_model = None
        
        if self._config.gpu_count > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

            parallel_model = multi_gpu_model(model,gpus=self._config.gpu_count)
            parallel_model.compile(loss='categorical_crossentropy',
                                       optimizer=sgd,
                                       metrics=['accuracy'],
                                       #options=p_opt, 
                                       #run_metadata=p_mtd
                                       )
        else:
            model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'],
                #options=p_opt, 
                #run_metadata=p_mtd
                )

        self.single = model
        self.parallel = parallel_model

        return (model,parallel_model)

    def _build_architecture(self,input_shape):
        original_vgg16 = vgg16.VGG16(weights=self.cache_m.fileLocation('vgg16_weights_notop.h5'),
                                         include_top=False,
                                         input_shape=input_shape)

        #Freeze initial layers, except for the last 3:
        #for layer in original_vgg16.layers[:-2]:
        #    layer.trainable = False
            
        model = Sequential()
        model.add(original_vgg16)
        model.add(Convolution2D(4096, (7, 7),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.75))
        model.add(Convolution2D(4096, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.75))
        model.add(Convolution2D(self._ds.nclasses, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(self._ds.nclasses))
        model.add(Activation('softmax'))

        return model

class EFVGG16(VGG16):
    """
    VGG variation, uses GroupNormalization and more dropout
    """
    def __init__(self,config,ds):
        super(EFVGG16,self).__init__(config=config,ds=ds,name = "EFVGG16")

    def _build_architecture(self,input_shape):

        original_vgg16 = vgg16.VGG16(weights=self.cache_m.fileLocation('vgg16_weights_notop.h5'),
                                         include_top=False,
                                         input_shape=input_shape)
        layer_dict = dict([(layer.name, layer) for layer in original_vgg16.layers])

        S = 1000
        if hasattr(self,'data_size'):
            S = self.data_size
        wd = lambda p,N: (1-p)*0.5/N

        filters = {7:self.rescale('width',7),
                    32:self.rescale('width',32),
                    48:self.rescale('width',48),
                    64:self.rescale('width',64),
                    80:self.rescale('width',80),
                    96:self.rescale('width',96),
                    128:self.rescale('width',128),
                    160:self.rescale('width',160),
                    192:self.rescale('width',192),
                    224:self.rescale('width',224),
                    256:self.rescale('width',256),
                    512:self.rescale('width',512),
                    4096:self.rescale('width',4096)}

        depth = round(self.rescale('depth',13))+3 #13 feature layers, keep the first tree always
        
        inp = Input(shape=input_shape)
        x = Convolution2D(filters.get(64,64), (3, 3),input_shape=input_shape,
                    strides=1,
                    padding='same',
                    name='block1_conv1',
                    kernel_initializer='he_normal',
                    #weights=layer_dict['block1_conv1'].get_weights(),
                    kernel_regularizer=regularizers.l2(wd(0.1,S)))(inp)
        #x = GroupNormalization(groups=4,axis=-1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
 
        #Second layer
        x = ZeroPadding2D(padding=1)(x)
        x = Convolution2D(filters.get(64,64), (3, 3),strides=1,
                    padding='valid',
                    name='block1_conv2',
                    kernel_initializer='he_normal',
                    #weights=layer_dict['block1_conv2'].get_weights(),
                    kernel_regularizer=regularizers.l2(wd(0.1,S)))(x)
        #x = GroupNormalization(groups=4,axis=-1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=2)(x)
        x = Dropout(0.1)(x)
 
        #Third layer
        x = ZeroPadding2D(padding=1)(x)
        x = Convolution2D(filters.get(128,128), (3, 3),strides=1,
                    padding='valid',
                    name='block2_conv1',
                    kernel_initializer='he_normal',
                    #weights=layer_dict['block2_conv1'].get_weights(),
                    kernel_regularizer=regularizers.l2(wd(0.1,S)))(x)
        #x = GroupNormalization(groups=4,axis=-1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
 
        #Fourth layer
        x = ZeroPadding2D(padding=1)(x)
        x = Convolution2D(filters.get(128,128), (3, 3),strides=1,
                padding='valid',
                name='block2_conv2',
                kernel_initializer='he_normal',
                #weights=layer_dict['block2_conv2'].get_weights(),
                kernel_regularizer=regularizers.l2(wd(0.1,S)))(x)
        #x = GroupNormalization(groups=4,axis=-1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=2)(x)
        x = Dropout(0.1)(x)
 
        #Fifth layer
        if depth >= 6:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(256,256), (3, 3),strides=1,
                                  padding='valid',
                                  name='block3_conv1',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block3_conv1'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.2,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)
 
        #Sith layer
        if depth >= 7:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(256,256), (3, 3),strides=1,
                                  padding='valid',
                                  name='block3_conv2',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block3_conv2'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.2,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)

        #Seventh layer
        if depth >= 8:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(256,256), (3, 3),strides=1,
                                  padding='valid',
                                  name='block3_conv3',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block3_conv3'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.2,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2),strides=2)(x)
            x = Dropout(0.2)(x)
 
        #Eigth layer
        if depth >= 9:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(512,512), (3, 3),strides=1,
                                  padding='valid',
                                  name='block4_conv1',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block4_conv1'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.2,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)

        #Nineth layer
        if depth >= 10:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(512,512), (3, 3),strides=1,
                                  padding='valid',
                                  name='block4_conv2',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block4_conv1'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.2,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)        
 
        #Tenth layer
        if depth >= 11:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(512,512), (3, 3),strides=1,
                                  padding='valid',
                                  name='block4_conv3',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block4_conv2'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.2,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2),strides=2)(x)
            x = Dropout(0.2)(x)
 
        #Eleventh layer
        if depth >= 12:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(512,512), (3, 3),strides=1,
                                  padding='valid',
                                  name='block5_conv1',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block4_conv3'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.2,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)
 
        #Twelth layer
        if depth >= 12:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(512,512), (3, 3),strides=1,
                                  padding='valid',
                                  name='block5_conv2',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block5_conv1'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.3,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.3)(x)

        #Thirtenth layer
        if depth >= 13:
            x = ZeroPadding2D(padding=1)(x)
            x = Convolution2D(filters.get(512,512), (3, 3),strides=1,
                                  padding='valid',
                                  name='block5_conv3',
                                  kernel_initializer='he_normal',
                                  #weights=layer_dict['block5_conv3'].get_weights(),
                                  kernel_regularizer=regularizers.l2(wd(0.3,S)))(x)
            #x = GroupNormalization(groups=4,axis=-1)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2),strides=2,name='feature')(x)
        x = Dropout(0.3)(x)
        
        #x = Convolution2D(filters.get(4096,4096), (7, 7),strides=1,padding='valid',kernel_initializer='he_normal',
        #                      kernel_regularizer=regularizers.l2(wd(0.5,S)))(x)

        x = Flatten()(x)
        x = Dense(4096,kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        #x = Convolution2D(filters.get(4096,4096), (1, 1),strides=1,padding='valid',kernel_initializer='he_normal',
        #                      kernel_regularizer=regularizers.l2(wd(0.5,S)))(x)
        x = Dense(4096,kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        #x = Convolution2D(self._ds.nclasses, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal')(x)
        x = Dense(self._ds.nclasses,kernel_initializer='he_normal')(x)
        output = Activation('softmax')(x)

        return Model(inp,output)

    def rescaleEnabled(self):
        """
        Returns if the network is rescalable
        """
        return True
