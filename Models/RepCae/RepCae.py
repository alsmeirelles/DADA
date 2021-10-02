#!/usr/bin/env python3
#-*- coding: utf-8

from Models import GenericModel

#Network
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D,Convolution2D, MaxPooling2D,AveragePooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation, LeakyReLU,BatchNormalization
from keras.layers import Input,Dot
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras import regularizers
from keras_contrib.layers import GroupNormalization
import keras
import numpy as np
import tensorflow as tf

class RepCae(GenericModel):
    """
    This is the CAE model as implemented in:
    'Spatial Organization And Molecular Correlation Of Tumor-Infiltrating Lymphocytes Using Deep Learning On Pathology Images'
    Published in Cell Reports
    """

    def __init__(self,config,ds):
        """
        Configuration defined by user
        """
        super().__init__(config,ds,'RepCae')

    def build(self):
        """
        Builds and returns a trainable model. 
        """
        
        width,height,channels = self._check_input_shape()
        
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)
            

        #Builds CAE model
        input_img = Input(shape=input_shape)
        
        #Layer 1
        l = Convolution2D(100, (5, 5),strides=1,padding='same',
                    kernel_initializer='he_normal',dilation_rate=1,input_shape=input_shape)(input_img)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        #Layer 2
        l = Convolution2D(120, (5, 5),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        l = AveragePooling2D(pool_size=(2, 2),strides=2)(l)
        #Layer 3
        l = Convolution2D(240, (3, 3),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        #Layer 4
        l = Convolution2D(320, (3, 3),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        l = AveragePooling2D(pool_size=(2, 2),strides=2)(l)
        #Layer 5
        l = Convolution2D(640, (3, 3),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        #Layer 6
        l = Convolution2D(1024, (3, 3),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        l = AveragePooling2D(pool_size=(2, 2),strides=2)(l)

        #Linearize - END Encoder
        l = Convolution2D(640, (1, 1),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        l = Convolution2D(100, (1, 1),strides=1,padding='same',kernel_initializer='he_normal',name='feat_map')(l)
        l = Activation('relu')(l)
        feat_map = GroupNormalization(groups=4,axis=-1)(l)
        l = Convolution2D(100, (1, 1),strides=1,padding='same',kernel_initializer='he_normal')(feat_map)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        l = Convolution2D(1, (1, 1),strides=1,padding='same',kernel_initializer='he_normal')(l)
        ## Original CAE applies a threshold layer before the Dot layer
        mask_rep = GroupNormalization(groups=1,axis=-1)(l)
        l = Dot(axes=1,name='Encoder')([feat_map,mask_rep])

        #Start decoder
        # Layer -6
        l = Conv2DTranspose(1024, (3, 3),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        # Layer -5
        l = Conv2DTranspose(640, (3, 3),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        # Layer -5 
        l = Conv2DTranspose(640, (4, 4),strides=2,padding='same',output_padding=(1,1),kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)
        # Layer -4
        l = Conv2DTranspose(320, (3, 3),strides=1,padding='same',kernel_initializer='he_normal')(l)
        l = LeakyRelu(alpha=0.2)(l)
        l = GroupNormalization(groups=4,axis=-1)(l)

        #TODO: complete model
        model = Model()
                
