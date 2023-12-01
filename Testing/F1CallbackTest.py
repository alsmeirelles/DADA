#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os,sys

import numpy as np
from Utils import CalculateF1Score
from Trainers import ThreadedGenerator

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

def run(config):
    print("Starting Callback Predictions...")
    #Run all tests below
    config.data = 'CellRep'
    config.info = True
    if config.local_test:
        config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    else:
        config.predst = '../data/lym_cnn_training_data/'


    if config.data:
        dsm = importlib.import_module('Datasources',config.data)
        if config.testdir:
            ds = getattr(dsm,config.data)(config.testdir,config.keepimg,config)
        else:
            ds = getattr(dsm,config.data)(config.predst,config.keepimg,config)

    net_module = importlib.import_module('Models',config.network)
    model = getattr(net_module,config.network)(config,ds)


    if os.path.isfile(model.get_model_cache()):
        try:
            pred_model = load_model(model.get_model_cache())
            if config.info:
                print("Model loaded from: {0}".format(model.get_model_cache()))
        except ValueError:
            pred_model,_ = model.build()
            pred_model.load_weights(model.get_weights_cache())
            if config.info:
                print("Model weights loaded from: {0}".format(model.get_weights_cache()))
    elif os.path.isfile(model.get_weights_cache()):
        pred_model,_ = model.build()
        pred_model.load_weights(model.get_weights_cache())
        if config.info:
            print("Model weights loaded from: {0}".format(model.get_weights_cache()))
                    
    else:
        if config.info:
            print("No trained model or weights file found")
        return None    

    if config.testdir is None:
        ds.load_metadata()
        _,_,(x_test,y_test) = ds.split_metadata(config.split)
    else:
        x_test,y_test = ds._run_dir(config.testdir)

    val_prep = ImageDataGenerator(
        samplewise_center=config.batch_norm,
        samplewise_std_normalization=config.batch_norm)
    val_generator = ThreadedGenerator(dps=(x_test,y_test),
                                        classes=ds.nclasses,
                                        dim=config.tdim,
                                        batch_size=config.batch_size,
                                        image_generator=val_prep,
                                        shuffle=True,
                                        verbose=config.verbose)
    
    f1cb = CalculateF1Score(val_generator,period=20,batch_size=config.batch_size,info=True)
    f1cb.model = pred_model

    print("Testing periodicity, should not run.")
    f1cb.on_epoch_end(epoch=4)
    print("Now, run callback prediction.")
    f1cb.on_epoch_end(epoch=20)
