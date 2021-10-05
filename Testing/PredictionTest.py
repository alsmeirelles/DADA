#!/usr/bin/env python3
#-*- coding: utf-8

from Datasources.CellRep import CellRep
from Trainers import Predictor
from Trainers.Predictions import print_prediction

from . import F1CallbackTest

def run(config):
    #Run all tests below
    config.data = 'CellRep'
    config.tdim = (300,300)
    if config.local_test:
        config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    else:
        config.predst = '../data/lym_cnn_training_data/'

    #First callback prediction test:
    F1CallbackTest.run(config)
    
    #Start predictions
    pred = Predictor(config)
    pred.run()
    if config.local_test:
        config.testdir = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/LHou/'
        config.data = 'LDir'
        pred = Predictor(config)
        pred.run()
    print_prediction(config)

