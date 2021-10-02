#!/usr/bin/env python3
#-*- coding: utf-8

import random
import numpy as np
from Datasources.CellRep import CellRep
from Trainers import ActiveLearningTrainer

def check_labels(X,Y,X2,Y2):
    """
    X,Y: Set to be checked
    X2,Y2: Reference set
    Checks all labels from a sequential run against the ones produced by the Datasource classes.
    """
    reference = {X2[i]:Y2[i] for i in range(len(X2))}

    for j in range(len(X)):
        if reference[X[j]] != Y[j]:
            print("Item labels differ.\n - Reference item: {0};\n - Reference label: {1};\n Corresponding in given set label: {2}".format(
                X2[j],reference[X2[j]],Y[j]))

    print("If no messages reporting misslabeling was displayed, everything is good.")
    
def run(config):
    #Run all tests below
    config.data = 'CellRep'
    config.balance = True
    if config.local_test:
        config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    else:
        config.predst = '../data/lym_cnn_training_data/'
    
    #Start training
    trainer = ActiveLearningTrainer(config)
    trainer.load_modules()
    
    #Test set balancing
    X,Y = trainer._ds.load_metadata()
    bX,bY = trainer._balance_classes(X,Y)
    check_labels(bX,bY,X,Y)

    #Dataset size:
    print("Balanced set: {0} items, {1} labels".format(len(bX),len(bY)))
    unique,count = np.unique(bY,return_counts=True)
    l_count = dict(zip(unique,count))
    if not 1 in l_count:
        l_count[1] = 0
    print("Balanced set labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1]))))
    
    trainer.run()
