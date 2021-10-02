#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os

from ALPlot import Plotter

if __name__ == "__main__":
    external_p = '/Volumes/Trabalho/Doutorado/Code/Segmentation/Active-Learning-Bayesian-Convolutional-Neural-Networks/Bridges-Results'
    external_f = ['Dropout_VarRatio_Q10_N1000_Accuracy_Results_Experiment_{0}.npy'.format(i) for i in range(0,3)]
    results = '/Volumes/Trabalho/Doutorado/Code/Segmentation/Segframe/results/MN'
    selected = ['36','34']
    

    p = Plotter()
    data = p.parseResults(results,selected)
    ext_acc = [p.extractNPData(os.path.join(external_p,f)) for f in external_f]
    for ind in range(len(ext_acc)):
        ext_data = {'accuracy':ext_acc[ind],
                    'trainset':p.generateIndexes(ext_acc[ind].shape[0],20,10)}
        data['Ext-{}'.format(ind)] = ext_data

    p.draw_multiline(data,'Paper experiments (external) vs MNIST experiments {}'.format(selected))
        
