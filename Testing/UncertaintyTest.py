#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import random
from scipy.stats import mode

def test_varratios(data_size,dp_steps,query,gen_random=True):
    All_Dropout_Classes = np.zeros(shape=(data_size,1))

    for d in range(dp_steps):
        print("Step {0}/{1}".format(d+1,dp_steps))
        if gen_random:
            proba = np.random.random((data_size,2))
        else:
            proba = np.zeros((data_size,2),dtype=np.float)
        dropout_classes = proba.argmax(axis=-1)
        dropout_classes = np.array([dropout_classes]).T
        All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

    Variation = np.zeros(shape=(data_size))

    print("All classes: {0}".format(All_Dropout_Classes))
    for t in range(data_size):
        L = np.array([0])
        for d_iter in range(dp_steps):
            L = np.append(L, All_Dropout_Classes[t, d_iter+1])

        print("Vetor L: {0}".format(L))
        Predicted_Class, Mode = mode(L[1:])
        print("Predicted_Class: {0}; Mode: {1}".format(Predicted_Class,Mode))
        v = np.array(  [1 - Mode/float(dp_steps)])
        Variation[t] = v

    print("Variation array: {0}".format(Variation))
    a_1d = Variation.flatten()
    print("Vetor a_1d: {0}".format(a_1d))
    a_1d_sorted = a_1d.argsort()
    print("Array a_1d sorted: {0}".format(a_1d_sorted))
    x_pool_index = a_1d_sorted[-query:][::-1]
    print("Indexes: {0}".format(x_pool_index))
    print("Sorted a_1d: {0}".format(a_1d[x_pool_index]))

def test_databalance(img_rows=28,img_cols=28):
    from keras.datasets import mnist
    
    (X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

    X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    random_split = np.asarray(random.sample(range(0,X_train_All.shape[0]), X_train_All.shape[0]))

    X_train_All = X_train_All[random_split, :, :, :]
    y_train_All = y_train_All[random_split]

    X_valid = X_train_All[10000:12000, :, :, :]
    y_valid = y_train_All[10000:12000]

    X_Pool = X_train_All[20000:60000, :, :, :]
    y_Pool = y_train_All[20000:60000]

    X_train_All = X_train_All[0:10000, :, :, :]
    y_train_All = y_train_All[0:10000]

	#training data to have equal distribution of classes
    idx_0 = np.array( np.where(y_train_All==0)  ).T
    idx_0 = idx_0[0:2,0]
    X_0 = X_train_All[idx_0, :, :, :]
    y_0 = y_train_All[idx_0]

    idx_1 = np.array( np.where(y_train_All==1)  ).T
    idx_1 = idx_1[0:2,0]
    X_1 = X_train_All[idx_1, :, :, :]
    y_1 = y_train_All[idx_1]

    idx_2 = np.array( np.where(y_train_All==2)  ).T
    idx_2 = idx_2[0:2,0]
    X_2 = X_train_All[idx_2, :, :, :]
    y_2 = y_train_All[idx_2]

    idx_3 = np.array( np.where(y_train_All==3)  ).T
    idx_3 = idx_3[0:2,0]
    X_3 = X_train_All[idx_3, :, :, :]
    y_3 = y_train_All[idx_3]

    idx_4 = np.array( np.where(y_train_All==4)  ).T
    idx_4 = idx_4[0:2,0]
    X_4 = X_train_All[idx_4, :, :, :]
    y_4 = y_train_All[idx_4]

    idx_5 = np.array( np.where(y_train_All==5)  ).T
    idx_5 = idx_5[0:2,0]
    X_5 = X_train_All[idx_5, :, :, :]
    y_5 = y_train_All[idx_5]

    idx_6 = np.array( np.where(y_train_All==6)  ).T
    idx_6 = idx_6[0:2,0]
    X_6 = X_train_All[idx_6, :, :, :]
    y_6 = y_train_All[idx_6]

    idx_7 = np.array( np.where(y_train_All==7)  ).T
    idx_7 = idx_7[0:2,0]
    X_7 = X_train_All[idx_7, :, :, :]
    y_7 = y_train_All[idx_7]

    idx_8 = np.array( np.where(y_train_All==8)  ).T
    idx_8 = idx_8[0:2,0]
    X_8 = X_train_All[idx_8, :, :, :]
    y_8 = y_train_All[idx_8]

    idx_9 = np.array( np.where(y_train_All==9)  ).T
    idx_9 = idx_9[0:2,0]
    X_9 = X_train_All[idx_9, :, :, :]
    y_9 = y_train_All[idx_9]

    X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    print('Distribution of Training Classes:', np.bincount(y_train))
    
if __name__ == "__main__":
    test_varratios(200,5,10,gen_random=False)
    test_databalance()
