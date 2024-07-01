#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation,LeakyReLU
from keras import backend,optimizers
from keras.utils import to_categorical
from keras.datasets import mnist
import keras.backend as K

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

import numpy as np
np.random.seed(7)


if __name__ == "__main__":
    
    img_rows, img_cols = 28, 28
    epocas = 1
    batch_size = 32
    nb_train_samples = 1000
    nb_validation_samples = 200
    
    if backend.image_data_format() == 'channels_first':
        input_shape = (1, img_cols, img_rows)
    else:
        input_shape = (img_cols, img_rows, 1)

    model = Sequential()

    #First layer
    model.add(Convolution2D(32, (3, 3),strides=1,padding='valid',kernel_initializer='he_normal',dilation_rate=1,input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    #Second layer
    model.add(Convolution2D(64, (3, 3),strides=1,padding='valid',kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    #third layer
    model.add(Convolution2D(128, (3, 3),padding='valid',kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1))
    #Output layer
    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.2e-3, decay=1.5e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    #Normalize
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255       
    tr_size = x_train.shape[0]
    test_size = x_test.shape[0]

    rand_w = model.get_weights()
    model.fit(x_train,to_categorical(y_train),epochs=epocas,batch_size=batch_size,verbose=1)
    trained_w = model.get_weights()
    
    print("Predict with trained weights")
    yh = np.argmax(model.predict(x_test,verbose=1),axis=1)
    print("Accuracy: {}".format(accuracy_score(y_test,yh)))

    print("Predict with random weights")
    model.set_weights(rand_w)
    yh = np.argmax(model.predict(x_test,verbose=1),axis=1)
    print("Accuracy: {}".format(accuracy_score(y_test,yh)))
