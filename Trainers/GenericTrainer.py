#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os,sys
import re
import numpy as np
import threading
import time
from datetime import timedelta

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
from Datasources.CellRep import CellRep
from Utils import SaveLRCallback,CalculateF1Score,EnsembleModelCallback
from Utils import Exitcodes,CacheManager
from .DataSetup import split_test
from AL.Common import load_model_weights

#Keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Training callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler,EarlyStopping
from keras.utils import to_categorical

#Preparing migration to TF 2.0
import tensorflow as tf
if tf.__version__ >= '1.14.0':
    v1 = tf.compat.v1
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    v1.logging.set_verbosity(v1.logging.ERROR)
    #v1.enable_eager_execution()
    #tf.disable_v2_behavior()

def run_training(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting training process....")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    trainer = Trainer(config)
    trainer.run()

def _reduce_lr_on_epoch(epoch,lr):
    #First 10 epochs, use a smaller LR, than raise to initially defined value
    #if epoch == 0:
    #    lr /= 10

    #if epoch == 9:
    #    lr *= 10
        
    #Reduces LR by a factor of 10 every 30 epochs
    if epoch > 9 and not (epoch%30):
        lr /= 10
    return lr

class Trainer(object):
    """
    Class that implements the training procedures applicable to all
    CNN models.

    Specialized training my be needed for some models and those should be 
    implemented elsewhere.

    @param config <argparse config>: configurations as specified by user
    @param ds <datasource>: some subclass of a GenericDatasource
    """

    def __init__(self,config):
        """
        @param config <parsed configurations>: configurations
        """
        self._config = config
        self._verbose = config.verbose
        self._ds = None
        self.min_epochs = config.epochs
        
    def load_modules(self,net_name = None, ds = None):
        if net_name is None:
            net_name = self._config.network
        if net_name is None or net_name == '':
            print("A network should be specified")
            return Exitcodes.RUNTIME_ERROR

        if not ds is None:
            self._ds = ds
        elif self._config.data:
            dsm = importlib.import_module('Datasources',self._config.data)
            self._ds = getattr(dsm,self._config.data)(self._config.predst,self._config.keepimg,self._config)
        else:
            self._ds = CellRep(self._config.predst,self._config.keepimg,self._config)

        net_module = importlib.import_module('Models',net_name)
        net_model = getattr(net_module,net_name)(self._config,self._ds)

        return net_model
    
    def run(self):
        """
        Checks configurations, loads correct module, loads data
        Trains!

        New networks should be inserted as individual modules. Networks should be imported
        by the Models module.
        """

        net_model = self.load_modules()

        #Test set splitting done in the same code now, outside GenericDatasource
        _,_,X,Y = split_test(self._config,self._ds)


        #Define training data
        train_data,val_data = None,None
        if self._config.sample != 1.0:
            X,Y,self.sample_idx = self._ds.sample_metadata(self._config.sample,data=(X,Y),pos_rt=self._config.pos_rt)
            self._ds.check_paths(X,self._config.predst)

        #After test set is separated, after data sampling is done, now split train and val
        train_data,val_data = self._ds.split_metadata(self._config.split[:2],data=(X,Y))

        _,sw_thread,_ = self.train_model(net_model,train_data,val_data,save_numpy=True)

        return sw_thread.join()

    def _choose_generator(self,train_data,val_data,fix_dim):
        """
        Returns a tuple with two batch generators: (train_generator,val_generator)
        The type of generator depends on the config.delay_load option
        """
        train_generator,val_generator = (None,None)

        train_prep = ImageDataGenerator(
            samplewise_center=self._config.batch_norm,
            samplewise_std_normalization=self._config.batch_norm)
        val_prep = ImageDataGenerator(
            samplewise_center=self._config.batch_norm,
            samplewise_std_normalization=self._config.batch_norm)

        if self._config.delay_load or self._config.phi > 0:
            from Trainers import ThreadedGenerator

            train_generator = ThreadedGenerator(dps=train_data,
                                                classes=self._ds.nclasses,
                                                dim=fix_dim,
                                                batch_size=self._config.batch_size,
                                                image_generator=train_prep,
                                                extra_aug=self._config.augment,
                                                shuffle=True,
                                                verbose=self._verbose,
                                                keep=self._config.keepimg)
            
            val_generator = ThreadedGenerator(dps=val_data,
                                                classes=self._ds.nclasses,
                                                dim=fix_dim,
                                                batch_size=self._config.batch_size,
                                                image_generator=val_prep,
                                                extra_aug=self._config.augment,
                                                shuffle=True,
                                                verbose=self._verbose,
                                                keep=self._config.keepimg)
        else:
            #Loads training images and validation images
            x_train,y_train = self._ds.load_data(split=None,keepImg=self._config.keepimg,data=train_data)
            
            x_val,y_val = self._ds.load_data(split=None,keepImg=self._config.keepimg,data=val_data)

            #Labels should be converted to categorical representation
            y_train = to_categorical(y_train,self._ds.nclasses)
            y_val = to_categorical(y_val,self._ds.nclasses)
            train_generator = train_prep.flow(x_train,y_train,batch_size=self._config.batch_size,shuffle=True)
            val_generator = val_prep.flow(x_val,y_val,batch_size=1)

        return (train_generator,val_generator)
    
    def train_model(self,model,train_data,val_data=None,**kwargs):
        """
        Generic trainer. Receives a GenericModel and trains it
        @param model <GenericModel>
        @param train_data <list>: Should be a collection of image metadata
        @param val_data <list>: Should be a collection of image metadata

        Optional keyword arguments:
        @param set_session <boolean>: configure session here
        @param verbose <int>: set verbosity level for training process. If not specified, use default
        @param summary <boolean>: print model summary
        @param clear_sess <boolean>: clears session and frees GPU memory
        @param allocated_gpus <int>: currently not used
        @param save_numpy <boolean>: save weights in numpy format instead of HDF5
        """

        if 'set_session' in kwargs:
            set_session = kwargs['set_session']
        else:
            set_session = True

        if 'stats' in kwargs:
            stats = kwargs['stats']
        else:
            stats = None

        if 'summary' in kwargs:
            summary = kwargs['summary']
        else:
            summary = True

        if 'clear_sess' in kwargs:
            clear_sess = kwargs['clear_sess']
        else:
            clear_sess = False

        if 'allocated_gpus' in kwargs:
            allocated_gpus = kwargs['allocated_gpus']
        else:
            allocated_gpus = self._config.gpu_count
            
        if 'save_numpy' in kwargs:
            save_numpy = kwargs['save_numpy']
        else:
            save_numpy = False
            
        # session setup
        if set_session:
            session = K.get_session()
            ses_config = tf.ConfigProto(
                device_count={"CPU":self._config.cpu_count,"GPU":self._config.gpu_count},
                intra_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count, 
                inter_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count,
                log_device_placement=True if self._verbose > 1 else False
                )
            session.config = ses_config
            K.set_session(session)
            
        if self._verbose > 0 and (stats is None or stats):
            unique,count = np.unique(train_data[1],return_counts=True)
            l_count = dict(zip(unique,count))
            if len(unique) > 2:
                print("Training items:")
                print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
            else:
                if unique.shape[0] == 1:
                    l_count[unique[0] ^ 1] = 0
                print("Train labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1]))))
            
            unique,count = np.unique(val_data[1],return_counts=True)
            l_count = dict(zip(unique,count))
            if len(unique) > 2:
                print("Validation items:")
                print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
            else:            
                if not 1 in l_count:
                    l_count[1] = 0
                print("Validation labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1]))))
            
            print("Train set: {0} items".format(len(train_data[0])))
            print("Validate set: {0} items".format(len(val_data[0])))

        train_generator,val_generator = self._choose_generator(train_data,val_data,model.check_input_shape())
        
        single,parallel = model.build(data_size=len(train_data[0]),allocated_gpus=allocated_gpus,preload_w=self._config.plw,layer_freeze=self._config.lyf)
        if not parallel is None:
            training_model = parallel
        else:
            training_model = single
            
        # try to resume the training - TODO: consider numpy weights file
        weights = list(filter(lambda f: (f.endswith(".h5") or f.endswith(".npy")) and f.startswith(model.name),os.listdir(self._config.weights_path)))
        weights.sort()
        old_e_offset = 0
        if len(weights) > 0 and not self._config.new_net:
            #print("[GenericTrainer] Intermediary weight load disabled, will train from scratch")
            training_model = load_model_weights(self._config,model,training_model)
        wf_header = "{0}-t{1}".format(model.name,old_e_offset+1)

        ### Define special behaviour CALLBACKS
        callbacks = []
        ## ModelCheckpoint
        if self._config.save_w:
            callbacks.append(ModelCheckpoint(os.path.join(
                self._config.weights_path, wf_header + "e{epoch:02d}.h5"), 
                save_weights_only=True, period=5,save_best_only=True,monitor='val_acc'))
        ## ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(monitor='loss',factor=0.7,\
                                           patience=10,verbose=self._verbose,\
                                           mode='auto',min_lr=1e-7))
        ##LearningRateScheduler
        callbacks.append(LearningRateScheduler(_reduce_lr_on_epoch,verbose=self._verbose))
        ##EarlyStopping
        #callbacks.append(EarlyStopping(monitor='acc',mode='max',min_delta=0.01,patience=5,baseline=0.9,verbose=1))
        ## CalculateF1Score
        if self._config.f1period > 0:
            callbacks.append(CalculateF1Score(val_generator,self._config.f1period,self._config.batch_size,self._config.info))

        if self._config.info and summary:
            print(single.summary())
        if self._config.verbose >= 1:
            print("Model parameters: {}".format(single.count_params()))
            print("Model layers: {}".format(len(single.layers)))

        hist = training_model.fit_generator(
            generator = train_generator,
            steps_per_epoch = len(train_generator), #// self._config.batch_size,
            epochs = self._config.epochs,
            validation_data = val_generator,
            validation_steps = len(val_generator), #//self._config.batch_size,
            verbose = self._verbose,
            use_multiprocessing = False,
            workers=self._config.cpu_count*2,
            max_queue_size=self._config.batch_size*3,
            callbacks=callbacks,
            )

        if self._verbose > 1:
            print("Done training model: {0}".format(hex(id(training_model))))

        if hasattr(self._config,'dye') and self._config.dye:
            epad = ((np.mean(hist.history['loss']) - hist.history['loss'][-1]) + (np.mean(hist.history['acc']) - hist.history['acc'][-1]))/ \
            (np.std(hist.history['loss'])+np.std(hist.history['acc']))
            dloss = np.subtract(hist.history['val_loss'],hist.history['loss'])
            ldloss = np.std(dloss)
            lmean = np.mean(dloss)
            macc = np.mean(hist.history['acc'][-3:])
            mhloss = np.mean(dloss[2*int(len(dloss)/3):])
            sign = -1.0 if ((mhloss > 1.1*(lmean+ldloss)) or macc >= 0.995)  else 1.0
            print("Cut point: {}; High band MEAN: {}".format(1.1*(lmean+ldloss),mhloss))
            epad = sign*abs(epad)
            if epad < -1.0:
                print('Case 1')
                epad = epad/(-1+epad)
            elif -1.0 <= epad < 0:
                print('Case 2')
                epad = abs(epad)
            elif 0 < epad <= 1.0:
                print('Case 3')
                epad += 1
        else:
            epad = 1

        sw_thread = threading.Thread(target=self._save_weights,name='save_weights',args=(model,single,parallel,clear_sess,save_numpy))
        sw_thread.start()
        return (training_model,sw_thread,epad)
        
    def _save_weights(self,model,single,parallel,clear_sess,save_numpy):
        #Save weights for single tower model and for multigpu model (if defined)
        cache_m = CacheManager()
        stime = None
        if self._config.info:
            stime = time.time()
            print("Saving weights, this could take a while...")
            
        if save_numpy and hasattr(model,'get_npweights_cache'):
            np.save(model.get_npweights_cache(),single.get_weights())
        else:
            single.save_weights(model.get_weights_cache())
            single.save(model.get_model_cache())
            
        if not parallel is None and not model.get_mgpu_weights_cache() is None:
            if save_numpy and hasattr(model,'get_npmgpu_weights_cache'):
                np.save(model.get_npmgpu_weights_cache(),parallel.get_weights())
            else:
                parallel.save_weights(model.get_mgpu_weights_cache())

        if self._config.info:
                etime = time.time()
                td = timedelta(seconds=(etime-stime))
                print("Weight saving took: {}".format(td))

        if clear_sess:
            K.clear_session()
            
        return Exitcodes.ALL_GOOD
        
