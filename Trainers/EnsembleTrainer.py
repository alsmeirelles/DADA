#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import numpy as np
import importlib
import random
import time
from datetime import timedelta

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#Preparing migration to TF 2.0
import tensorflow as tf
if tf.__version__ >= '1.14.0':
    tf = tf.compat.v1
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.logging.set_verbosity(tf.logging.ERROR)
    #tf.disable_v2_behavior()
    
#Local
from .ALTrainer import ActiveLearningTrainer
from .Predictions import Predictor

#Module
from Utils import Exitcodes,CacheManager

def run_training(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting active learning process....")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    trainer = EnsembleALTrainer(config)
    trainer.run()

class EnsembleALTrainer(ActiveLearningTrainer):
    """
    Implements the structure of active learning:
    - Uses a selection function to acquire new training points;
    - Manages the training/validation/test sets

    Methods that should be overwriten by specific AL strategies:
    - 
    """

    def __init__(self,config):
        """
        @param config <argparse>: A configuration object
        """
        super().__init__(config)
        self.tower = None

    def _initializer(self,**kwargs):

        #initialize tensorflow session
        gpus = kwargs.get('gpus',0)
        processes = kwargs.get('processes',1)
        
        gpu_options = None
        if gpus > 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            gpu_options.allow_growth = True
            gpu_options.Experimental.use_unified_memory = False
            gpu_options.visible_device_list = ",".join([str(g) for g in range(gpus)])

        sess = tf.Session(config=tf.ConfigProto(
            device_count={"CPU":processes,"GPU":gpus},
            intra_op_parallelism_threads=3, 
            inter_op_parallelism_threads=3,
            log_device_placement=False,
            gpu_options=gpu_options
            ))
        #sess.config = config
        K.set_session(sess)

    def _print_stats(self,train_data,val_data):
        unique,count = np.unique(train_data[1],return_counts=True)
        l_count = dict(zip(unique,count))
        if len(unique) > 2:
            print("Training items:")
            print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
        else:
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

    def _build_predictor(self):
        return Predictor(self._config,keepImg=False,build_ensemble=True)
    
    def run_al(self,model,function,predictor,cache_m):
        """
        Coordenates the AL process
        """
        from keras import backend as K

        stime = None
        etime = None
        train_time = None
        sw_thread = None
        end_train = False
        t_models = None
        run_pred = False
        
        self._initializer(gpus=self._config.gpu_count,processes=self._config.cpu_count)

        final_acq = self.initial_acq+self._config.acquisition_steps
        for r in range(self.initial_acq,final_acq):
            if self._config.info:
                print("\n--------------------------------")
                print("[EnsembleTrainer] Starting acquisition step {0}/{1}".format(r+1,final_acq))
                stime = time.time()

            #Save current dataset and report partial result (requires multi load for reading)
            fid = 'al-metadata-{1}-r{0}.pik'.format(r,model.name)
            cache_m.registerFile(os.path.join(self._config.logdir,fid),fid)
            cache_m.dump(((self.train_x,self.train_y),(self.val_x,self.val_y),(self.test_x,self.test_y)),fid)

            self._print_stats((self.train_x,self.train_y),(self.val_x,self.val_y))
            sw_thread = None
            #Track training time
            train_time = time.time()

            t_models,sw_thread,cpad = self._target_net_train(model,reset=True)
                
            if self._config.info:
                print("Training step took: {}".format(timedelta(seconds=time.time()-train_time)))

            #If sw_thread was provided, we should check the availability of model weights
            if not sw_thread is None:
                for k in range(len(sw_thread)):
                    if sw_thread[k].is_alive():
                        print("Waiting ensemble model {} weights' to become available...".format(k))
                        sw_thread[k].join()

            run_pred = self.test_target(predictor,r,end_train)

            #Epoch adjustment
            if self._config.dye:
                epad = np.mean(cpad)
                ne = int(self._config.epochs * epad)
                print("Adjusting epochs ({}*{}): {}".format(self._config.epochs,epad,ne))
                self._config.epochs = max(min(ne,100),self.min_epochs)
                
            if r == (self._config.acquisition_steps - 1) or not self.acquire(function,model,acquisition=r,emodels=t_models,sw_thread=sw_thread):
                if self._config.info:
                    print("[EnsembleTrainer] No more acquisitions are in order")
                end_train = True
                model.reset() #Last AL iteration, force ensemble build for prediction
                model.tmodels = t_models
                
            #Set load_full loads a full model stored in file
            #Test target network if needed
            if not run_pred:
                predictor.run(self.test_x,self.test_y,load_full=end_train,net_model=model,target=self._config.tnet is None)

            #Attempt to free GPU memory
            K.clear_session()
            
            if self._config.info:
                etime = time.time()
                td = timedelta(seconds=(etime-stime))
                print("AL step took: {0}".format(td))
                
            if end_train:
                return None

        
    def _target_net_train(self,model,reset=True):

        t_models, sw_thread,cpad = {},[],[]
        for m in range(self._config.emodels):
            #Some models may take too long to save weights
            while True:
                if len(sw_thread) > 0 and sw_thread[-1].is_alive():
                    print("[EnsembleTrainer] Waiting for model weights.")
                    sw_thread[-1].join(60.0)
                else:
                    break
                    
            if hasattr(model,'register_ensemble'):
                model.register_ensemble(m)
            else:
                print("Model not ready for ensembling. Implement register_ensemble method")
                raise AttributeError

            if self._config.info:
                print("[EnsembleTrainer] Starting model {} training".format(m))
                    
            tm,st,epad = self.train_model(model,(self.train_x,self.train_y),(self.val_x,self.val_y),
                                         set_session=False,stats=False,summary=False,
                                         clear_sess=False,save_numpy=True)
            t_models[m] = tm
            sw_thread.append(st)
            cpad.append(epad)

        if reset:
            model.reset() #Last AL iteration, force ensemble build for prediction
            model.tmodels = t_models
        return t_models,sw_thread,cpad
