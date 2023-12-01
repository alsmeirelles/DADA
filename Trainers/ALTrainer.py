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

#Local
from .GenericTrainer import Trainer
from .Predictions import Predictor
from .DataSetup import split_test

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
    trainer = ActiveLearningTrainer(config)
    trainer.run()

    
class ActiveLearningTrainer(Trainer):
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

        self.pool_x = None
        self.pool_y = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.initial_acq = 0
        if self._config.spool > 0:
            self.superp_x = None
            self.superp_y = None
            self.acq_idx = None
            self.sample_idx = None
            self.pool_size = None


    def coreset_pool(self):
        """
        Use CoreSet as a pool selection method
        """
        pass

    def _balance_classes(self,X,Y):
        """
        Returns a tuple (X,Y) of balanced classes

        X and Y are lists
        """
        #Work with NP arrays
        if isinstance(X,list):
            X = np.asarray(X)
        if isinstance(Y,list):
            Y = np.asarray(Y)
            
        #Count the occurrences of each class
        unique,count = np.unique(Y,return_counts=True)

        #Extracts the positions of each class in the dataset
        class_members = {i:np.where(Y == i)[0] for i in unique}

        #Remount the lists
        nX,nY = ([],[])
        mcount = count.min()
        for c in unique:
            if count[c] > mcount:
                ids = np.random.choice(count[c],mcount,replace=False)
                nX.extend(X[class_members[c][ids]])
                nY.extend(Y[class_members[c][ids]])
            else:
                nX.extend(X[class_members[c]])
                nY.extend(Y[class_members[c]])

        #Reshufle all elements
        combined = list(zip(nX,nY))
        random.shuffle(combined)
        nX[:],nY[:] = zip(*combined)

        return nX,nY


    def _restore_last_train(self):
        """
        Restore the last training set used in a previous experiment
        """
        cache_m = CacheManager()
        files = filter(lambda f:f.startswith('al-metadata'),os.listdir(self._config.logdir))
        metadata = {}
        for f in files:
            ac_id = int(f.split('.')[0].split('-')[3][1:])
            metadata[ac_id] = os.path.join(self._config.logdir,f)
        last = max(metadata.keys())
        name = os.path.basename(metadata[last]).split('.')[0].split('-')[2]
        train,_,_ = cache_m.load_file(metadata[last])

        return train[0],train[1],name,last
        
    def _restore_pools(self,sp):
        """
        Remove all past acquisitions from superpool. It's considerd that past acquisitions are already loaded
        into self.train_x

        @param sp <boolean>: use superpool as the reference or current pool ()
        """
        cache_m = CacheManager()
        pool_x, pool_y = None, None

        if sp:
            pool_x,pool_y = self.superp_x,self.superp_y
        else:
            pool_x,pool_y = self.pool_x, self.pool_y
            
        count = pool_x.shape[0]
        if self._config.info:
            print("Starting pool regeneration...({})".format(count))
            
        pool_dct = {pool_x[k]:k for k in range(count)}
        indexes = np.zeros((self.train_x.shape[0],),dtype=np.int32)
        for k in range(self.train_x.shape[0]):
            s = self.train_x[k]
            indexes[k] = pool_dct[s] if s in pool_dct else 0

        un_indexes = np.nonzero(indexes)[0]
        if self._config.info:
            print("Found {} patches in pool:".format(len(un_indexes)))
            print(" - Removing from pool (current size: {})".format(count))

        pool_x = np.delete(pool_x,indexes)
        pool_y = np.delete(pool_y,indexes)
        if self._config.info:
            print(" - Regenerated pool (new size {})".format(pool_x.shape[0]))
            removed = count - pool_x.shape[0]
            print(" - Removed {} patches from superpool".format(removed))
            print(" - Train set had {} duplicated patches.".format(self.train_x.shape[0]-removed))

        if sp:
            self.superp_x, self.superp_y = pool_x, pool_y
        else:
            self.pool_x, self.pool_y = pool_x, pool_y
    
    def _refresh_pool(self,r,name,**kwargs):
        """
        In kwargs:
        - regen_f : a function to choose the new pool points (if not given, selection is random). The function's first
        parameter should be the data;
        - regen_p : a tuple with this function's parameters, except the data which will be defined here.
        """
        
        if self.superp_x is None or self.superp_y is None:
            return None
        
        #Store every pool
        cache_m = CacheManager()
        fid = 'al-pool-{1}-r{0}.pik'.format(r,name)
        cache_m.registerFile(os.path.join(self._config.logdir,fid),fid)
        
        self.pool_x = None
        self.pool_y = None

        if self._config.info:
            print("[ALTrainer] Regenerating pool from superpool ({} patches available)".format(self.superp_x.shape[0]))
        if self._config.verbose > 0:
            to_remove = self.acq_idx.shape[0] if not self.acq_idx is None else 0
            print("[ALTrainer] To be removed from super pool ({1}): {0}".format(self.acq_idx,to_remove))

        if not self.acq_idx is None:
            self.superp_x = np.delete(self.superp_x,self.acq_idx)
            self.superp_y = np.delete(self.superp_y,self.acq_idx)
        if not 'regen_f' in kwargs:
            self.sample_idx = np.random.choice(self.superp_x.shape[0],self.pool_size,replace=False)
        else:
            if 'regen_p' in kwargs:
                rg_params = ((self.superp_x,self.superp_y),*kwargs['regen_p'])
            else:
                rg_params = ((self.superp_x,self.superp_y),)
            self.sample_idx = kwargs['regen_f'](*rg_params)
            
        self.pool_x = self.superp_x[self.sample_idx]
        self.pool_y = self.superp_y[self.sample_idx]
        cache_m.dump((self.pool_x,self.pool_y,name),fid)
        self.acq_idx = None
        self._ds.check_paths(self.pool_x,self._config.predst)
        
        if self._config.info:
            print("[ALTrainer] Pool regenerated: {}. Superpool size: {}".format(self.pool_y.shape[0],self.superp_x.shape[0])) 

    def _build_predictor(self):
        return Predictor(self._config,keepImg=self._config.keepimg)

    def _initializer(self,**kwargs):
        pass
    
    def configure_sets(self):
        """
        Creates the initial sets: training (X,Y); example pool; validation set; test set

        All sets are kept as NP arrays
        """

        self.test_x,self.test_y,X,Y = split_test(self._config,self._ds)
        
        #Use a sample of the metadata if so instructed
        if self._config.sample != 1.0:
            if self._config.spool > 0:
                self.superp_x = X
                self.superp_y = Y            
            X,Y,self.sample_idx = self._ds.sample_metadata(self._config.sample,data=(X,Y),pos_rt=self._config.pos_rt)
            self._ds.check_paths(X,self._config.predst)

        if self._config.balance:
            X,Y = self._balance_classes(X,Y)
            if self._config.info:
                print("[ALTrainer] Using a balanced initial dataset for AL ({} total elements).".format(len(X)))
        elif self._config.info:
            print("[ALTrainer] Using an UNBALANCED initial dataset for AL ({} total elements).".format(len(X)))

        if self._config.spool > 0:
            self.pool_size = X.shape[0]
            
        self.pool_x = np.asarray(X)
        self.pool_y = np.asarray(Y)
        del(X)
        del(Y)

        #Initial training set will be choosen at random from pool if a default is not provided
        cache_m = CacheManager()
        if self._config.restore:
            train_idx = None
            self.train_x, self.train_y, name, self.initial_acq = self._restore_last_train()
            if self._config.spool > 0:
                self._restore_pools(sp=True)
                self._refresh_pool(self.initial_acq,name)
            else:
                self._restore_pools(sp=False)
        elif self._config.load_train and not self._config.balance and cache_m.checkFileExistence('initial_train.pik'):
            train_idx = cache_m.load('initial_train.pik')
            if not train_idx is None and self._config.info:
                print("[ALTrainer] Using initial training set from cache. This is DANGEROUS. Use the metadata correspondent to the initial set.")
        else:
            if not self._config.load_train and self._config.balance and self._config.info:
                print("[ALTrainer] Dataset balancing and initial train set loading not possible at the same time.")
                
            train_idx = np.random.choice(len(self.pool_x),self._config.init_train,replace=False)
            cache_m.dump(train_idx,'initial_train.pik')

        #Validation element index definition
        val_samples = int((self._config.init_train*self._config.split[1])/self._config.split[0])
        val_samples = max(val_samples,100)
        val_idx = np.random.choice(np.setdiff1d(np.arange(self.pool_x.shape[0]),train_idx),val_samples,replace=False)

        if not train_idx is None:
            self.train_x = self.pool_x[train_idx]
            self.train_y = self.pool_y[train_idx]
        
        #Initial validation set - keeps the same split ratio for train/val as defined in the configuration
        self.val_x = self.pool_x[val_idx]
        self.val_y = self.pool_y[val_idx]

        #Remove the selected items from pool and set superset indexes to be removed when regenerating
        remove = val_idx if train_idx is None else np.concatenate((train_idx,val_idx),axis=0) 
        self.pool_x = np.delete(self.pool_x,remove)
        self.pool_y = np.delete(self.pool_y,remove)
        if self._config.spool > 0:
            self.acq_idx = self.sample_idx[remove]
        if self._config.sample != 1.0:
            self.sample_idx = np.delete(self.sample_idx,remove)


    def run(self):
        """
        General AL logic
        """
        
        #Loaded CNN model and Datasource
        model = self.load_modules()

        #Define initial sets
        self.configure_sets()
        #AL components
        cache_m = CacheManager()
        function = None
        
        if not self._config.ac_function is None:
            acq = importlib.import_module('AL','AcquisitionFunctions')
            function = getattr(acq,self._config.ac_function)
        else:
            print("You should specify an acquisition function")
            sys.exit(Exitcodes.RUNTIME_ERROR)

        #Starting from scratch or restoring?
        if self.initial_acq > 0:
            self.initial_acq += 1

        predictor = self._build_predictor()
        cache_m.dump(tuple(self._config.split),'split_ratio.pik')
        
        self.run_al(model,function,predictor,cache_m)
        
        
    def run_al(self,model,function,predictor,cache_m):
        """
        Coordenates the AL process.
        """
        from keras import backend as K
        
        stime = None
        etime = None
        train_time = None
        sw_thread = None
        end_train = False
        run_pred = False
        
        final_acq = self.initial_acq+self._config.acquisition_steps
        for r in range(self.initial_acq,final_acq):
            if self._config.info:
                print("[ALTrainer] Starting acquisition step {0}/{1}".format(r+1,final_acq))
                stime = time.time()

            #Save current dataset and report partial result (requires multi load for reading)
            fid = 'al-metadata-{1}-r{0}.pik'.format(r,model.name)
            cache_m.registerFile(os.path.join(self._config.logdir,fid),fid)
            cache_m.dump(((self.train_x,self.train_y),(self.val_x,self.val_y),(self.test_x,self.test_y)),fid)

            #Track training time
            train_time = time.time()
            tmodel,sw_thread,epad = self.train_model(model,(self.train_x,self.train_y),(self.val_x,self.val_y),save_numpy=True)
                                                             
            if self._config.info:
                print("Training step took: {}".format(timedelta(seconds=time.time()-train_time)))

            #Some models may take too long to save weights
            if not sw_thread is None and sw_thread.is_alive():
                if self._config.info:
                    print("[ALTrainer] Waiting for model weights...")
                sw_thread.join()
                
            run_pred = self.test_target(predictor,r,end_train)

            #Epoch adjustment
            if self._config.dye:
                ne = int(self._config.epochs * epad) if epad < 0 else int((1-epad)*self._config.epochs)
                print("Adjusting epochs ({} -> {}).".format(self._config.epochs,ne))
                self._config.epochs = max(min(ne,100),self.min_epochs)
                
            if r == (self._config.acquisition_steps - 1) or not self.acquire(function,model,acquisition=r,sw_thread=sw_thread):
                if self._config.info:
                    print("[ALTrainer] No more acquisitions are in order")
                end_train = True
                    
            #Set load_full to false so dropout is disabled
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

    def acquire(self,function,model,**kwargs):
        """
        Adds items to training and validation sets, according to split ratio defined in configuration. 
        Test set is fixed in the begining.

        Returns True if acquisition was sucessful and param 'return_aq' is False, otherwise return selected patches and updated pool
        """
        from Trainers import ThreadedGenerator
        import gc

        if kwargs is None:
            kwargs = {}

        kwargs['config'] = self._config

        #Some acquisition functions may need access to GenericModel
        kwargs['model'] = model

        tmodels = kwargs.get('emodels',None)
        return_aq = kwargs.get('return_aq',False)
        single_r = kwargs.get('single_r',False)
        
        #An acquisition function should return a NP array with the indexes of all items from the pool that 
        #should be inserted into training and validation sets
        if self.pool_x.shape[0] < self._config.acquire:
            return False

        fix_dim = model.check_input_shape()

        #Pools are big, use a data generator
        pool_prep = ImageDataGenerator(
            samplewise_center=self._config.batch_norm,
            samplewise_std_normalization=self._config.batch_norm)

        #Acquisition functions that require a generator to load data
        generator_params = {
            'dps':None,
            'classes':self._ds.nclasses,
            'dim':fix_dim,
            'keep':False,#self._config.keepimg,
            'batch_size':self._config.gpu_count * self._config.batch_size if self._config.gpu_count > 0 else self._config.batch_size,
            'image_generator':pool_prep,
            'shuffle':False, #DO NOT SET TRUE!
            'verbose':self._config.verbose}

        #Regenerate pool if defined
        if not single_r and self._config.spool > 0 and kwargs['acquisition'] > 0 and ((kwargs['acquisition'] + 1) % (self._config.spool)) == 0:
            if self._config.spool_f is None:
                self._refresh_pool(kwargs['acquisition'],model.name)
            else:
                acq = importlib.import_module('Trainers')
                spool_f = getattr(getattr(acq,'DataSetup'),self._config.spool_f)
                kwargs['space'] = 2
                params = (self.pool_size,generator_params,kwargs)
                self._refresh_pool(kwargs['acquisition'],model.name,regen_f=spool_f,regen_p=params)
            
        #Clear some memory before acquisitions
        gc.collect()

        #Set pool generator
        generator_params['dps'] = (self.pool_x,self.pool_y)
        generator = ThreadedGenerator(**generator_params)

        #For functions that need to access train data
        generator_params['dps'] = (self.train_x,self.train_y)
        train_gen = ThreadedGenerator(**generator_params)
        kwargs['train_gen'] = train_gen

        if not tmodels is None:
            pred_model = tmodels
        elif self._config.gpu_count > 1:
            pred_model = model.parallel
        else:
            pred_model = model.single

        if self._config.verbose > 0:
            print("\nStarting acquisition...(pool size: {})".format(self.pool_x.shape[0]))

        if self._config.verbose > 1:
            print("Starting acquisition using model: {0}".format(hex(id(pred_model))))

        #Track acquisition time
        ac_time = time.time()
        pooled_idx = function(pred_model,generator,self.pool_x.shape[0],**kwargs)
        if self._config.info:
            print("Acquisition step took: {}".format(timedelta(seconds=time.time() - ac_time)))

        if pooled_idx is None:
            if self._config.info:
                print("[ALTrainer] No indexes returned. Something is wrong.")
            sys.exit(1)
            
        #Store acquired patches indexes in pool set
        if not single_r and self._config.spool > 0:
            if self.acq_idx is None:
                self.acq_idx = self.sample_idx[pooled_idx]
            else:
                self.acq_idx = np.concatenate((self.acq_idx,self.sample_idx[pooled_idx]),axis=0)
            self.sample_idx = np.delete(self.sample_idx,pooled_idx)
 
        del(generator)
        
        self.train_x = np.concatenate((self.train_x,self.pool_x[pooled_idx]),axis=0)
        selected = None
        if return_aq:
            selected = self.pool_x[pooled_idx]
        self.pool_x = np.delete(self.pool_x,pooled_idx)
        
        if not self.pool_y is None:
            self.train_y = np.concatenate((self.train_y,self.pool_y[pooled_idx]),axis=0)        
            self.pool_y = np.delete(self.pool_y,pooled_idx)

        if return_aq:
            return (selected,self.pool_x)
        else:
            return True


    def _target_net_train(self,model):
        """
        Override this function for AL Transfer training
        """
        tm,st,_ = self.train_model(model,(self.train_x,self.train_y),(self.val_x,self.val_y),
                                     set_session=False,stats=False,summary=False,
                                     clear_sess=False,save_numpy=True)

        return [tm],[st],None
    
    def test_target(self,predictor,acqn,end_train,results=False):

        #Only run training/testing at every tnpred iterations
        if self._config.tnpred == 0:
            return False
        elif not end_train and acqn != 0 and ((acqn + 1) % (self._config.tnpred)) != 0:
            return False

        model = self.load_modules(self._config.tnet)
        model.setName("{}-Test".format(model.getName()))
        
        if model.rescaleEnabled():
            model.setPhi(self._config.tnphi)
            
        if self._config.info:
            print("\nStarting target network training...")

        intime = time.time()
        
        tm,st,_ = self._target_net_train(model)

        #Some models may take too long to save weights
        if not st is None and st[-1].is_alive():
            if self._config.info:
                print("[ALTrainer] Waiting for model weights...")
            st[-1].join()
                    
        #Set load_full to false so dropout is disabled
        res = predictor.run(self.test_x,self.test_y,load_full=model.is_ensemble(),net_model=model,target=True)        

        if self._config.info:
             print("Target net evaluation took: {}".format(timedelta(seconds=time.time() - intime)))
             print("\n")

        if results:
            return res
        else:
            return True
