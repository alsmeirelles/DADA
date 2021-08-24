#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os,sys
import copy
import shutil
import math
import argparse
import pickle
import importlib
import re
from sklearn.cluster import KMeans
    
def _process_al_metadata(config):
    """
    Returns the images acquired in each acquisition, as stored in al-metadata files
    """
    if config.sdir is None or not os.path.isdir(config.sdir):
        print("Directory not found: {}".format(config.sdir))
        sys.exit(1)

    files = os.listdir(config.sdir)

    acfiles = {}
    for f in files:
        if f == '{}-inittrain.pik'.format(config.ds) and config.pinit:
            ac_id = 0
            acfiles[ac_id] = os.path.join(config.sdir,f)
        elif f.startswith('al-metadata'):
            ac_id = int(f.split('.')[0].split('-')[3][1:]) + 1
            net = f.split('.')[0].split('-')[2]
            if not config.ac_n is None and max(config.ac_n)+1 < ac_id:
                continue
            if (not config.net is None and config.net == net) or config.net is None:
                acfiles[ac_id] = os.path.join(config.sdir,f)

    ordered_k = list(acfiles.keys())
    ordered_k.sort()
    initial_set = None
    ac_imgs = {}
    for k in ordered_k:
        with open(acfiles[k],'rb') as fd:
            try:
                train,val,test = pickle.load(fd)

            #Needed when inital training data is requested
            except ValueError:
                fd.seek(0)
                idx = pickle.load(fd)
                fd.close()
                print('Grabing metadata from sample indexes.')
                mt = os.path.join(config.sdir,'{}-sampled_metadata.pik'.format(config.ds))
                if os.path.isfile(mt):
                   fd = open(mt,'rb')
                   sd = pickle.load(fd)
                   if len(sd) == 3:
                       tx,ty,_ = sd
                   else:
                       tx,ty,_,_ = sd
                   fd.close()
                   tx = np.asarray(tx)[idx]
                   train = (tx,ty)
                else:
                    print("Not ready to extract from full DS. Missing sampled metadata.")
                    sys.exit(1)
            
        if initial_set is None:
            #Acquisitions are obtained from keys k and k-1
            initial_set = list(train[0])
        else:
            ctrain = list(train[0])
            if k == 1 and config.pinit:
                mask = np.isin(ctrain,initial_set,assume_unique=True,invert=False)
            else:
                mask = np.isin(ctrain,initial_set,assume_unique=True,invert=True)
                if config.debug:
                    dbm = np.isin(ctrain,initial_set,assume_unique=True,invert=False)
                    print("Images in both sets: {}".format(dbm.shape[0]))

            imgs = train[0][mask]
            labels = train[1][mask]
            print("Acquired {} images in acquisition {}".format(imgs.shape[0],k-1))
            ac_imgs[k-1] = (imgs,labels)
            initial_set = ctrain

    return ac_imgs

def _process_test_images(config):
    """
    Selects a predefined number of test patches for extraction
    """
    if config.sdir is None or not os.path.isdir(config.sdir):
        print("Directory not found: {}".format(config.sdir))
        sys.exit(1)

    files = os.listdir(config.sdir)

    dsfiles = {}
    testfile = '{}-testset.pik'.format(config.ds)
    fX = None
    if not testfile in files:
        #TODO: if no test file, test set is defined as last items in metadata array and can be returned
        return None
    else:
        with open(os.path.join(config.sdir,testfile),'rb') as fd:
            _,tset = pickle.load(fd)
        with open(os.path.join(config.sdir,'{}-metadata.pik'.format(config.ds)),'rb') as fd:
            X,Y,_ = pickle.load(fd)
            fX = np.asarray(X)
            del(X)
        
        tx = fX[tset[:config.test]]

    return tx
        
    
def _process_wsi_cluster(km,s,members,config):
    """
    Generate statistics for each cluster.
    km -> KMeans instance
    s -> WSI name
    members -> cluster members (img,label)
    config -> configurations
    """
    distances = {}
    wrad = 0
    wsi_mean = 0.0
    for c in range(config.nc):
        idx = np.where(km.labels_ == c)[0]
        count_m = 0
        c_count = 0
        mean_d = 0.0
        if config.info:
            print("Cluster {} center: {}".format(c,km.cluster_centers_[c]))
        x2 = km.cluster_centers_[c]
        for p in idx:
            x1 = members[p].getCoord()
            if x1 is None:
                continue
            c_count += 1
            dist = math.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)
            mean_d += dist
            if dist < config.radius:
                distances[members[p]] = dist
                count_m += 1
        if config.info:
            print("    - patches in cluster: {}".format(idx.shape[0]))
            print("    - {} patches are within {} pixels from cluster center".format(count_m,config.radius))
        if c_count > 0:
            mean_d = mean_d / c_count
            wsi_mean += mean_d
            if config.info:
                print("    - {:2.2f} % of cluster patches are within range".format(100*count_m/c_count))
        if config.info:
            print("    - Mean distance from cluster center: {:.1f}".format(mean_d))
        wrad += count_m        
    wsi_mean = wsi_mean/config.nc
    if config.info:
        print("{:2.2f}% of all WSI patches are within cluster center ranges".format(100*wrad/len(members)))
        print("Mean distance of clustered patches to centers: {:.1f}".format(wsi_mean))

    return wsi_mean

def _combine_acquired_ds(wsis,ds_wsis,max_w=20):
    """
    wsis and ds_wsis: dictionary of tuples:
    wsi[k] -> ([imgs],[labels])
    k -> WSI name
    """
    acquired = list(wsis.keys())
    acquired.sort(key=lambda x:len(wsis[x][0]),reverse=True)

    acquired = acquired[:max_w]

    total_patches = 0
    for d in ds_wsis:
        total_patches += len(ds_wsis[d][0])

    print("\n"*3 + " "*10 + "CHECKING ACQUIRED IMAGES AGAINST DATASET")
    print(" "*10 + "Total of patches in dataset: {}".format(total_patches))
    for w in acquired:
        acq = len(wsis[w][0])
        dsp = len(ds_wsis[w][0])
        print("WSI {}:".format(w))
        print("   - {} patche(s) acquired".format(acq))
        print("   - {:2.4f} % of available patches from the WSI ({})".format(100*acq/dsp,dsp))
        print("   - {:2.4f} % of dataset patches are from this WSI".format(100*dsp/total_patches))
        
def _save_wsi_stats(wsis,dest,ac_save):

    init_k = 1
    for a in ac_save:
        save_to = os.path.join(dest,'patch_coordinates-{}-{}.txt'.format(init_k,a))
        fd = open(save_to,'w')
        fd.write('X-coordinate Y-coordinate Width Height\n')
        patches = {}
        for k in range(init_k,a):
            for w in wsis[k]:
                patches.setdefault(w,[])
                patches[w].extend(wsis[k][w])

        for ws in patches:
            fd.write('WSI {}: {}\n'.format(ws,len(patches[ws])))
            for i in patches[ws]:
                coord = i.getCoord()
                if not coord is None:
                    try:
                        dim = i.getImgDim()
                    except FileNotFoundError:
                        dim = (0,0)
                    fd.write('{0} {1} {2} {3}\n'.format(coord[0],coord[1],dim[0],dim[1]))
                else:
                    fd.write('No coordinates\n')
        init_k = a
        fd.close()

def _generate_label_files(config):
    """
    This should be run after process_al_metadata
    """
    dirs = os.listdir(config.out_dir)

    labels = {}
    for d in dirs:
        if not d in labels:
            if config.keep:
                ol = open(os.path.join(config.cp_orig,d,'label.txt'),'r')
            labels[d] = {}
            for line in ol.readlines():
                fields = line.strip().split(' ')
                labels[d][fields[0]] = fields[1:]
            ol.close()
        copy_to = os.path.join(config.out_dir,d)
        if os.path.isfile(os.path.join(copy_to,'label.txt')):
            fd = open(os.path.join(copy_to,'label.txt'),'a')
        else:
            fd = open(os.path.join(copy_to,'label.txt'),'w')
        for i in os.listdir(copy_to):
            if i.endswith('png'):
                fd.write('{} {}\n'.format(i,' '.join(labels[d][i])))

        fd.close()

def _generate_label_from_name(config):
    """
    Regenerate the label.txt files based on patch name
    """
    pf_form = '(UN-(?P<unc>[0-9])+-){,1}(?P<tcga>TCGA-.*-.*-.*-.*-.*)-(?P<x>[0-9]+)-(?P<y>[0-9]+)-(?P<s1>[0-9]+)-(?P<s2>[0-9]+)(_(?P<lb>[01])){,1}\\.png'
    rg = re.compile(pf_form)
    dirs = os.listdir(config.out_dir)

    for d in dirs:
        files = os.listdir(os.path.join(config.out_dir,d))
        files = list(filter(lambda i:i.endswith('png'),files))
        label = open(os.path.join(config.out_dir,d,'label.txt'),'w')
        for f in files:
            match = rg.match(f)
            if not match:
                continue
            line = '{} {} {} {} {}\n'.format(f,match.group('lb'),match.group('tcga'),match.group('x'),match.group('y'))
            label.write(line)

def _append_label(config):
    """
    This should be run after process_al_metadata
    """
    dirs = os.listdir(config.out_dir)

    labels = {}
    for d in dirs:
        if not d in labels:
            ol = open(os.path.join(config.cp_orig,d,'label.txt'),'r')
            labels[d] = {}
            for line in ol.readlines():
                fields = line.strip().split(' ')
                labels[d][fields[0]] = fields[1] if int(fields[1]) >= 0 else '0'
            ol.close()
        move_to = os.path.join(config.out_dir,d)
        for i in os.listdir(move_to):
            if i.endswith('png'):
                nn = i.split('.')
                nn[0] += '_{}'.format(labels[d][i])
                shutil.move(os.path.join(move_to,i),os.path.join(move_to,'.'.join(nn)))


def _dataset_wsi_metadata(cache_file,wsis,pos_patches,title="DATASET"):
    #Generate dataset stats
    total_patches = 0
    discarded = 0
    total_pos = 0
    ds_wsis = {}

    print("\n\n"+" "*10+"{} PATCHES STATISTICS".format(title))
    if not cache_file is None:
        with open(cache_file,'rb') as fd:
            dt = pickle.load(fd)
        if len(dt) == 3:
            X,Y,_ = dt
        else:
            X,Y,_,_ = dt
        ac_patches = len(X)
        for ic in range(ac_patches):
            img = X[ic]
            label = Y[ic]
            
            if hasattr(img,'getOrigin'):
                origin = img.getOrigin()
            elif hasattr(img,'_origin'):
                origin = img._origin
            else:
                print("Image has no origin information: {}".format(img.getPath()))
                continue
            
            if img.getCoord() is None:
                discarded += 1
                #continue
                
            if origin.startswith('log.'):
                origin = origin.split('.')[1]
                
            if origin in ds_wsis:
                ds_wsis[origin][0].append(img)
                ds_wsis[origin][1].append(label)
            else:
                c_type = os.path.basename(os.path.dirname(img.getPath()))
                ds_wsis[origin] = ([img],[label],c_type)
        

        cancer_type = {}
        for s in ds_wsis:
            n_patches = len(ds_wsis[s][0])
            labels = np.asarray(ds_wsis[s][1])
            cancer_type.setdefault(ds_wsis[s][2],[])
            cancer_type[ds_wsis[s][2]].append(s)
            
            #Count positive patches:
            unique,count = np.unique(labels,return_counts=True)
            l_count = dict(zip(unique,count))
            if 1 in l_count:
                if s in pos_patches:
                    pos_patches[s].append(l_count[1])
                    total_pos += l_count[1]
                else:
                    pos_patches[s] = [0,l_count[1]]
                    total_pos += l_count[1]
            else:
                if s in pos_patches:
                    pos_patches[s].append(0)
                else:
                    pos_patches[s] = [0,0]
            
            print("******   {} ({} total patches)  *******".format(s,n_patches))
            print("Positive patches: {} ({:2.2f}%)".format(pos_patches[s][1],100*pos_patches[s][1]/n_patches))
            if pos_patches[s][1] > 0:
                print("Positive patches acquired from this WSI: {} ({:2.2f}%)".format(pos_patches[s][0],100*pos_patches[s][0]/pos_patches[s][1]))
            else:
                print("Positive patches acquired from this WSI: {} (0.0%)".format(pos_patches[s][0]))
            if config.nc > 0:
                features = []
                for p in range(n_patches):
                    if not ds_wsis[s][0][p].getCoord() is None:
                        features.append(ds_wsis[s][0][p].getCoord())
                features = np.asarray(features)
                if features.shape[0] > config.minp:
                    km = KMeans(n_clusters = config.nc, init='k-means++',n_jobs=2).fit(features)
                    _process_wsi_cluster(km,s,ds_wsis[s][0],config)
        print("-----------------------------------------------------")
        print("Total patches in dataset: {}".format(ac_patches))
        print("Total of positive patches in dataset: {} ({:2.2f}%)".format(total_pos,100*total_pos/ac_patches))
        print("WSIs in dataset: {}".format(len(ds_wsis)))
        print("WSIS by cancer type:\n")
        for ct in cancer_type:
            print("{} ({} WSIs):\n{}".format(ct,len(cancer_type[ct]),"\n".join(cancer_type[ct])))
        print("-----------------------------------------------------")
        
        if config.comb_wsi:
            _combine_acquired_ds(wsis,ds_wsis)

        return ds_wsis


def _wsi_check_test(wsis,testf,dsf,question):

    def _run(t_x,t_y,wsis,question,title):
        ac_patches = len(t_x)
        ts_wsis = {}
        discarded = 0
        cancer_t = {}
        for ic in range(ac_patches):
            img = t_x[ic]
            label = t_y[ic]
            
            if hasattr(img,'getOrigin'):
                origin = img.getOrigin()
            elif hasattr(img,'_origin'):
                origin = img._origin
            else:
                print("Image has no origin information: {}".format(img.getPath()))
                continue
            
            if img.getCoord() is None:
                discarded += 1
                #continue
                
            if origin.startswith('log.'):
                origin = origin.split('.')[1]
                
            if origin in ts_wsis:
                ts_wsis[origin][0].append(img)
                ts_wsis[origin][1].append(label)
            else:
                c_type = os.path.basename(os.path.dirname(img.getPath()))
                cancer_t.setdefault(c_type,[])
                cancer_t[c_type].append(origin)
                ts_wsis[origin] = ([img],[label],c_type)

        #Slides in test set:
        tkeys = set(cancer_t.keys())
        qkeys = set([wsis[w][2] for w in wsis])
        
        print("\n\nCancer tissues common to {} and {}: ".format(question,title),end='')
        print(", ".join([k for k in tkeys.intersection(qkeys)]))
        print("\n\nCancer tissues present in {} and NOT in {}: ".format(title,question),end='')
        print(", ".join([k for k in tkeys.difference(qkeys)]))

        return ts_wsis
        
    #Main logic
    if testf is None or not os.path.isfile(testf):
        return None

    with open(testf,'rb') as tfd:
        fids,sids = pickle.load(tfd)

    with open(dsf,'rb') as dfd:
        X,Y,_ = pickle.load(dfd)
        X,Y = np.asarray(X),np.asarray(Y)
        
    t_x,t_y = X[fids],Y[fids]
    s_x,s_y = X[sids],Y[sids]

    del(X)
    del(Y)
    
    ts_wsis = _run(t_x,t_y,wsis,question,"TEST")
    st_wsis = _run(s_x,s_y,wsis,question, "SAMPLED TEST")

    print("\n\nTest set is composed of slides:\n - {}".format(", ".join([k for k in ts_wsis])))
    print("\n\nSampled from test set is composed of slides:\n - {}".format(", ".join([k for k in st_wsis])))
    
def process_wsi_metadata(config):
    """
    Metadata should contain information about the WSI that originated the patch
    """
    ac_imgs = _process_al_metadata(config)
    
    wsis = {}
    acquisitions = {}
    total_patches = 0
    total_pos = 0
    discarded = 0
    positives = []

    if not ac_imgs:
        print("No metadata to analyze...quiting.")
        sys.exit(1)
        
    if config.ac_n is None or config.save:
        #Use all data if specific acquisitions were not defined
        config.ac_n = list(ac_imgs.keys())
        config.ac_n.sort()

    if config.save and config.ac_save is None:
        config.ac_save = [min(config.ac_n),max(config.ac_n)]

    
    for k in config.ac_n:
        print("In acquisition {}:\n".format(k)) 
        acquisitions[k] = {}
        
        if not k in ac_imgs:
            continue
            
        ac_patches = len(ac_imgs[k][0])

        if ac_patches == 0:
            continue
        
        total_patches += ac_patches
        pos_count = 0
        for ic in range(ac_patches):
            img = ac_imgs[k][0][ic]
            label = ac_imgs[k][1][ic]

            if label > 0:
                pos_count += 1
                
            if hasattr(img,'getOrigin'):
                origin = img.getOrigin()
            elif hasattr(img,'_origin'):
                origin = img._origin
            else:
                print("Image has no origin information: {}".format(img.getPath()))
                continue
            
            if img.getCoord() is None:
                discarded += 1
                #continue

            if origin.startswith('log.'):
                origin = origin.split('.')[1]
            
            if origin in wsis:
                wsis[origin][0].append(img)
                wsis[origin][1].append(label)
            else:
                c_type = os.path.basename(os.path.dirname(img.getPath()))
                wsis[origin] = ([img],[label],c_type)
                
            if origin in acquisitions[k]:
                acquisitions[k][origin].append(img)
            else:
                acquisitions[k][origin] = [img]

        positives.append(pos_count)
        for w in acquisitions[k]:
            coords = [str(p.getCoord()) for p in acquisitions[k][w]]
            print(' '*3 + '**{} ({} patches):{}'.format(w,len(coords),','.join(coords)))
        print("   Positive patches acquired: {} ({:2.2f})".format(pos_count,pos_count/ac_patches))
        
    print("{} patches were disregarded for not having coordinate data".format(discarded))

    if config.save:
        _save_wsi_stats(acquisitions,config.sdir,config.ac_save)

    if config.info:
        print_wsi_metadata(wsis,config,total_patches,total_pos)
        
    return wsis,acquisitions

def print_wsi_metadata(wsis,config,total_patches,total_pos):
    print("\n"+" "*10+"ACQUIRED PATCHES STATISTICS\n\n")
    #This dict will store, for each WSI, [#positive patches acquired, #total of positive patches]
    pos_patches = {}
    wsi_means = []
    for s in wsis:
        n_patches = len(wsis[s][0])
        labels = np.asarray(wsis[s][1])
        
        #Count positive patches:
        unique,count = np.unique(labels,return_counts=True)
        l_count = dict(zip(unique,count))
        if 1 in l_count:
            pos_patches[s] = [l_count[1]]
            total_pos += l_count[1]
        else:
            pos_patches[s] = [0]

        print("******   {} ({} total patches)  *******".format(s,n_patches))
        print("Positive patches acquired: {} ({:2.2f}%)".format(pos_patches[s][0],100*pos_patches[s][0]/n_patches))
        if config.nc > 0:
            features = []
            for p in range(n_patches):
                if not wsis[s][0][p].getCoord() is None:
                    features.append(wsis[s][0][p].getCoord())
            features = np.asarray(features)
            if features.shape[0] > config.minp:
                km = KMeans(n_clusters = config.nc, init='k-means++',n_jobs=2).fit(features)
                wsi_means.append(_process_wsi_cluster(km,s,wsis[s][0],config))
                

    print("-----------------------------------------------------")
    print("Total of acquired patches: {}".format(total_patches))
    print("Total of positive patches acquired: {} ({:2.2f}%)".format(total_pos,100*total_pos/total_patches))
    print("WSIs used in acquisitions: {}".format(len(wsis)))
    if config.nc > 0:
        print("Acquired patches dispersion around cluster means: {:.1f}".format(np.mean(wsi_means)))


    _dataset_wsi_metadata(config.cache_file,wsis,copy.deepcopy(pos_patches),"DATASET")
    if config.ctest:
        tst_file = os.path.join(config.sdir,"{}-testset.pik".format(config.ds))
        _wsi_check_test(wsis,tst_file,config.cache_file,"ACQUIRED")
        
    sp_file = os.path.join(config.sdir,"{}-sampled_metadata.pik".format(config.ds))
    if os.path.isfile(sp_file):
        sds = _dataset_wsi_metadata(sp_file,wsis,pos_patches,"SAMPLED")
        if config.ctest:
            tst_file = os.path.join(config.sdir,"{}-testset.pik".format(config.ds))
            _wsi_check_test(sds,tst_file,config.cache_file,"POOL")
    
def process_al_metadata(config):
    def change_root(s,d):
        """
        s -> original path
        d -> change location to d
        """
        components = tuple(s.split(os.path.sep)[-2:])
        relative_path = os.path.join(*components)

        return os.path.join(d,relative_path)
    
    if not os.path.isdir(config.out_dir):
        os.makedirs(config.out_dir)

    if config.cp_orig is None:
        print("Define local patches location by setting -orig")
        sys.exit(1)
        
    ac_imgs = _process_al_metadata(config)

    if config.ac_n is None:
        #Use all data if specific acquisitions were not defined
        config.ac_n = list(ac_imgs.keys())
        config.ac_n.sort()    
     
    acquisitions = [ac_imgs[k][0][:config.n] for k in config.ac_n]

    print("# of acquisitions obtained: {} -> ".format(len(acquisitions)),end='')
    print(" ".join([str(len(d)) for d in acquisitions]))
    print("Copying...")

    same_name = {}
    for a in range(len(config.ac_n)):
        ac_path = os.path.join(config.out_dir,str(config.ac_n[a]))
        
        if not os.path.isdir(ac_path) and not config.keep:
            os.mkdir(ac_path)
            
        for k in range(len(acquisitions[a])):
            img = acquisitions[a][k]
            img.setPath(change_root(img.getPath(),config.cp_orig))
            img_name = os.path.basename(img.getPath())
            if not config.keep:
                if img_name in same_name:
                    print("Images with the same name detected: {}\n{}\n{}".format(img_name,img.getPath(),same_name[img_name]))
                else:
                    same_name[img_name] = img.getPath()
                shutil.copy(img.getPath(),os.path.join(ac_path,"UN-{}-{}".format(k,img_name)))
            else:
                cur_dir = os.path.split(os.path.dirname(img.getPath()))[-1]
                copy_to = os.path.join(config.out_dir,cur_dir,"UN-{}-{}".format(k,img_name))
                if not os.path.isdir(copy_to):
                    os.mkdir(copy_to)
                shutil.copy(img.getPath(),copy_to)

    #Extract some test patches
    if config.test > 0:
        ts_imgs = _process_test_images(config)
        copy_to = os.path.join(config.out_dir,'testset')
        if not os.path.isdir(copy_to):
            os.mkdir(copy_to)
        if config.gen_label:
            label = open(os.path.join(config.out_dir,'testset','label.txt'),'w')
        for img in ts_imgs:
            img_path = change_root(img.getPath(),config.cp_orig)
            img_name = os.path.basename(img_path)
            shutil.copy(img_path,os.path.join(copy_to,img_name))
            if config.gen_label:
                im_lb = img_name.split('.')[0].split('_')[1]
                x,y = img.getCoord()
                label.write("{} {} {} {} {}\n".format(img_name,im_lb,img.getOrigin(),x,y))
                
    if config.gen_label:
        print("Generating label files...")
        if config.keep:
            _generate_label_files(config)
        else:
            _generate_label_from_name(config)
    elif config.add_label:
        print("Appending label to file names...")
        _append_label(config)
        
    print("Acquired images copied to output dir.")

def process_cluster_metadata(config):
    def _copy_img(dst_dir,acq,cln,img,img_in,orig=None):
        ac_path = os.path.join(dst_dir,str(acq),str(cln))
        if not os.path.isdir(ac_path):
            os.makedirs(ac_path)
        img_name = os.path.basename(img.getPath())
        img_class = img_name.split('.')[0].split('_')
        if len(img_class) > 1:
            img_class = img_class[1]
        else:
            img_class = 'X'
        if orig is None:
            shutil.copy(img.getPath(),os.path.join(ac_path,'{}_{}.{}'.format(img_in,img_class,img_name.split('.')[1])))
        else:
            subdir = os.path.split(os.path.dirname(img.getPath()))[1]
            orig_img = os.path.join(orig,subdir,os.path.basename(img.getPath()))
            shutil.copy(orig_img,os.path.join(ac_path,'{}_{}.{}'.format(img_in,img_class,img_name.split('.')[1])))
            
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)

    if config.sdir is None or not os.path.isdir(config.sdir):
        print("Directory not found ({}), define -sd otion".format(config.sdir))
        sys.exit(1)

    files = os.listdir(config.sdir)

    acfiles = {}
    for f in files:
        if f.startswith('al-clustermetadata'):
            ac_id = int(f.split('.')[0].split('-')[3][1:])
            acfiles[ac_id] = os.path.join(config.sdir,f)

    ac_imgs = {}
    for k in config.ac_n:
        if not k in acfiles:
            print("Requested acquisition ({}) is not present".format(k))
            return None

        with open(acfiles[k],'rb') as fd:
            pool,un_clusters,un_indexes = pickle.load(fd)
    
        for cln in range(len(un_clusters)):
            ind = np.asarray(un_clusters[cln])
            print("Cluster {}, # of items: {}".format(cln,ind.shape[0]))
            posa = np.ndarray(shape=(1,),dtype=np.int32)
            for ii in range(min(ind.shape[0],config.n)):
                if ii == 0:
                    posa[0] = np.where(un_indexes == ind[ii])[0]
                else:
                    posa = np.hstack((posa,np.where(un_indexes == ind[ii])[0]))
                #Copy image
                _copy_img(config.out_dir,k,cln,pool[0][ind[ii]],posa[ii],config.cp_orig)
            print("Cluster {} first items positions in index array (at most {}): {}".format(cln,config.n,posa))
    

def process_train_set(config):

    trainsets = {}
    for f in config.trainset:
        if not os.path.isfile(f):
            print("File not found: {}".format(f))
            return None
        with open(f,'rb') as fd:
            train,_,_ = pickle.load(fd)
        for i in train[0]:
            if i in trainsets:
                trainsets[i] += 1
            else:
                trainsets[i] = 1

    miss_count = 0
    hit_count = 0
    for j in trainsets:
        if trainsets[j] == 1:
            print("Image {} occurs in only one of the sets".format(j))
            miss_count += 1
        elif trainsets[j] == 2:
            hit_count += 1

    print("{} patches were found in only one set.".format(miss_count))
    print("{} patches were in both sets".format(hit_count))


def process_wsi_plot(config,acqs):
    """
    @Params:
    - acqs <dict>: the output of process_wsi_metadata

    @Saves file:
    acquisition_stats.pik with a tuple: wsis, wsi_means,patch_count
    - wsis <dict>: keys -> slide names; values -> list of patches acquired from the slide
    - wsi_means <dict>: keys -> acq number; values -> dictionary with slides as keys and list of mean distance in pixels to cluster centers
    - patch_count <dict>: keys -> acq number; values -> list of number of patches acquired from each WSI in current acquisition
    """

    wsis = {}
    total_patches = 0
    #Acumulate patches in sequential acquisitions in this list
    wsi_means = {}
    patch_count = {}
    sorted_acq = sorted(list(acqs.keys()))
    for k in sorted_acq:
        wsi_means.setdefault(k,{})
        patch_count[k] = []
        for w in acqs[k]:
            wsis.setdefault(w,[])
            patch_count[k].append(len(acqs[k][w]))
            wsis[w].extend(acqs[k][w])
        total_patches += sum(patch_count[k])
        print("Patches to cluster: {}".format(total_patches))
        for s in wsis:
            n_patches = len(wsis[s])
            wsi_means[k].setdefault(s,0.0)
            if config.nc > 0:
                features = []
                for p in range(n_patches):
                    if not wsis[s][p].getCoord() is None:
                        features.append(wsis[s][p].getCoord())
                features = np.asarray(features)
                if features.shape[0] > config.minp:
                    km = KMeans(n_clusters = config.nc, init='k-means++',n_jobs=2).fit(features)
                    wsi_means[k][s] = _process_wsi_cluster(km,s,wsis[s],config)


    with open(os.path.join(config.sdir,'acquisition_stats_NC-{}.pik'.format(config.nc)),'wb') as fd:
        pickle.dump((wsis,wsi_means,patch_count),fd)
    slides = wsis.keys()
    print("Slides ({}): {}".format(len(slides),"\n".join([" - {}".format(sn) for sn in slides])))


if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    parser.add_argument('--meta', dest='meta', action='store_true', 
        help='Acquire images from ALTrainer metadata.', default=False)
    parser.add_argument('--cluster', dest='cluster', action='store_true', 
        help='Acquire from KM clustering metadata.', default=False)
    parser.add_argument('--wsi', dest='wsi', action='store_true', 
        help='Identify the patches of each WSI in the acquisitions.', default=False)
    parser.add_argument('--train_set', dest='trainset', type=str, nargs=2,
        help='Check if the training sets of two experiments are the same.', default=None)
        
    parser.add_argument('-ac', dest='ac_n', nargs='+', type=int,
        help='Acquisitions to obtain images.', default=None, required=False)
    parser.add_argument('-sd', dest='sdir', type=str,default=None, 
        help='Experiment result path.')
    parser.add_argument('-ds', dest='ds', type=str,default='CellRep', 
        help='Dataset used in experiments.')        
    parser.add_argument('-od', dest='out_dir', type=str,default='img_grab', 
        help='Save selected images to this directory.')
    parser.add_argument('-n', dest='n', type=int, 
        help='Grab this many images. If cluster, grab this many images per cluster (Default: 200)', default=200,required=False)
    parser.add_argument('-orig', dest='cp_orig', type=str, nargs='?', default=None, const='../data/lym_cnn_training_data',
        help='Copy original images instead of the normalized ones. Define location of the originals.')
    parser.add_argument('-net', dest='net', type=str,default=None, 
        help='Network name for metadata analysis.')
    parser.add_argument('-minp', dest='minp', type=int, 
        help='Cluster patches if WSI had this number of selected patches', default=10,required=False)
    parser.add_argument('-nc', dest='nc', type=int, 
        help='Number of clusters to group patches of each WSI', default=0,required=False)
    parser.add_argument('-radius', dest='radius', type=int, 
        help='Radius in pixels to group patches in each cluster.', default=300,required=False)
    parser.add_argument('-cache_file', dest='cache_file', type=str,default=None, 
        help='Dataset metadata for WSI statistics.')
    parser.add_argument('-save', action='store_true', dest='save',
        help='Saves generated statistics to file.',default=False)    
    parser.add_argument('-ac_save', dest='ac_save', nargs='+', type=int,
        help='Save image statistics from this acquisitions.', default=None, required=False)
    parser.add_argument('-keep', action='store_true', dest='keep',
        help='Keep original dataset structure when copying tiles.',default=False)    
    parser.add_argument('-gen_label', action='store_true', dest='gen_label',
        help='Generate label file for extracted patches.',default=False)
    parser.add_argument('-add_label', action='store_true', dest='add_label',
        help='Append label to file names as used in quip_classification.',default=False)
    parser.add_argument('-comb_wsi', action='store_true', dest='comb_wsi',
        help='Check patches acquired from a WSI in comparison to total WSI patches available.',default=False)
    parser.add_argument('-pinit', action='store_true', dest='pinit',
        help='Parse and extract initial training set, if it exists.',default=False)
    parser.add_argument('-ctest', action='store_true', dest='ctest',
        help='Check acquired and sampled sets against test set.',default=False)
    parser.add_argument('-ni', action='store_false', dest='info',
        help='Do NOT display info.',default=True)
    parser.add_argument('-bplot', action='store_true', dest='bplot',
        help='Save plotable data.',default=False)
    parser.add_argument('-test', dest='test', type=int, 
        help='Grab this many images from test set (Default: 0)', default=0,required=False)
    parser.add_argument('-db', action='store_true', dest='debug',
        help='Execute debuging actions for each run mode.',default=False)
    
    config, unparsed = parser.parse_known_args()

    if config.meta:
        process_al_metadata(config)
    elif config.cluster:
        process_cluster_metadata(config)
    elif config.wsi:
        wsis,acqs = process_wsi_metadata(config)
        if config.bplot:
            process_wsi_plot(config,acqs)
    elif not config.trainset is None:
        process_train_set(config)
    else:
        print("You should choose between ALTrainer metadata (--meta), KM metadat (--cluster) or WSI metadata (--wsi)")
