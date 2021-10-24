#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import sys
import os
import argparse
import random
import shutil


if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Dataset manipulation.')

    parser.add_argument('-sd', dest='sdir', type=str,default=None, required=True,
        help='Source directory (path to patches).')
    parser.add_argument('-out', dest='out', type=str,default='data', required=False,
        help='Destination directory (save here).')
    parser.add_argument('-type', dest='type', type=str, nargs='+',
        help='Cancer types: \n \
        blca; \n \
        brca; \n \
        cesc; \n \
        coad; \n \
        paad; \n \
        prad; \n \
        read; \n \
        skcm; \n \
        stad; \n \
        ucec; \n',
       choices=['blca','brca','cesc','coad','paad','prad','read','skcm','stad','ucec'],default=None)
    parser.add_argument('-ct', dest='ct', nargs='+', type=int, 
        help='Grab this many patches of each type.', default=100,required=False)    

    config, unparsed = parser.parse_known_args()

    if not os.path.isdir(config.sdir):
        print("Directory not found: {}".format(config.sdir))
        sys.exit(1)

    if not os.path.isdir(config.out):
        os.mkdir(config.out)

    dirs = list(filter(lambda k: os.path.isdir(os.path.join(config.sdir,k)),os.listdir(config.sdir)))
    if config.type is None:
        cancer = {c:None for c in dirs}
    else:
        cancer = {c:None for c in config.type}

    i = 0
    for d in dirs:
        if d in cancer:
            if isinstance(config.ct,int):
                count = config.ct
            else:
                count = config.ct[i] if len(config.ct) == len(dirs) else config.ct[0]

            src = os.path.join(config.sdir,d)
            cancer[d] = list(filter(lambda p: p.endswith('.png'),os.listdir(src)))
            if len(cancer[d]) > count:
                cancer[d] = random.sample(cancer[d],k=count)
            else:
                print("Requested number of samples unvailable in {} ({})".format(d,len(cancer[d])))
            with open(os.path.join(src,'label.txt'),'r') as fd:
                lines = fd.readlines()

            labels = {}
            for l in lines:
                ls = l.strip().split(' ')
                labels[ls[0]] = ls[1:]
            del(lines)

            dst = os.path.join(config.out,d)
            if not os.path.isdir(dst):
                os.mkdir(dst)

            lb = open(os.path.join(dst,'label.txt'),'w')
            print("Start copy: {} patches from {} to {}".format(count,src,dst))
            for p in cancer[d]:
                shutil.copy(os.path.join(src,p),os.path.join(dst,p))
                lb.write("{} {}\n".format(p," ".join(labels[p])))
            lb.close()
        i += 1
