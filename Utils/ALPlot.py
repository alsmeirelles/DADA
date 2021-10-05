#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

#import matplotlib
#matplotlib.use('macosx')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=1, figsize=(13.5, 6.7), dpi=100, facecolor='w', edgecolor='k')

import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

import datetime
import numpy as np
import os
import re
import sys
import argparse
import scipy.stats

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 21}

mpl.rc('font',**font)

palette = plt.get_cmap('Dark2')
newcolors = np.ones((12,4))
newcolors[0,0:3] = np.array([29/256,39/256,125/256]) #8
newcolors[1,0:3] = np.array([110/256,29/256,5/256]) #9
newcolors[2,0:3] = np.array([0/256,44/256,163/256]) #10
newcolors[3,0:3] = np.array([0/256,5/256,105/256]) #11
newcolors[4,0:3] = np.array([159/256,32/256,54/256]) #12
newcolors[5,0:3] = np.array([76/256,0/256,153/256]) #13
newcolors[6,0:3] = np.array([0.0,0.0,0.0]) #14
newcolors[7,0:3] = np.array([135/256,92/256,116/256]) #15
newcolors[8,0:3] = np.array([81/256,60/256,44/256]) #16
newcolors[9,0:3] = np.array([224/256,44/256,41/256]) #17
newcolors[10,0:3] = np.array([146/256,38/256,115/256]) #18
newcolors[11,0:3] = np.array([176/256,108/256,75/256]) #19
newcolors = np.vstack((palette(np.linspace(0,1,len(palette.colors))),
                           newcolors))
   
palette = ListedColormap(newcolors,name='CustomDark')
del(newcolors)

linestyle = [
    ('solid', (0,())),
    #('loosely dotted',        (0, (1, 10))),
    ('dotted',                (0, (1, 5))),
    ('densely dotted',        (0, (1, 1))),
    
    #('loosely dashed',        (0, (5, 10))),
    ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),
    
    #('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (3, 3, 1, 3))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),

    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 8, 1, 8, 1, 8))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

#markers = ['','*','+','x','^','.','2','v']
markers = ['*','+','x','^','.','2','v','s','p','D',8,9,10,'4','']
patterns = ["+", ".", "/" , "\\" , "|" , "-","o"]
lang = 'pt'

class Plotter(object):

    def __init__(self,data=None, path=None, ncols=2):
        if not path is None and os.path.isdir(path):
            self.path = path
        else:
            self.path = None

        self._nX = None
        self._yIDX = None
        self._ncols = ncols

    def draw_uncertainty(self,data,xticks,spread=1,title='',subfig=False,sub_acq=None):
        """
        Data: list of tuples in the form (indexes,uncertainties), where both indexes and uncertainties are numpy
        arrays.
        """
        mpl.rcParams['agg.path.chunksize'] = 1000000
        max_points = 20000
        maxu = 0.0
        plots = None
        ax = plt.gca()
        if xticks is None:
            xticks = list(range(len(data)+spread))
        if subfig and not sub_acq is None:
            sub_acq = sub_acq[0] if isinstance(sub_acq,list) else sub_acq
        for k in range(len(data)):
            if subfig:
                indexes,unc,clusters,_ = data[k]
            elif len(data[k]) == 3:
                indexes,unc,_ = data[k]
            else:
                indexes,unc = data[k]
            selected = unc[indexes]
            unselected = np.delete(unc,indexes)
            smean = np.mean(selected)
            usmean = np.mean(unselected)
            print("Acquisition # {}".format(k))
            print("Selected patches mean uncertainty ({}): {:.3f}".format(selected.shape[0],smean))
            print("Unselected patches mean uncertainty ({}): {:.3f}\n****".format(unselected.shape[0],np.mean(unselected)))
            mk = np.max(unc)
            if mk > maxu:
                maxu = mk

            n_points = min(max_points,unselected.shape[0])
            pl1 = ax.plot(np.random.rand(n_points)*spread+xticks[k],np.random.choice(unselected,n_points,replace=False),
                         'oy',markersize=2,alpha=0.8)                
            pl2 = ax.plot(np.random.rand(selected.shape[0])*spread+xticks[k],selected, 'oc',alpha=0.3,markersize=6)
            pl3 = ax.plot(0.5*spread+xticks[k],smean,'or',alpha=0.6,markersize=8)
            pl3e = ax.errorbar(0.5*spread+xticks[k],smean, yerr=np.std(selected), fmt='none',color='r',alpha=0.6,markersize=8,zorder=10)
            pl4 = ax.plot(0.5*spread+xticks[k],usmean,'og',alpha=0.6,markersize=10)
            pl4e = ax.errorbar(0.5*spread+xticks[k],usmean,yerr=np.std(unselected),fmt='none',color='g',alpha=0.6,markersize=10,zorder=10)
            
            plots = [pl1,pl2,pl3,pl4]
            cmax = 0.0
            if k == sub_acq and subfig:
                from mpl_toolkits.axes_grid.inset_locator import inset_axes
                palette = plt.get_cmap('tab20')
                inset_ax = inset_axes(ax, 
                    width="51%", # width = 52% of parent_bbox
                    height="35%") # height : 35%

                for c in clusters:
                    inset_ax.plot(np.random.rand(clusters[c].shape[0])+c,clusters[c], 'o',color=palette(c),alpha=0.6,markersize=2)
                    lmax = np.max(clusters[c])
                    if cmax < lmax:
                        cmax = lmax
                if lang == 'en':
                    inset_ax.text(0,0.97*cmax,"Acquisition #{}".format(sub_acq),fontsize=8)
                else:
                    inset_ax.text(0,0.97*cmax,"Aquisição #{}".format(sub_acq),fontsize=8)
                inset_ax.set_xlabel("Cluster #")
                #inset_ax.set_xticks(list(clusters.keys()))
                inset_ax.xaxis.set_tick_params(rotation=-35)
                inset_ax.set_yticks(np.round(np.linspace(0.0, cmax, 5),2))

        if lang == 'en':
            labels = ['Uns. patches','Sel. patches','Sel. mean','Uns. mean']
        else:
            labels = ['Patches NS','Patches Sel.','Média Sel.','Média NS']
        ncol = 1 if subfig else self._ncols
        ax.legend(plots,labels=labels,loc=2,ncol=ncol,prop=dict(weight='bold'),fontsize=14)
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(0.0, maxu+0.1, 0.05))
        ax.set_title(title, loc='center', fontsize=12, fontweight=0, pad=2.0, color='orange')
        if lang == 'en':
            ax.set_xlabel("Acquisition #")
            ax.set_ylabel("Uncertainty")
        else:
            ax.set_xlabel("Aquisição #")
            ax.set_ylabel("Incerteza")

        plt.tight_layout()
        ax.grid(True)
        plt.show()

    def draw_cluster_distribution(self,data,spread=1.0,title=''):
        """
        Data is returned by parseSlurm. 'labels' key refers to a dictionary with cluster number as keys. Values are lists:
        [(x,y),(...),...] -> x: number of negative samples; y: number of positive samples
        """
        import scipy.stats as st
        
        clusters = data['labels']
        mpl.rcParams['agg.path.chunksize'] = 1000000
        plots = []
        pl1,pl2 = None,None
        xticks = range(0,len(clusters.keys())+1)

        X,Y = [],[]
        a = []
        py = []
        
        #Number of clusters in each acquisition should be the same
        cn = len(clusters[0])
        for acq in clusters:

            #Total elements per acquisition
            total = sum([clusters[acq][i][0] + clusters[acq][i][1] for i in clusters[acq]])

            #Scatter sizes
            a.extend([1200*((clusters[acq][i][0]+clusters[acq][i][1])/total) for i in clusters[acq]])

            X.extend(np.random.rand(cn)*spread+xticks[acq])
            #Percentage of positives
            py.extend([clusters[acq][c][1]/(clusters[acq][c][0]+clusters[acq][c][1]) for c in clusters[acq]])

            cur_y = [np.mean(data['unc'][acq][x]) for x in range(cn)]
            Y.extend(cur_y)
            umean = np.mean(cur_y)
            plt.plot(list(np.arange(xticks[acq],xticks[acq]+spread,0.1)),[umean]*10,'g--',markersize=3)

        a = np.asarray(a)
        py = np.asarray(py)
        X = np.asarray(X)
        Y = np.asarray(Y)
        colors = np.zeros((len(X),3))
        #Red scale
        colors[:,0] = 1 - py 
        #Green scale
        colors[:,1] = py
        #Blue scale
        colors[:,2] = py
        pl1 = plt.scatter(X,Y,s=a,c=colors,alpha=0.5,label='Positive')
                        
        xticks = list(range(0,len(clusters)+spread,spread))

        fig = plt.gcf()
        maxy = max(Y)
        yticks = list(np.arange(0.0,maxy,0.01))
        plt.axis([-0.5,xticks[-1],0.01,maxy+0.001])
        sort_idx = py.argsort()
        colors = colors[sort_idx]
        #Drop first 200 colors, as they tend to be very similar
        colors = colors[200:]
        final_colors = np.array([(0.0,0.8,0.8),(0.0,0.85,0.85),(0.0,0.9,0.9),(0.0,0.95,0.95),(0.0,1.0,1.0)])
        colors = np.vstack((colors,final_colors))
        cmap = ListedColormap(colors,name='Heatm')
        if lang == 'en':
            fig.colorbar(mpl.cm.ScalarMappable(norm=None,cmap=cmap),label='Positive %',pad=0.1,fraction=0.05)
        else:
            fig.colorbar(mpl.cm.ScalarMappable(norm=None,cmap=cmap),label='% Positivo',pad=0.1,fraction=0.05)

        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        if lang == 'en':
            plt.xlabel("Acquisition #")
            plt.ylabel("Uncertainty mean")
        else:
            plt.xlabel("Aquisição #")
            plt.ylabel("Incerteza média")

        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def draw_time_stats(self,data,xticks,auc_only,metrics,labels=None,spread=1,title='',colors=None,yscale=False,maxy=0.0,miny=0.0,merge=False):
        """
        @param data <list>: a list as returned by calculate_stats
        """
        import matplotlib.patches as mpatches
        
        color = 0
        line = 0
        marker = 0
        plots = []
        lbcount = 0
        xmax = 0
        xmin = np.inf
        tmax = 0
        tmin = np.inf
        metric_patches = []
        tset = None
        nexp = len(data)
        hatch_color = 'black'
        
        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        ax = plt.subplot(111)

        for d in data:
            nmetrics = len(metrics)
            if nmetrics > 1:
                print("Stacked bar plot selected. Using the first metric provided ({}) as reference".format(metrics[0]))
            metric = metrics[0]
            if not metric in data[d]:
                print("Requested metric not available in experiment ({}): {}".format(d,metric))
                return None
            else:
                x_data,tdata,ci,y_label,color = data[d][metric]

            if not colors is None and colors[d] >= 0:
                color = colors[d]
            elif color < 0:
                color = 0

            if labels is None:
                lb = d
            else:
                lb = labels[d]
                lbcount = d
                
            line = color%len(linestyle)
            marker = color%len(markers)
            color = color % len(palette.colors)

            #Do nothing if requested metric is not available
            if tdata.shape[0] == 0:
                lbcount += 1
                continue

            #Check current trainset
            if self._nX is None:
                self._nX = x_data
                tset = x_data
            else:
                if len(self._nX) != len(x_data) and not merge:
                    self._yIDX = np.in1d(x_data,self._nX)
                    tset = x_data[self._yIDX]
                    if len(tdata) < len(self._yIDX):
                        samples = min(len(x_data),len(tdata))
                        self._yIDX = self._yIDX[:samples]
                        #tset = tset[:samples]
                    tdata = tdata[self._yIDX]
                else:
                    tset = x_data
                    
            #Repeat last point if needed
            if tset.shape[0] > tdata.shape[0]:
                print("Shape mismatch:\n Trainset: {}; {}:{}".format(tset.shape,tdata.shape,metric))
                tdata = np.hstack((tdata,tdata[-1:]))

            #Set bar x positions
            bar_x = tset + (lbcount-(nexp/2))*40
            print("{}:\n - X:{};\n - Y:{}".format(lb,bar_x,tdata))
            
            #Check lower and upper values to axis scaling
            local_max = np.max(x_data)
            local_min = np.min(x_data)
            xmax = local_max if local_max > xmax else xmax
            xmin = local_min if local_min < xmin else xmin
            local_max = np.max(tdata)
            local_min = np.min(tdata)
            tmax = local_max if local_max > tmax else tmax
            tmin = local_min if local_min < tmin else tmin

            if not self._yIDX is None:
                print("Plotting only predefined FN points")

            if nmetrics > 1:
                bottom = np.zeros(len(tdata))
                colorf = np.asarray((1.4,1.4,1.4,0.85))
                for m in range(1,nmetrics):
                    if m < nmetrics - 1:
                        bottom += data[k][metrics[m+1]]
                        colorf *= 0.75
                    mcolor = np.clip(np.asarray(palette(color)) * colorf,0.0,1.0)
                    #Repeat last point if needed
                    _,bar_y,_,_,_ = data[d][metrics[m]]
                    if bar_x.shape[0] > bar_y.shape[0]:
                        bar_y = np.hstack((bar_y,bar_y[-1:]))
                    else:
                        bar_y = bar_y[:bar_x.shape[0]]
                    plt.bar(bar_x,bar_y,width=40,color=mcolor,bottom=bottom,edgecolor=hatch_color,hatch=patterns[color%len(patterns)],linewidth=2)
                plt.bar(bar_x,tdata-bar_y,width=40,color=palette(color),label=lb,bottom=bar_y,edgecolor=hatch_color,hatch=patterns[color%len(patterns)],linewidth=2)
            else:
                plt.bar(bar_x,tdata,width=40,color=palette(color),label=lb,edgecolor=hatch_color,hatch=patterns[color%len(patterns)])
            metric_patches.append(mpatches.Patch(facecolor=palette(color),label=lb,hatch=patterns[color%len(patterns)],edgecolor=hatch_color))
            
            formatter = FuncFormatter(self.format_func)
            ax.yaxis.set_major_formatter(formatter)

        if not metrics is None:
            plt.legend(handles=metric_patches,loc=2,ncol=self._ncols,prop=dict(weight='bold'))
        else:
            plt.legend(loc=0,ncol=2,labels=config.labels,prop=dict(weight='bold'))
            
        if xmax > 1000:
            plt.xticks(rotation=30)

        #Defining ticks
        axis_t = []
        xlim = xmax+(0.7*xticks)
        mtick = np.arange(xmin, xlim, xticks)
        axis_t.extend([mtick.min()*0.6,xlim])
        plt.xticks(mtick)

        if not metrics is None:
            mtick = 1.1*tmax if maxy == 0.0 else maxy
            ticks = np.linspace(0.0, mtick,7)
            np.around(ticks,2,ticks)
            axis_t.extend([ticks.min(),ticks.max()])
            plt.yticks(ticks)
        else:
            if scale or maxy == 0.0:
                ticks = np.linspace(min(0.6,0.9*tmin), tmax+0.1, 8)
                np.round(ticks,2,ticks)
            else:
                ticks = np.arange(0.65,maxy,0.05)
                np.round(ticks,2,ticks)
            axis_t.extend([ticks.min(),ticks.max()])
            plt.yticks(ticks)

        plt.axis(axis_t)
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        if lang == 'en':
            plt.xlabel("Training set size")
        else:
            plt.xlabel("Conjunto de treinamento")

        if not metrics is None:
            plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25,axis='y')
            if metric == 'time':
                if lang == 'en':
                    plt.ylabel('AL step time \n(hh:min:sec)')
                else:
                    plt.ylabel('Tempo da iteração AL\n(hh:min:sec)')
            elif metric == 'acqtime':
                if lang == 'en':
                    plt.ylabel('Acquisition step time \n(hh:min:sec)')
                else:
                    plt.ylabel('Tempo de aquisição \n(hh:min:sec)')
            elif metric == 'traintime':
                if lang == 'en':
                    plt.ylabel('Training step time \n(hh:min:sec)')
                else:
                    plt.ylabel('Tempo de treinamento \n(hh:min:sec)')
            elif metric == 'auc':
                plt.ylabel('AUC')

        plt.tight_layout()
        plt.show()            
            
    def draw_stats(self,data,xticks,auc_only,labels=None,spread=1,title='',colors=None,yscale=False,maxy=0.0,miny=0.0):
        """
        @param data <list>: a list as returned by calculate_stats
        """
        color = 0
        line = 0
        marker = 0
        plots = []
        up = 0.0
        low = 1.0
        lbcount = 0
        xmax = 0
        xmin = np.inf
        
        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        if labels is None:
            labels = [l[3] for l in data]
            
        for d in data:
            x_data,y_data,ci,y_label,color = d
            if not colors is None and colors[lbcount] >= 0:
                color = colors[lbcount]
            elif color < 0:
                color = 0

            lbcount += 1
                
            line = color%len(linestyle)
            marker = color%len(markers)
            color = color % len(palette.colors)

            local_max = np.max(x_data)
            local_min = np.min(x_data)
            xmax = local_max if local_max > xmax else xmax
            xmin = local_min if local_min < xmin else xmin
            if auc_only and y_label != 'AUC':
                continue

            c = plt.plot(x_data, y_data, lw = 2.0, marker=markers[marker],linestyle=linestyle[line][1],color=palette(color), alpha = 1)
            plots.append(c)
            # Shade the confidence interval
            if not np.isnan(ci).any():
                low_ci = np.clip(y_data - ci,0.0,1.0)
                upper_ci = np.clip(y_data + ci,0.0,1.0)
                ym = np.max(upper_ci)
                yl = np.min(low_ci)
            else:
                yl = np.min(y_data)
                ym = np.max(y_data)
            up = ym if ym > up else up
            low = yl if yl < low else low
            if not np.isnan(ci).any():
                plt.fill_between(x_data, low_ci, upper_ci, color = palette(color), alpha = 0.4)
            color += 1
            line = (line+1)%len(linestyle)
            marker = color%len(markers)

            if lang == 'en':
                plt.xlabel("Trainset size")
            else:
                plt.xlabel("Conjunto de treinamento")
            plt.ylabel(y_label)
                
        # Label the axes and provide a title
        plt.legend(plots,labels=labels,loc=0,ncol=self._ncols,prop=dict(weight='bold'))
        #ydelta = (1.0 - low)/10
        ydelta = 0.1
        if yscale or (maxy == 0.0 and miny == 0.0):
            yrg = np.clip(np.arange(max(low,0.0), up + ydelta, ydelta),0.0,1.0)
        else:
            yrg = np.clip(np.arange(miny, maxy, ydelta),0.0,1.0)
        np.around(yrg,2,yrg)
        plt.yticks(yrg)
        xrg = np.arange(xmin, xmax+xticks, xticks)
        plt.xticks(xrg)
        plt.axis([min(xrg)-20,max(xrg),min(yrg),max(yrg)])
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        if xmax > 1000:
            plt.xticks(rotation=30)
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
    def draw_data(self,data,title='',xlabel='Trainset size',metric=None,color=0,xtick=200):

        fig = plt.figure(1)
        fig.suptitle(title)
        fig.subplots_adjust(top=0.5)

        if metric is None:
            tdata = data['time']
            metric = 'time'
        elif metric in data:
            tdata = data[metric]
        else:
            tdata = None

        #Train size x Acquisition step time (if that was logged)
        if not tdata is None:
            fig_pos = 211
            if tdata.shape != data['trainset'].shape:
                maxi = min(tdata.shape[0],data['trainset'].shape[0])
            else:
                maxi = max(tdata.shape[0],data['trainset'].shape[0])
        else:
            fig_pos = 110

        if tdata.shape[0] > 0:
            ax = plt.subplot(fig_pos)
            plt.bar(data['trainset'][:maxi],tdata,width=30,color=palette(color),edgecolor='black',hatch=patterns[color%len(patterns)])
            ax.set_xticks(np.arange(data['trainset'][0],data['trainset'].max()+50,xtick))
            plt.axis([data['trainset'][0]-50,data['trainset'].max()+50,0.0,tdata.max()+.2])
            #fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            formatter = FuncFormatter(self.format_func)
            ax.yaxis.set_major_formatter(formatter)
            plt.xlabel(xlabel)
            if metric == 'time':
                plt.ylabel('AL step time \n(hh:min:sec)')
            elif metric == 'acqtime':
                plt.ylabel('Acquisition step time \n(hh:min:sec)')
            elif metric == 'traintime':
                plt.ylabel('Training step time \n(hh:min:sec)')
            elif metric == 'wsave':
                plt.ylabel('Weights saving time \n(hh:min:sec)')
            elif metric == 'wload':
                plt.ylabel('Weights loading time \n(hh:min:sec)')
            else:
                plt.ylabel('Time frame not defined')

        if 'auc' in data and data['auc'].shape[0] > 0:
            #Repeat last point if needed
            if data['trainset'].shape[0] > data['auc'].shape[0]:
                print("Shape mismatch:\n Trainset: {}; Labels:{}".format(data['trainset'].shape,data['auc'].shape))
                data['auc'] = np.hstack((data['auc'],data['auc'][-1:]))            
            plt.subplot(fig_pos+1)
            min_auc = data['auc'].min()
            plt.plot(data['trainset'],data['auc'],'y-')
            plt.axis([data['trainset'][0],data['trainset'][-1:][0],min_auc-0.1,1.0])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(xlabel)
            plt.ylabel('AUC')
        elif data['accuracy'].shape[0] > 0:
            plt.subplot(fig_pos+1)
            min_auc = data['accuracy'].min()
            plt.plot(data['trainset'],data['accuracy'],'y-')
            plt.axis([data['trainset'][0],data['trainset'][-1:][0],min_auc-0.1,1.0])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(xlabel)
            plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def draw_multitime(self,data,title,xtick,metrics,colors=None,labels=None):
        """
        Plot multiple time references for the same experiment. Timing parameters are given by the metrics
        """
        import matplotlib.patches as mpatches
        from matplotlib.legend_handler import HandlerPatch
        
        color = 0
        hatch_color = 'black'
        min_x = []
        max_x = []
        min_t = []
        max_t = []
        metric_patches = []
        lbcount = 0
        nexp = len(metrics)
        
        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        ax = plt.subplot(111)

        for k in metrics:
            if data['trainset'].shape[0] > data[k].shape[0]:
                print("Shape mismatch:\n Trainset: {}; {}:{}".format(data['trainset'].shape,k,data[k].shape))
                data[k] = np.hstack((data[k],data[k][-1:]))

            if labels is None:
                lb = k
            else:
                lb = labels[lbcount]

            if not colors is None and colors[lbcount] >= 0:
                color = colors[lbcount]
            elif 'color' in data:
                color = data['color']                

            bar_x = data['trainset'] + (lbcount-(nexp/2))*30
            plt.bar(bar_x,data[k],width=30,color=palette(color),edgecolor='black',hatch=patterns[color%len(patterns)])
            metric_patches.append(mpatches.Patch(facecolor=palette(color),label=lb,hatch=patterns[color%len(patterns)],edgecolor=hatch_color))
            lbcount += 1
            
        formatter = FuncFormatter(self.format_func)
        ax.yaxis.set_major_formatter(formatter)
        plt.legend(handles=metric_patches,loc=2,ncol=self._ncols,prop=dict(weight='bold'))
        ax.set_xticks(np.arange(data['trainset'].min(), data['trainset'].max()+1, xtick))
        if data['trainset'].max() > 1000:
            plt.setp(ax.get_xticklabels(),rotation=30)
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        if lang == 'en':
            ax.set_xlabel("Training set size")
            ax.set_ylabel("Time\n(hh:mm:ss)")
        else:
            ax.set_xlabel("Conjunto de treinamento")
            ax.set_ylabel("Tempo\n(hh:mm:ss)")
        plt.tight_layout()
        plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25,axis='y')
        plt.show()
        
    def draw_multilabel(self,data,title,xtick,metrics,labels=None,scale=True):
        lbcount = 0
        color = 0

        if not 'trainset' in data:
            print("Trainset should be provided as X axis")
            sys.exit(1)

        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        fig = plt.gcf()
        ax1 = plt.gca()
        ax2 = None

        for k in metrics:
            if k == 'labels' and data[k].shape[0] > 0:
                #Repeat last point if needed
                if data['trainset'].shape[0] > data[k].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; Labels:{}".format(data['trainset'].shape,data[k].shape))
                    data['labels'] = np.hstack((data[k],data[k][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                yd = [100*(data[k][z][0]/data['trainset'][z]) for z in range(data['trainset'].shape[0])]

                #Prepare plot
                if ax2 is None:
                    if lang == 'en':
                        ax1.set_ylabel("% Positive",color=palette(color))
                    else:
                        ax1.set_ylabel("% Positivo",color=palette(color))
                    ax1.plot(data['trainset'],yd, marker='',color=palette(color),linewidth=1,alpha=0.9,label=lb)
                    ax1.set_yticks(np.arange(max(0,min(yd)-10), min(100,10+max(yd)), 5))
                    ax1.tick_params(axis='y', labelcolor=palette(color))
                    ax2 = ax1.twinx()
                else:
                    if lang == 'en':
                        ax2.set_ylabel("% Positive",color=palette(color))
                    else:
                        ax2.set_ylabel("% Positivo",color=palette(color))
                    ax2.plot(data['trainset'],yd, marker='',color=palette(color),linewidth=1,alpha=0.9,label=lb)
                    ax2.tick_params(axis='y', labelcolor=palette(color))
                    ax2.set_yticks(np.arange(max(0,min(yd)-10), min(100,10+max(yd)), 5))
                color += 1
                
            elif (k == 'auc' or k == 'fnauc') and data[k].shape[0] > 0:
                #Repeat last point if needed
                if k == 'auc':
                    xdata = data['trainset']
                else:
                    xdata = data['fntrainset']
                    
                if xdata.shape[0] > data[k].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; AUC:{}".format(xdata.shape,data[k].shape))
                    data[k] = np.hstack((data[k],data[k][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                if ax2 is None:
                    ax1.set_ylabel("AUC",color=palette(color))
                    ax1.plot(xdata,data[k], marker='',color=palette(color),linewidth=1,alpha=0.9,label=lb)
                    ax1.tick_params(axis='y', labelcolor=palette(color))
                    if scale:
                        ax1.set_yticks(np.arange(data[k].min(), 1.0, 0.06))
                    else:
                        ax1.set_yticks(np.arange(0.6, 1.0, 0.05))
                    ax2 = ax1.twinx()
                else:
                    ax2.set_ylabel("AUC",color=palette(color))
                    ax2.plot(xdata,data[k], marker='',color=palette(color),linewidth=1,alpha=0.9,label=lb)
                    ax2.tick_params(axis='y', labelcolor=palette(color))
                    if scale:
                        ax2.set_yticks(np.arange(data[k].min(), 1.0, 0.06))
                    else:
                        ax2.set_yticks(np.arange(0.6, 1.0, 0.05))
                color += 1
                
            elif k == 'accuracy' and data[k].shape[0] > 0:
                #Repeat last point if needed
                if data['trainset'].shape[0] > data[k].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; ACC:{}".format(data['trainset'].shape,data[k].shape))
                    data[k] = np.hstack((data[k],data[k][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                if ax2 is None:
                    ax1.set_ylabel("Accuracy",color=palette(color))
                    plt.plot(data['trainset'],data[k], marker='',color=palette(color),linewidth=1,alpha=0.9,label=k)
                    ax1.tick_params(axis='y', labelcolor=palette(color))
                    ax1.set_yticks(np.arange(data[k].min(), 1.0, 0.06))
                    ax2 = ax1.twinx()
                else:
                    ax2.set_ylabel("Accuracy",color=palette(color))
                    ax2.plot(data['trainset'],data[k], marker='',color=palette(color),linewidth=1,alpha=0.9,label=k)
                    ax2.tick_params(axis='y', labelcolor=palette(color))
                    ax2.set_yticks(np.arange(data[k].min(), 1.0, 0.06))
                    
                color += 1

        fig.legend(bbox_to_anchor=(0.15,0.9),loc=2,ncol=self._ncols,labels=config.labels,prop=dict(weight='bold'))
        if 'fntrainset' in data:
            lx = min(data['trainset'].min(),data['fntrainset'].min())
            mx = max(data['trainset'].max(),data['fntrainset'].max())
            ax1.set_xticks(np.arange(lx,mx+1, xtick))
        else:
            ax1.set_xticks(np.arange(data['trainset'].min(), data['trainset'].max()+1, xtick))
        if data['trainset'].max() > 1000:
            plt.setp(ax1.get_xticklabels(),rotation=30)
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        if lang == 'en':
            ax1.set_xlabel("Training set size")
        else:
            ax1.set_xlabel("Conjunto de treinamento")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def draw_wsi_stat(self,data,title,labels=None,metrics=None,colors=None,ytick=None,xtick=None,err_bar=False):
        from matplotlib.legend import Legend
        
        color = 0
        line = 0
        marker = 0
        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0
        multi_label = False
        
        #experiments = sorted(list(data.keys()))
        experiments = list(data.keys())
        if len(metrics) > 2:
            print("WSI plotting suports only 2 Y axis. X axis is always acquisition #")
            return None
        elif len(metrics) == 2:
            if len(data) > 1:
                print("Multilabel plot will be done only in the first experiment: {}".format(experiments[0]))
                experiments = experiments[:1]
            multi_label = True
        
        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        ax1 = plt.gca()
        ax2 = None

        for z in range(len(data.keys())):
            k = experiments[z]
            if not colors is None and colors[z] >= 0:
                color = colors[z]
            elif 'color' in data[k]:
                color = data[k]['color']
                
            if not multi_label:
                if labels is None:
                    lb = z
                else:
                    lb = labels[z]
                    #lbcount += 1

            line = color%len(linestyle)
            marker = color%len(markers)
            color = color % len(palette.colors)

            X = data[k]['acquired']
            lmax = X.max()
            lmin = X.min()
            max_x = max_x if lmax < max_x else lmax
            min_x = min_x if lmin > min_x else lmin
            
            y_color = '#000000'
            for m in metrics:
                if not m in data[k]:
                    print("No such metric in data: {}".format(m))
                    continue
                if multi_label:
                    if labels is None:
                        lb = z
                    else:
                        lb = labels[z]
                        lbcount += 1
                #Repeat last point if needed
                if X.shape[0] > data[k][m].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; {}:{}".format(data[k]['trainset'].shape,m,data[k][m].shape))
                    data[k][m] = np.hstack((data[k][m],data[k][m][-1:]))
                    
                if multi_label:
                    y_color = palette(color)
                else:
                    max_y = max_y if data[k][m].max() < max_y else data[k][m].max()
                    min_y = min_x if data[k][m].min() > min_y else data[k][m].min()
                    
                if m == 'pmean':
                    maxerr = (data[k]['pdp'] + data[k][m]).max()
                    max_y = max_y if maxerr < max_y else maxerr
                    yticks = np.arange(0.0, max_y, ytick)
                    #This limits lower error bars so it won't go below 0
                    yerr = [data[k]['pdp']-(np.clip(data[k]['pdp']-data[k][m],0.0,maxerr)),np.clip(data[k]['pdp'],0.0,100)]
                    if ax2 is None:
                        if lang == 'en':
                            ax1.set_ylabel("Mean # of patches\nper WSI",color=y_color)
                        else:
                            ax1.set_ylabel("# médio de patches\npor WSI",color=y_color)
                        random_displace = 0.0
                        if not multi_label:
                            random_displace = np.random.rand()
                        pl=ax1.plot(X+random_displace,data[k][m],
                                        marker=markers[marker],color=palette(color),linewidth=2.3,linestyle=linestyle[line][1],alpha=0.9,label=lb)
                        if err_bar:
                            ax1.errorbar(X+random_displace,data[k][m], yerr=yerr, fmt='none',color=palette(color),alpha=0.6,markersize=8,zorder=10)
                        ax1.tick_params(axis='y', labelcolor=y_color)
                        ax1.set_yticks(yticks)
                        if multi_label:
                            ax2 = ax1.twinx()
                    else:
                        if lang == 'en':
                            ax2.set_ylabel("Mean # of patches\nper WSI",color=y_color)
                        else:
                            ax2.set_ylabel("# médio de patches\npor WSI",color=y_color)
                        pl=ax2.plot(X,data[k][m], marker=markers[marker],color=palette(color),linewidth=2.3,linestyle=linestyle[line][1],alpha=0.9,label=lb)
                        if err_bar:
                            ax2.errorbar(X,data[k][m], yerr=yerr, fmt='none',color=palette(color),alpha=0.6,markersize=8,zorder=10)
                        ax2.tick_params(axis='y', labelcolor=y_color)
                        ax2.set_yticks(yticks)
                        
                elif m == 'wsicount':
                    if multi_label:
                        ytick = 10
                    if ax2 is None:
                        ax1.set_ylabel("WSI #",color=y_color)
                        pl=ax1.plot(X,data[k][m], marker=markers[marker],color=palette(color),linewidth=2.3,linestyle=linestyle[line][1],alpha=0.9,label=lb)
                        ax1.tick_params(axis='y', labelcolor=y_color)
                        ax1.set_yticks(np.arange(min(data[k][m].min(),20), data[k][m].max()+ytick, ytick))
                        if multi_label:
                            ax2 = ax1.twinx()
                    else:
                        ax2.set_ylabel("WSI #",color=y_color)
                        pl=ax2.plot(X,data[k][m], marker=markers[marker],color=palette(color),linewidth=2.3,linestyle=linestyle[line][1],alpha=0.9,label=lb)
                        ax2.tick_params(axis='y', labelcolor=y_color)
                        ax2.set_yticks(np.arange(min(data[k][m].min(),20), data[k][m].max()+ytick, ytick))

                elif m == 'wsimeandis':
                    if multi_label:
                        ytick = 200
                    if ax2 is None:
                        if lang == 'en':
                            ax1.set_ylabel("Mean intra-cluster distance (pixels)",color=y_color)
                        else:
                            ax1.set_ylabel("Distância intra-cluster média (pixels)",color=y_color)
                        pl=ax1.plot(X,data[k][m], marker=markers[marker],color=palette(color),linewidth=2.3,
                                        linestyle=linestyle[line][1],alpha=0.9,label=lb,markersize=10)
                        ax1.tick_params(axis='y', labelcolor=y_color)
                        print("Experiment {} ({}): {} - {} elements".format(lb,k,data[k][m],data[k][m].shape[0]))
                        ax1.set_yticks(np.arange(min_y, max_y, ytick))
                        if multi_label:
                            ax2 = ax1.twinx()
                    else:
                        if lang == 'en':
                            ax2.set_ylabel("Mean intra-cluster distance (pixels)",color=y_color)
                        else:
                            ax2.set_ylabel("Distância intra-cluster média (pixels)",color=y_color)
                        pl=ax2.plot(X,data[k][m], marker=markers[marker],color=palette(color),linewidth=2.3,linestyle=linestyle[line][1],alpha=0.9,label=lb)
                        ax2.tick_params(axis='y', labelcolor=y_color)
                        ax2.set_yticks(np.arange(data[k][m].min(), data[k][m].max(), ytick))

                if multi_label:
                    color += 1
        if not (len(labels) == 1 and labels[0] == ""):
            lines,leg_lbl = ax1.get_legend_handles_labels()
            if multi_label:
                lines2,labels2 = ax2.get_legend_handles_labels()
                lines += lines2
            ax1.legend(lines,labels,loc=0,ncol=self._ncols,prop=dict(weight='bold'),columnspacing=0.7)

        ax1.set_xticks(np.arange(min_x, max_x+1,xtick))
        if max_x > 1000:
            plt.setp(ax1.get_xticklabels(),rotation=30)
        ax1.set_title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        if lang == 'en':
            ax1.set_xlabel("Training set size")
        else:
            ax1.set_xlabel("Conjunto de treinamento")
        plt.tight_layout()
        ax1.grid(True)
        plt.show()
        
    def draw_multiline(self,data,title,xtick,**kwargs):
        
        import matplotlib.patches as mpatches
        from matplotlib.legend_handler import HandlerPatch

        labels=kwargs.get('labels',None)
        pos=kwargs.get('pos',False)
        auc=kwargs.get('auc',False)
        other=kwargs.get('other',None)
        colors=kwargs.get('colors',None)
        maxy=kwargs.get('maxy',0.0)
        miny=kwargs.get('miny',0.0)
        scale=kwargs.get('scale',True)
        merge=kwargs.get('merge',False)

        color = 0
        hatch_color = 'white'
        line = 0
        marker = 0
        plotAUC = False
        min_x = []
        max_x = []
        min_y = []
        max_y = []
        min_t = []
        max_t = []
        metric_patches = None
        lbcount = 0
        nexp = len(data)

        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        ax = plt.subplot(111)
        for k in data:
            if pos and data[k]['labels'].shape[0] > 0:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['labels'].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; Labels:{}".format(data[k]['trainset'].shape,data[k]['labels'].shape))
                    data[k]['labels'] = np.hstack((data[k]['labels'],data[k]['labels'][-1:]))

                if not colors is None and colors[lbcount] >= 0:
                    color = colors[lbcount]
                elif 'color' in data[k]:
                    color = data[k]['color']
                line = color%len(linestyle)
                marker = color%len(markers)
                color = color % len(palette.colors)
                
                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                yd = [100*(data[k]['labels'][z][0]/data[k]['trainset'][z]) for z in range(data[k]['trainset'].shape[0])]
                plt.plot(data[k]['trainset'],yd, marker=markers[marker],color=palette(color),linewidth=2.3,linestyle=linestyle[line][1],alpha=0.9,label=lb)
                
                plotAUC = False
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())
                min_y.append(min(yd))
                max_y.append(max(yd))
      
            elif auc and data[k]['auc'].shape[0] > 0:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['auc'].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; AUC:{}".format(data[k]['trainset'].shape,data[k]['auc'].shape))
                    data[k]['auc'] = np.hstack((data[k]['auc'],data[k]['auc'][-1:]))

                if not colors is None and colors[lbcount] >= 0:
                    color = colors[lbcount]
                elif 'color' in data[k]:
                    color = data[k]['color']
                line = color%len(linestyle)
                marker = color%len(markers)
                color = color % len(palette.colors)

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                plt.plot(data[k]['trainset'],data[k]['auc'], marker=markers[marker],color=palette(color),
                             linewidth=2.3,linestyle=linestyle[line][1],alpha=0.9,label=lb,markersize=10)
                if merge and data[k]['fnauc'].shape[0] > 0:
                    if metric_patches is None:
                        metric_patches = []
                    fcolor = np.asarray(palette(color))
                    fcolor[:3,] *= 0.7
                    mcolor = np.clip(fcolor,0.0,1.0)
                    plt.plot(data[k]['fntrainset'],data[k]['fnauc'], marker=markers[marker],color=mcolor,
                             linewidth=2.0,linestyle=linestyle[line][1],alpha=1.0,label=f"{lb}-FN",markersize=8)
                    metric_patches.append(mpatches.Patch(facecolor=palette(color),label=lb,edgecolor=hatch_color))
                    metric_patches.append(mpatches.Patch(facecolor=mcolor,label=f"{lb}-FN",edgecolor=hatch_color))
                    min_x.append(data[k]['fntrainset'].min())
                    max_x.append(data[k]['fntrainset'].max())
                    min_y.append(data[k]['fnauc'].min())
                    max_y.append(data[k]['fnauc'].max())                    
                    print("Feature network evaluations: {}; - FNAUC: {}".format(data[k]['fntrainset'],data[k]['fnauc']))
                    
                plotAUC = True
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())
                min_y.append(data[k]['auc'].min())
                max_y.append(data[k]['auc'].max())
            elif not other is None:
                nmetrics = len(other)
                if nmetrics > 1:
                    print("Stacked bar plot selected. Using the first one provided ({}) as reference".format(other[0]))
                metric = other[0]
                if not metric in data[k]:
                    print("Requested metric not available: {}".format(metric))
                    return None
                else:
                    tdata = data[k][metric]

                #Do nothing if requested metric is not available
                if tdata.shape[0] == 0:
                    lbcount += 1
                    continue

                if metric_patches is None:
                    metric_patches = []

                #Check current trainset
                if 'fntrainset' in data[k]:
                    if not merge:
                        (_,_),(tset,tdata) = self.return_fndata(data[k],metric,merge)
                    else:
                        (tset,_),(_,tdata) = self.return_fndata(data[k],metric,merge)
                    self._nX = data[k]['fntrainset']
                else:
                    if not self._nX is None:
                        self._yIDX = np.in1d(data[k]['trainset'],self._nX)
                        tset = data[k]['trainset'][self._yIDX]
                        tdata = tdata[self._yIDX]
                    else:
                        tset = data[k]['trainset']
                    
                #Repeat last point if needed
                if tset.shape[0] > tdata.shape[0]:
                    print("Shape mismatch:\n Trainset: {}; {}:{}".format(data[k]['trainset'].shape,tdata.shape,metric))
                    tdata = np.hstack((tdata,tdata[-1:]))

                if not colors is None and colors[lbcount] >= 0:
                    color = colors[lbcount]
                elif 'color' in data[k]:
                    color = data[k]['color']
                line = color%len(linestyle)
                marker = color%len(markers)
                color = color % len(palette.colors)
                
                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                bar_x = tset + (lbcount-(nexp/2))*30
                print("{}:\n - X:{};\n - Y:{}".format(lb,bar_x,tdata))

                if not self._yIDX is None:
                    print("Plotting only predefined FN points")

                if nmetrics > 1:
                    bottom = np.zeros(len(tdata))
                    colorf = np.asarray((1.4,1.4,1.4,0.85))
                    for m in range(1,nmetrics):
                        if m < nmetrics - 1:
                            bottom += data[k][other[m+1]]
                            colorf *= 0.75
                        mcolor = np.clip(np.asarray(palette(color)) * colorf,0.0,1.0)
                        #Repeat last point if needed
                        _,bar_y = self.return_fndata(data[k],other[m],merge,self._yIDX)[1]
                        if bar_x.shape[0] > bar_y.shape[0]:
                            bar_y = np.hstack((bar_y,bar_y[-1:]))
                        plt.bar(bar_x,bar_y,width=30,color=mcolor,bottom=bottom,edgecolor=hatch_color,hatch=patterns[color%len(patterns)],linewidth=2)
                    plt.bar(bar_x,tdata-bar_y,width=30,color=palette(color),label=lb,bottom=bar_y,edgecolor=hatch_color,hatch=patterns[color%len(patterns)],linewidth=2)
                else:
                    plt.bar(bar_x,tdata,width=30,color=palette(color),label=lb,edgecolor=hatch_color,hatch=patterns[color%len(patterns)])
                metric_patches.append(mpatches.Patch(facecolor=palette(color),label=lb,hatch=patterns[color%len(patterns)],edgecolor=hatch_color))
                if not metric == 'auc':
                    formatter = FuncFormatter(self.format_func)
                    ax.yaxis.set_major_formatter(formatter)
                min_x.append(tset.min())
                max_x.append(bar_x.max())
                min_t.append(tdata.min())
                max_t.append(tdata.max())
            else:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['accuracy'].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; ACC:{}".format(data[k]['trainset'].shape,data[k]['accuracy'].shape))
                    data[k]['accuracy'] = np.hstack((data[k]['accuracy'],data[k]['accuracy'][-1:]))

                if not colors is None and colors[lbcount] >= 0:
                    color = colors[lbcount]
                elif 'color' in data[k]:
                    color = data[k]['color']
                line = color%len(linestyle)
                marker = color%len(markers)
                color = color % len(palette.colors)
                
                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                    
                plt.plot(data[k]['trainset'],data[k]['accuracy'], marker=markers[marker],color=palette(color),linewidth=2.3,linestyle=linestyle[line][1],alpha=0.9,label=k)
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())                
                min_y.append(data[k]['accuracy'].min())
                max_y.append(data[k]['accuracy'].max())
                print(data[k]['trainset'])

            color += 1
            line = (line+1)%len(linestyle)
            marker = color%len(markers)


        if not other is None or merge:
            plt.legend(handles=metric_patches,loc=0,ncol=self._ncols,prop=dict(weight='bold'))
        else:
            plt.legend(loc=0,ncol=self._ncols,labels=config.labels,prop=dict(weight='bold'))
            
        if max(max_x) > 1000:
            plt.xticks(rotation=30)

        #Defining ticks
        axis_t = []
        xlim = max(max_x)+(0.5*xtick)
        mtick = np.arange(min(min_x), xlim, xtick)
        axis_t.extend([mtick.min()*0.8,xlim])
        plt.xticks(mtick)
        if pos:
            if scale:
                mtick = 1.1*max(max_y) if maxy == 0.0 else maxy
                ticks = np.linspace(min(min_y)-5, mtick, 7)
            else:
                ticks = np.linspace(0.0,maxy,10)
            axis_t.extend([ticks.min(),ticks.max()])
            plt.yticks(ticks)
        elif not other is None:
            mtick = 1.1*max(max_t) if maxy == 0.0 else maxy
            ticks = np.linspace(0.0, mtick,7)
            np.around(ticks,2,ticks)
            axis_t.extend([ticks.min(),ticks.max()])
            plt.yticks(ticks)
        else:
            if scale or (maxy and miny) == 0.0:
                ticks = np.linspace(min(0.6,0.9*min(min_y)), min(max(max_y)+0.1,1.0), 8)
                np.round(ticks,2,ticks)
            elif not scale and maxy > 0.0:
                ticks = np.arange(miny,maxy,0.1)
                np.round(ticks,2,ticks)
            else:
                ticks = np.arange(0.55,maxy,0.05)
                np.round(ticks,2,ticks)
            axis_t.extend([ticks.min(),ticks.max()])
            plt.yticks(ticks)

        plt.axis(axis_t)
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        if lang == 'en':
            plt.xlabel("Training set size")
        else:
            plt.xlabel("Conjunto de treinamento")

        if not other is None:
            metric = other[0]
            plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25,axis='y')
            if metric == 'time':
                if lang == 'en':
                    plt.ylabel('AL step time \n(hh:min:sec)')
                else:
                    plt.ylabel('Tempo da iteração AL \n(hh:min:sec)')
            elif metric == 'acqtime':
                if lang == 'en':
                    plt.ylabel('Acquisition step time \n(hh:min:sec)')
                else:
                    plt.ylabel('Tempo de aquisição \n(hh:min:sec)')
            elif metric == 'traintime':
                if lang == 'en':
                    plt.ylabel('Training step time \n(hh:min:sec)')
                else:
                    plt.ylabel('Tempo de treinamento \n(hh:min:sec)')
            elif metric == 'auc':
                plt.ylabel('AUC')
        else:
            plt.grid(True)
            if plotAUC:
                plt.ylabel("AUC")
            elif pos:
                if lang == 'en':
                    plt.ylabel("% Positive")
                else:
                    plt.ylabel("% Positivo")
            else:
                if lang == 'en':
                    plt.ylabel("Accuracy")
                else:
                    plt.ylabel("Acurácia")

        plt.tight_layout()
        plt.show()

    def plotFromExec(self,data):
        pass

    def parseMetrics(self,data,e_id,metrics):
        split_data = {}
        
        for k in metrics:
            if k == 'pos':
                split_data['labels'] = data['labels']
            else:
                split_data[k] = data[k]
        split_data['trainset'] = data['trainset']
        if 'fntrainset' in data:
            split_data['fntrainset'] = data['fntrainset']

        return split_data

    def parseResults(self,path,al_dirs,n_ids=None,maxx=None,concat=False,wsi=False,ncf=20):

        def parseDirs(path,al_dirs,concat):
            data = {}
            for d in range(len(al_dirs)):
                if isinstance(path,list):
                    d_path = "{0}-{1}".format(path[d],al_dirs[d])
                else:
                    d_path = "{0}-{1}".format(path,al_dirs[d])
                if os.path.isdir(d_path):
                    exp = os.path.basename(d_path)
                    if wsi:
                        data[exp] = self.compileWSIData(d_path,maxx=maxx,concat=concat,ncf=ncf)
                    else:
                        data[exp] = self.parseSlurm(d_path,maxx=maxx,concat=concat)
                else:
                    print("Results dir not found: {}".format(d_path))
            return data

        if isinstance(path,list):
            data = {}
            if n_ids is None:
                return parseDirs(path,al_dirs,concat)
            li = 0
            for k in range(len(path)):
                data.update(parseDirs(path[k],al_dirs[li:li+n_ids[k]],concat))
                li += n_ids[k]
            return data
        else:
            return parseDirs(path,al_dirs,concat)

    def compileWSIData(self,path=None,maxx=None,concat=False,ncf=20,init_train=500):
        import pickle
        
        if path is None and self.path is None:
            print("No directory found")
            sys.exit(1)
        elif path is None:
            path = self.path
        elif isinstance(path,list):
            print("Parse a single file at a time")
            return None

        stats_path = os.path.join(path,'acquisition_stats_NC-{}.pik'.format(ncf))
        if not os.path.isfile(stats_path):
            print("No stats file in folder: {}".format(path))
            sys.exit(1)

        with open(stats_path,'rb') as fd:
            wsis,wsi_means,patch_count = pickle.load(fd)

        data = {}
        data['acquired'] = np.asarray([sum(patch_count[k]) for k in sorted(list(patch_count.keys()))])
        data['trainset'] = np.zeros(data['acquired'].shape[0],dtype=np.int32)
        #Workaround to get trainset size for each acquisition
        csize = init_train
        for i in range(data['acquired'].shape[0]):
            acq = data['acquired'][i]
            data['trainset'][i] = csize
            csize += acq
            data['acquired'][i] += data['acquired'][i-1] if i > 0 else acq
        data['acquired'][0] = sum(patch_count[1])
        data['acqn'] = np.asarray(sorted(list(patch_count.keys())))
        data['pmean'] = np.asarray([np.mean(patch_count[k]) for k in data['acqn']])
        data['pdp'] = np.asarray([np.std(patch_count[k]) for k in data['acqn']])
        data['wsicount'] = np.asarray([len(wsi_means[k]) for k in data['acqn']])

        wsimeandis = []
        for k in data['acqn']:
            distances = []
            for w in wsi_means[k]:
                if wsi_means[k][w] > 0.0:
                    distances.append(wsi_means[k][w])
            if len(distances) > 0:
                wsimeandis.append(np.mean(distances))
            else:
                wsimeandis.append(0.0)
        data['wsimeandis'] = np.asarray(wsimeandis)

        if not maxx is None:
            if maxx > np.max(data['trainset']):
                print("Stats file ({}) does not have that many samples ({}). Maximum is {}.".format(stats_path,maxx,np.max(data['trainset'])))
                #sys.exit(1)
                upl = data['trainset'].shape[0]
            else:
                upl = np.where(data['trainset'] >= maxx)[0][0]
                upl += 1

            data['trainset'] = data['trainset'][:upl]
            data['acqn'] = data['acqn'][:upl]
            data['acquired'] = data['acquired'][:upl]
            data['wsicount'] = data['wsicount'][:upl]
            data['pmean'] = data['pmean'][:upl]
            data['pdp'] = data['pdp'][:upl]
            data['wsimeandis'] = data['wsimeandis'][:upl]

        return data


    def return_fndata(self,data,key,merge=False,idx=None):
        """
        When feature network data is present, it should be split from target net data before plotting

        Returns: ( (X,Y) from target net, (X,Y) from feature net)

        If merge is True, return merged trainset, merged Y
        """
        if not 'fnidx' in data:
            if idx is None:
                return ((data['trainset'],data[key]),(data['trainset'],data[key]))
            else:
                return ((data['trainset'][idx],data[key][idx]),(data['trainset'][idx],data[key][idx]))
        elif merge:
            return ((np.hstack((data['trainset'],data['fntrainset'])),[]),([],np.hstack((data[key][data['tnidx']],data[key][data['fnidx']]))))
        else:
            m = data[key].shape[0]
            fnkey = 'fn'+ key if not key.startswith('fn') else key
            if not fnkey in data:
                return ((data['trainset'],data[key][data['tnidx'][:m]]),(data['fntrainset'],data[key][data['fnidx'][:m]]))
            else:
                return ((data['trainset'],data[key]),(data['fntrainset'],data[fnkey]))
        
    def format_func(self,x, pos):
        hours = int(x//3600)
        minutes = int((x%3600)//60)
        seconds = int(x%60)

        return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    
    def extractNPData(self,path):
        if not os.path.isfile(path):
            print("Given file path is incorrect. Check that.")
            sys.exit(1)

        data = np.load(path)
        return data

    def generateIndexes(self,size,start,step):
        """
        Should be used when X values are not given but are a sequence of fixed period integers
        """
        return np.asarray(range(start,(size*step)+start,step))
                              
    def parseSlurm(self,path=None,maxx=None,concat=False):

        if path is None and self.path is None:
            print("No directory found")
            sys.exit(1)
        elif path is None:
            path = self.path
        elif isinstance(path,list):
            print("Parse a single file at a time")
            return None

        dir_contents = list(filter(lambda f:f.startswith('slurm'),os.listdir(path)))
        sk = lambda f:int(f.split('.')[0].split('-')[1])
        dir_contents.sort(key=sk)        
        slurm_path = None

        if not dir_contents:
            print("No slurm file in path: {0}".format(path))
            return None

        lines = []
        if concat:
            for fi in dir_contents:
                slurm_path = os.path.join(path,fi)
                with open(slurm_path,'r') as fd:
                    lines.extend(fd.readlines())
        else:
            slurm_path = os.path.join(path,dir_contents[-1])
            with open(slurm_path,'r') as fd:
                lines = fd.readlines()

        print("Experiment {} ({}):".format(os.path.basename(path),os.path.basename(slurm_path)))
        data = {'time':[],
                'acqtime':[],
                'traintime':[],
                'taqtime':[],
                'auc':[],
                'fnauc':[],
                'trainset':[],
                'accuracy':[],
                'fnaccuracy':[],
                'labels':[],
                'wsave':[],
                'wload':[],
                'kmeans':[],
                'feature':[],
                'cluster':{}}
        start_line = 0
        timerex = r'AL step took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        trainrex = r'Training step took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        acqrex = r'Acquisition step took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        wsaverex = r'Weight saving took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        wloadrex = r'Weights loading took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        kmeansrex = r'KMeans took (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        featrex = r'Feature extraction took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        aucrex = r'AUC: (?P<auc>0.[0-9]+)'
        fnaucrex = r'FN AUC: (?P<auc>0.[0-9]+)'
        accrex = r'Accuracy: (?P<acc>0.[0-9]+)'
        fnaccrex = r'FN Accuracy: (?P<acc>0.[0-9]+)'
        trainsetrex = r'Train set: (?P<set>[0-9]+) items'
        clusterrex = r'Cluster (?P<cln>[0-9]+) labels: (?P<neg>[0-9]+) are 0; (?P<pos>[0-9]+) are 1;'
        labelrex = r'Train labels: (?P<neg>[0-9]+) are 0; (?P<pos>[0-9]+) are 1;'
        colorrex = r'ColorCode: (?P<code>[0-9]+)'
        
        timerc = re.compile(timerex)
        trainrc = re.compile(trainrex)
        acqrc = re.compile(acqrex)
        wsaverc = re.compile(wsaverex)
        wloadrc = re.compile(wloadrex)
        kmeansrc = re.compile(kmeansrex)
        featrc = re.compile(featrex)
        aucrc = re.compile(aucrex)
        fnaucrc = re.compile(fnaucrex)
        trainsetrc = re.compile(trainsetrex)
        accrc = re.compile(accrex)
        fnaccrc = re.compile(fnaccrex)        
        clusterrc = re.compile(clusterrex)
        labelrc = re.compile(labelrex)
        colorrc = re.compile(colorrex)

        #Set a time reference
        #zero = datetime.datetime(2020,1,1)
        #zero_num = mdates.date2num(zero)
        wstime = None
        wltime = None
        skip_repeat = False
        for line in lines:
            lstrip = line.strip()
            trainmatch = trainrc.fullmatch(lstrip)
            acqmatch = acqrc.fullmatch(lstrip)
            tmatch = timerc.fullmatch(lstrip)
            wsavematch = wsaverc.fullmatch(lstrip)
            wloadmatch = wloadrc.fullmatch(lstrip)
            kmeansmatch = kmeansrc.fullmatch(lstrip)
            featmatch = featrc.fullmatch(lstrip)
            aucmatch = aucrc.fullmatch(lstrip)
            fnaucmatch = fnaucrc.fullmatch(lstrip)
            trmatch = trainsetrc.fullmatch(lstrip)
            accmatch = accrc.fullmatch(lstrip)
            fnaccmatch = fnaccrc.fullmatch(lstrip)
            clustermatch = clusterrc.fullmatch(lstrip)
            labelmatch = labelrc.fullmatch(lstrip)
            colormatch = colorrc.fullmatch(lstrip)
            if trmatch:
                ssize = int(trmatch.group('set'))
                if ssize in data['trainset']:
                    skip_repeat = True
                    continue
                else:
                    data['trainset'].append(ssize)
                    skip_repeat = False
            #Skip aquisition if concatenation is on        
            if skip_repeat and concat:
                continue
            
            if tmatch:
                td = datetime.timedelta(hours=int(tmatch.group('hours')),minutes=int(tmatch.group('min')),seconds=round(float(tmatch.group('sec'))))
                data['time'].append(td.total_seconds())
                if not wstime is None:
                    data['wsave'].append(wstime.total_seconds())
                    wstime = None
                if not wltime is None:
                    data['wload'].append(wltime.total_seconds())
                    wltime = None
            if trainmatch:
                td = datetime.timedelta(hours=int(trainmatch.group('hours')),minutes=int(trainmatch.group('min')),seconds=round(float(trainmatch.group('sec'))))
                data['traintime'].append(td.total_seconds())
            if acqmatch:
                td = datetime.timedelta(hours=int(acqmatch.group('hours')),minutes=int(acqmatch.group('min')),seconds=round(float(acqmatch.group('sec'))))
                data['acqtime'].append(td.total_seconds())
            if wsavematch:
                td = datetime.timedelta(hours=int(wsavematch.group('hours')),minutes=int(wsavematch.group('min')),seconds=round(float(wsavematch.group('sec'))))
                if wstime is None:
                    wstime = td
                else:
                    wstime += td
            if wloadmatch:
                td = datetime.timedelta(hours=int(wloadmatch.group('hours')),minutes=int(wloadmatch.group('min')),seconds=round(float(wloadmatch.group('sec'))))
                if wltime is None:
                    wltime = td
                else:
                    wltime += td
            if kmeansmatch:
                td = datetime.timedelta(hours=int(kmeansmatch.group('hours')),minutes=int(kmeansmatch.group('min')),seconds=round(float(kmeansmatch.group('sec'))))
                data['kmeans'].append(td.total_seconds())
            if featmatch:
                td = datetime.timedelta(hours=int(featmatch.group('hours')),minutes=int(featmatch.group('min')),seconds=round(float(featmatch.group('sec'))))
                data['feature'].append(td.total_seconds())
            if aucmatch:
                data['auc'].append(float(aucmatch.group('auc')))
            if fnaucmatch:
                data['fnauc'].append(float(fnaucmatch.group('auc')))
                data.setdefault('fntrainset',[])
                data['fntrainset'].append(len(data['trainset']) - 1)
            if accmatch:
                data['accuracy'].append(float(accmatch.group('acc')))
            if fnaccmatch:
                data['fnaccuracy'].append(float(fnaccmatch.group('acc')))                
            if labelmatch:
                data['labels'].append((int(labelmatch.group('pos')),int(labelmatch.group('neg'))))
            if colormatch:
                data['color'] = int(colormatch.group('code'))
                
            if clustermatch:
                cln = int(clustermatch.group('cln'))
                tot = int(clustermatch.group('pos')) + int(clustermatch.group('neg'))
                if tot > 5:
                    if cln in data['cluster']:
                        data['cluster'][cln].append((int(clustermatch.group('pos')),int(clustermatch.group('neg'))))
                    else:
                        data['cluster'][cln] = [(int(clustermatch.group('pos')),int(clustermatch.group('neg')))]
                else:
                    print("Cluster {} has only {} items and will be discarded for plotting".format(cln,tot))
                

        #Use NP arrays
        data['traintime'] = np.asarray(data['traintime'])
        #Last iteration doesn't have an aquisition phase (zero time)
        if len(data['traintime']) > len(data['acqtime']):
            data['acqtime'].append(0.0)
        data['acqtime'] = np.asarray(data['acqtime'])
        data['time'] = np.asarray(data['time'])
        data['wsave'] = np.asarray(data['wsave'])
        data['wload'] = np.asarray(data['wload'])
        data['kmeans'] = np.asarray(data['kmeans'])
        data['feature'] = np.asarray(data['feature'])
        data['auc'] = np.asarray(data['auc'])
        data['fnauc'] = np.asarray(data['fnauc'])
        data['trainset'] = np.asarray(data['trainset'])
        data['accuracy'] = np.asarray(data['accuracy'])
        data['fnaccuracy'] = np.asarray(data['fnaccuracy'])        
        data['labels'] = np.asarray(data['labels'])
        data['taqtime'] = data['acqtime'] + data['traintime']
            
        if not maxx is None:
            if maxx > np.max(data['trainset']):
                print("Slurm file ({}) does not have that many samples ({}). Maximum is {}.".format(slurm_path,maxx,np.max(data['trainset'])))
                #sys.exit(1)
                upl = data['trainset'].shape[0]
            else:
                upl = np.where(data['trainset'] >= maxx)[0][0]
                upl += 1
            
            data['time'] = data['time'][:upl]
            data['traintime'] = data['traintime'][:upl]
            data['acqtime'] = data['acqtime'][:upl]
            data['wsave'] = data['wsave'][:upl]
            data['wload'] = data['wload'][:upl]
            data['kmeans'] = data['kmeans'][:upl]
            data['feature'] = data['feature'][:upl]            
            data['auc'] = data['auc'][:upl]
            data['trainset'] = data['trainset'][:upl]
            data['accuracy'] = data['accuracy'][:upl]
            data['labels'] = data['labels'][:upl]
            data['taqtime'] = data['taqtime'][:upl]

        #Generate indexes that correspond to data from Feature Net and Target Net
        if data['fnauc'].shape[0] > 0:
            data['fntrainset'] = np.asarray(data['fntrainset'])
            fnids = np.zeros(data['trainset'].shape[0],dtype=np.int32)
            if np.max(data['fntrainset']) >= fnids.shape[0]:
                upl = np.where(data['fntrainset'] >= fnids.shape[0])[0][0]
                data['fntrainset'] = data['fntrainset'][:upl] #Don't use indexes already removed by maxx
                #data['auc'] = data['auc'][:upl]
                data['fnauc'] = data['fnauc'][:upl]
                #data['accuracy'] = data['accuracy'][:upl]
                data['fnaccuracy'] = data['fnaccuracy'][:upl]
            fnids[data['fntrainset']] = 1
            data['fnidx'] = data['fntrainset']
            data['fntrainset'] = data['trainset'][data['fnidx']]
            data['tnidx'] = np.logical_xor(fnids,1)
            data['trainset'] = data['trainset'][data['tnidx']]
            
        #Round AUC to 2 decimal places
        np.around(data['auc'],2,data['auc'])

        if data['auc'].shape[0] > 0:
            print("X: {}; AUC: {}".format(data['trainset'],data['auc']))
            print("Min AUC: {0}; Max AUC: {1}; Mean AUC: {2:.3f}\n".format(data['auc'].min(),data['auc'].max(),data['auc'].mean()))
        if data['accuracy'].shape[0] > 0:
            print("Min accuracy: {0}; Max accuracy: {1}; Mean accuracy: {2:.3f}\n".format(data['accuracy'].min(),data['accuracy'].max(),data['accuracy'].mean()))

        return data

    def generateDataFromPik(self,pik_dir=None):
        """
        Reads pickle files generated by AL debug function. 
        Returns data to be plotted as if parsed from SLURM files.
        """
        if pik_dir is None:
            pik_dir = self.path
            
        if pik_dir is None or not os.path.isdir(pik_dir):
            return None

        import pickle
        from sklearn import metrics

        def extract_acq(f):
            if f.startswith('al-probs-'):
                return f.split('.')[0].split('-')[3][1:]
            else:
                return -1

        def sort_key(f):
            return int(extract_acq(f))
            
        files = os.listdir(pik_dir)
        files.sort(key=sort_key)
        data = {'accuracy':[],
                'trainset':[]}
        for f in files:
            if f.startswith('al-probs-'):
                fd = open(os.path.join(pik_dir,f),'rb')
                y_true,sprobs = pickle.load(fd)
                fd.close()
                s_pred_all = sprobs[:,:].argmax(axis=0)
                print("Votes array ({})".format(s_pred_all.shape))
                for k in range(0,10):
                    print(s_pred_all[k])
                y_pred = np.asarray([np.bincount(s_pred_all[i]).argmax(axis=0) for i in range(0,s_pred_all.shape[0])])
                print(y_pred)
                acc = metrics.accuracy_score(y_true,y_pred)
                acq = extract_acq(f)
                data['accuracy'].append(acc)
                data['trainset'].append(int(acq))
                print("Acquisition {} accuracy: {}".format(acq,acc))

        data['trainset'] = np.asarray(data['trainset'])
        data['accuracy'] = np.asarray(data['accuracy'])
        return data

    def retrieveUncertainty(self,config):
        unc_files = []
        cluster_files = []
        import pickle

        acq = 0
        def extract_acq(f):
            if f.startswith('al-'):
                return f.split('.')[0].split('-')[3][1:]
            else:
                return -1

        def sort_key(f):
            return int(extract_acq(f))        
        
        def kmuncert_acquire(fc,fu,config,acq):
            #Check uncertainty by the indexes, lower indexes correspond to greater uncertainty
            ind = None
            posa = {}
            un_clusters = {}
            cl_labels = {}
            verbose = 1
            print('ACQ {} ****************************'.format(acq))
            with open(os.path.join(config.sdir,fc),'rb') as fd:
                (X,Y),clusters,un_indexes = pickle.load(fd)
            with open(os.path.join(config.sdir,fu),'rb') as fd:
                _,uncertainties = pickle.load(fd)

            for k in range(config.kmclusters):
                ind = np.asarray(clusters[k],dtype=np.int32)
                posa[k] = []
                for ii in range(min(ind.shape[0],config.query)):
                    posa[k].append(np.where(un_indexes == ind[ii])[0][0])
                posa[k] = np.asarray(posa[k],dtype=np.int32)
                un_clusters[k] = uncertainties[ind]

                #If debug
                if config.debug:
                    expected = Y
                    print("Cluster {}, # of items: {}".format(k,ind.shape[0]))
                    print("Cluster {} first items positions in index array (first 30): {}".format(k,posa[k][:30]))
                    #Check % of items of each class in cluster k
                    c_labels = expected[ind]
                    unique,count = np.unique(c_labels,return_counts=True)
                    l_count = dict(zip(unique,count))
                    if len(unique) > 2:
                        print("Cluster {} items:".format(k))
                        print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
                    else:
                        if unique.shape[0] == 1:
                            l_count[unique[0] ^ 1] = 0
                        #print("Cluster {3} labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1])),k))
                        cl_labels[k] = (l_count[0],l_count[1])
                    del(expected)                
            
            #Selection
            ac_count = 0
            acquired = []
            j,n = 0,0
            cmean = np.asarray([np.mean(posa[k]) for k in range(config.kmclusters)])
            glb = np.sum(cmean)
            frac = (glb/cmean)/np.sum(glb/cmean)
            frac[np.isnan(frac)] = 1/config.kmclusters #Prevent NaN values if cmean is zero
            sel = np.zeros(config.kmclusters,dtype=np.int32)
            while ac_count < config.query:
                cln = n % config.kmclusters
                q = clusters[cln]
                cl_aq = int(np.ceil(frac[cln]*config.query))
                cl_aq = 1 if cl_aq == 0 else cl_aq #Select at least 1 patch from each cluster
                first = sel[cln]
                if len(q) >= first + cl_aq:
                    acquired.extend(q[first:first+cl_aq])
                    ac_count += cl_aq
                    sel[cln] = first+cl_aq
                    j += 1
                    n = j
                    if config.debug or verbose > 0:
                        print("[km_uncert] Selected {} patches for acquisition from cluster {}".format(cl_aq,cln))
                else:
                    l = len(q)
                    acquired.extend(q[first:l])
                    ac_count += l-first
                    sel[cln] = l
                    np.random.seed(n*(n+1))
                    r = np.random.randint(0,config.kmclusters)
                    n = r if r != n else n+1
                    if verbose > 0:
                        print("[km_uncert] Cluster {} exausted, all {} patches acquired. Will try to acquire remaining patches from cluster {}".format(cln,l-first,(n)%config.kmclusters))
            acquired = np.asarray(acquired[:config.query],dtype=np.int32)
            return acquired,uncertainties,un_clusters,cl_labels
        ####
        
        if not config.all:
            for i in config.ac_n:
                unc_file = 'al-clustermetadata-Inception-r{}.pik'.format(i)
                if os.path.isfile(os.path.join(config.sdir,unc_file)):
                    cluster_files.append(unc_file)
                
                unc_file = 'al-uncertainty-{}-r{}.pik'.format(config.ac_func,i)
                if os.path.isfile(os.path.join(config.sdir,unc_file)):
                    unc_files.append(unc_file)
        else:
            items = os.listdir(config.sdir)
            for f in items:
                if f.startswith('al-clustermetadata'):
                    cluster_files.append(f)
                elif f.startswith('al-uncertainty'):
                    unc_files.append(f)
            if len(unc_files) == 0 and len(cluster_files) == 0:
                print("No data to plot. Perhaps the wrong experiment dir was used.")
                sys.exit(1)

        data = []
        cluster_files.sort(key=sort_key)
        unc_files.sort(key=sort_key)

        if len(cluster_files) == 0:
            for f in unc_files:
                with open(os.path.join(config.sdir,f),'rb') as fd:
                    indexes,uncertainties = pickle.load(fd)
                data.append((indexes,uncertainties))
        else:
            for k in range(len(cluster_files)):
                indexes,uncertainties,un_clusters,cl_labels = kmuncert_acquire(cluster_files[k],unc_files[k],config,acq)
                data.append((indexes,uncertainties,un_clusters,cl_labels))
                acq += 1
            
        return data

    def calculate_stats(self,data,auc_only,ci,metrics = None):
        """
        @param data <dict>: a dictionary as returned by parseResults

        Calculates mean and standard deviation for AUC and/or Accuracy.

        Returns a list of tuples (trainset,mean_values,std dev,label,color) for each AUC and Accuracy
        """
        color = -1
        trainset = None
        stats = None
        max_samples = None
        mvalues = None
        
        def calc_ci(val,ci):
            a = np.zeros(shape=val.shape[0],dtype=np.float32)
            for k in range(a.shape[0]):
                n = val[k].shape[0]
                se = scipy.stats.sem(val[k])
                a[k] = se * scipy.stats.t.ppf((1 + ci) / 2., n-1)
            return a

        def calc_metric(data,metric,auc_only):
            max_samples = np.inf
            trainset = None
            mvalues = None
            idx = None
            i = 0
            exp_n = len(data)
            #Check if all experiments had the same number of samples
            for k in data:
                if not metric in data[k] or data[k][metric].shape[0] == 0:
                    print("Requested metric not available ({}) in experiment {}".format(metric,k))
                    exp_n -= 1
                    continue
                if (auc_only and data[k]['auc'].shape[0] > 0) or not 'fntrainset' in data[k]:
                    max_samples = min(max_samples,len(data[k]['trainset']))
                elif not auc_only and not metric.startswith('fn'):
                    max_samples = min(max_samples,len(data[k]['trainset']))
                elif not auc_only: #data[k][metric].shape[0] > 0:
                    max_samples = min(max_samples,len(data[k]['fntrainset']))

            if max_samples == np.inf:
                return (None,None,None)
            
            mvalues = np.zeros(shape=(exp_n,max_samples),dtype=np.float32)
            
            for k in data:
                if not metric in data[k] or data[k][metric].shape[0] == 0:
                    continue
                
                if auc_only and data[k]['auc'].shape[0] > 0:
                    dd = mvalues.shape[1] - data[k]['auc'].shape[0]
                    trainset = data[k]['trainset']
                    if dd > 0:
                        print("Wrong dimensions. Expected {} points in experiment {} but got {}".format(mvalues.shape[1],k,data[k]['auc'].shape[0]))
                        mvalues = np.delete(mvalues,mvalues.shape[1] - 1,axis=1)
                        trainset = trainset[:-1]
                        max_samples -= 1
                    mvalues[i] = data[k]['auc'][:max_samples]
                if not auc_only and data[k][metric].shape[0] > 0:                    
                    trainset = data[k]['fntrainset'] if metric.startswith('fn') else data[k]['trainset']
                    tdata = data[k][metric][:max_samples]
                    #(_,_),(_,tdata) = self.return_fndata(data[k],metric,False)
                    if not tdata is None:
                        dd = mvalues.shape[1] - tdata.shape[0]
                        if dd > 0:
                            print("Wrong dimensions. Expected {} points in experiment {} but got {}".format(mvalues.shape[1],k,tdata.shape[0]))
                            mvalues = np.delete(mvalues,mvalues.shape[1] - dd,axis=1)
                            trainset = trainset[:-dd]
                        mvalues[i,:tdata.shape[0]] = tdata[:mvalues[i].shape[0]]
                    else:
                        mvalues[i] = data[k][metric][:max_samples]
                
                if 'color' in data[k]:
                    color = data[k]['color']
                    
                i += 1

            print(trainset,mvalues,max_samples)
            return (trainset,mvalues,max_samples)

        metric = None
        if metrics is None or len(metrics) == 0:
            metric = 'auc'
            trainset,mvalues,max_samples = calc_metric(data,metric,auc_only)
        elif len(metrics) >= 1:
            stats = {}
            for m in metrics:
                stats[m] = calc_metric(data,m,auc_only)

        #Return mean and STD dev
        if auc_only:
            d = (trainset[:max_samples],np.mean(mvalues.transpose(),axis=1),calc_ci(mvalues.transpose(),ci),"AUC",color)
            print("Max AUC: {:1.3f}; Mean AUC ({} acquisitions): {:1.3f}".format(np.max(d[1]), d[1].shape[0],np.mean(d[1])))
            return [d]
        else:
            for m in stats:
                trainset,mvalues,max_samples = stats[m]
                if trainset is None or mvalues is None:
                    continue
                d = (trainset[:max_samples],np.mean(mvalues.transpose(),axis=1),calc_ci(mvalues.transpose(),ci),metric,color)
                stats[m] = d
                print("Max {0}: {1:1.3f}; Mean {0} ({2} acquisitions): {3:1.3f}".format(m,np.max(d[1]),d[1].shape[0],np.mean(d[1])))
            return stats
                                                                                                              
    
if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    ##Multiline SLURM parse
    parser.add_argument('--multi', action='store_true', dest='multi', default=False, 
        help='Plot multiple lines from slurm files.')
    parser.add_argument('-pos', action='store_true', dest='pos', default=False, 
        help='Plot percentage of positive patches during aquisition.')
    parser.add_argument('-merge', action='store_true', dest='merge', default=False, 
        help='In AL Transfer, merge plots from FN and TN in the same figure.')    
    parser.add_argument('-ids', dest='ids', nargs='+', type=int, 
        help='Experiment IDs to plot.', default=None,required=False)
    parser.add_argument('-colors', dest='colors', nargs='+', type=int, 
        help='Line colors. Follow the order of the IDs.', default=None,required=False)
    parser.add_argument('-xtick', dest='xtick', type=int, 
        help='xtick interval.', default=200,required=False)
    parser.add_argument('-ytick', dest='ytick', type=int, 
        help='ytick interval.', default=200,required=False)
    parser.add_argument('-ncols', dest='ncols', type=int, 
        help='# of columns in label box.', default=2,required=False)
    parser.add_argument('-maxx', dest='maxx', type=int, 
        help='Plot maximum X.', default=None,required=False)
    parser.add_argument('-maxy', dest='maxy', type=float, 
        help='Plot maximum Y.', default=0.0,required=False)
    parser.add_argument('-miny', dest='miny', type=float, 
        help='Plot maximum Y.', default=0.0,required=False)
    parser.add_argument('-t', dest='title', type=str,default='', 
        help='Figure title.')
    parser.add_argument('-labels', dest='labels', nargs='+', type=str, 
        help='Curve labels.',default=None,required=False)
    parser.add_argument('-concat', action='store_true', dest='concat', default=False, 
        help='Concatenate multiple slurm files if more then one is present.')    
    parser.add_argument('-metrics', dest='metrics', type=str, nargs='+',
        help='Metrics to plot: \n \
        time - AL iteration time; \n \
        traintime - Train step time; \n \
        acqtime - Acquisition step time; \n \
        taqtime - Train + Acquisition step time; \n \
        wsave - Weights saving time; \n \
        wload - Weights loading time; \n \
        kmeans - KMeans execution time; \n \
        feature - Feature extraction time; \n \
        auc - AUC; \n \
        acc - Accuracy; \n \
        fnauc - Feature network AUC; \n \
        fnacc - Feature network Accuracy; \n \
        labels - Positive labeled patches acquired; \n \
        pmean - Mean # of patches acquired from each WSI; \n \
        wsicount - # of WSIs used in each acquisition step; \n \
        wsimeandis - Mean distance, in pixels, from patches to its cluster centers.',
       choices=['time','auc','acc','labels','traintime','acqtime','taqtime','pmean','wsicount','wsimeandis','wsave','wload','kmeans','feature','fnauc','fnacc'],
                            default=None)    
    parser.add_argument('-type', dest='tmode', type=str, nargs='+',
        help='Experiment type: \n \
        AL - General active learning experiment; \n \
        MN - MNIST dataset experiment.',
       choices=['AL','MN','DB','OR','KM','EN','TMP'],default='AL')
    parser.add_argument('-yscale', action='store_true', dest='yscale', default=False, 
        help='Scale y axis ticks to data.')    

    ##Multitime
    parser.add_argument('--mtime', action='store_true', dest='mtime', default=False, 
        help='Plot timing slices from a single experiment.')
    
    ##Single experiment plot
    parser.add_argument('--single', action='store_true', dest='single', default=False, 
        help='Plot data from a single experiment.')
    parser.add_argument('-sd', dest='sdir', type=str,default=None, required=True,
        help='Experiment result path (should contain an slurm file).')

    ##Make stats
    parser.add_argument('--stats', action='store_true', dest='stats', default=False, 
        help='Make mean and STD dev from multiple runs.')
    parser.add_argument('-auc', action='store_true', dest='auc_only', default=False, 
        help='Calculate statistics for AUC only.')
    parser.add_argument('-ci', dest='confidence', nargs=1, type=float, 
        help='CI (T-Student).', default=0.95,required=False)
    parser.add_argument('-n', dest='n_exp', nargs='+', type=int, 
        help='N experiments for each curve (allows plotting 2 curves together).',
        default=(3,3),required=False)
    
    ##Draw uncertainties
    parser.add_argument('--uncertainty', action='store_true', dest='unc', default=False, 
        help='Plot experiment uncertainties selected acquisitions.')
    parser.add_argument('-ac', dest='ac_n', nargs='+', type=int, 
        help='Acquisitions to plot.', default=None,required=False)
    parser.add_argument('-all', action='store_true', dest='all', default=False, 
        help='Plot all acquisitions.')
    parser.add_argument('-ac_func', dest='ac_func', type=str,default='bayesian_bald', 
        help='Function to look for uncertainties.')
    parser.add_argument('-sp', dest='spread', type=int, 
        help='Spread points in interval.', default=10,required=False)
    parser.add_argument('-kmu', action='store_true', dest='kmu', default=False, 
        help='Plot cluster patch uncertainties.')
    parser.add_argument('-query', dest='query', nargs=1, type=int, 
        help='If KMUncert, number of acquired patches.', default=200,required=False)
    parser.add_argument('-kmclusters', dest='kmclusters', nargs=1, type=int, 
        help='If KMUncert, number of clusters used.', default=20,required=False)
    parser.add_argument('-plt_cluster', dest='plt_cluster', nargs=1, type=int, 
        help='Plot cluster patch uncertainty for this acquisition.', default=10,required=False)

    ##Plot debugging data
    parser.add_argument('--debug', action='store_true', dest='debug', default=False, 
        help='Plot experiment uncertainties selected acquisitions. Single or multiline.')
    parser.add_argument('-clusters', action='store_true', dest='clusters', default=False, 
        help='Plot cluster composition.')
    parser.add_argument('-meta', action='store_true', dest='meta', default=False, 
        help='Extract cluster data from pik metafiles.')
    parser.add_argument('-unc', action='store_true', dest='draw_unc', default=False, 
        help='Use cluster uncertainties.')

    ##Plot WSI metadata stats data
    parser.add_argument('--wsi', action='store_true', dest='wsi', default=False, 
        help='Plot data obtained by MetadataExtract.')
    parser.add_argument('-err_bar', action='store_true', dest='err_bar', default=False, 
        help='Plot error bars.')
    parser.add_argument('-ncf', dest='ncf', type=int, 
        help='Use the acquisition file correspondent to this number of clusters.', default=20,required=False)    
    
    config, unparsed = parser.parse_known_args()

    if config.ids is None and config.sdir is None:
        print("You should either specify an ID or a full path to grab data")
        sys.exit(1)
        
    if config.sdir is None or not os.path.isdir(config.sdir):
        print("Directory not found: {}".format(config.sdir))
        sys.exit(1)
        
    if config.multi and not (config.stats or config.debug):
        if config.sdir is None:
            print("You should specify an experiment directory (use -sd option).")
            sys.exit(1)

        if len(config.tmode) == 1:
            exp_type = os.path.join(config.sdir,config.tmode[0],config.tmode[0])
        else:
            exp_type = [os.path.join(config.sdir,tmode,tmode) for tmode in config.tmode]
            
        p = Plotter(ncols=config.ncols)

        if not config.metrics is None and len(config.ids) == 1:
            ex_dir = "{}-{}".format(exp_type,str(config.ids[0]))
            data = p.parseMetrics(p.parseSlurm(ex_dir,config.maxx),config.ids[0],config.metrics)
            p.draw_multilabel(data,config.title,config.xtick,config.metrics,config.labels,config.yscale)
        else:
            data = p.parseResults(exp_type,config.ids,maxx=config.maxx,concat=config.concat)
            if len(data) == 0:
                print("Something is wrong with your command options. No data to plot")
                sys.exit(1)
            kwargs = {'labels':config.labels,'pos':config.pos,'auc':config.auc_only,'other':config.metrics,'colors':config.colors,
                          'maxy':config.maxy,'miny':config.miny,'scale':config.yscale,'merge':config.merge}
            p.draw_multiline(data,config.title,config.xtick,**kwargs)
                
    elif config.single:
        p = Plotter(path=config.sdir,ncols=config.ncols)
        if not config.metrics is None:
            p.draw_data(p.parseSlurm(),config.title,metric=config.metrics[0],color=config.colors[0])
        else:
            p.draw_data(p.parseSlurm(),config.title,color=config.colors[0])

    elif config.mtime:
        p = Plotter(path=config.sdir,ncols=config.ncols)
        if not config.metrics is None:
            p.draw_multitime(p.parseSlurm(),config.title,config.xtick,metrics=config.metrics,colors=config.colors,labels=config.labels)
        else:
            p.draw_data(p.parseSlurm(),config.title,color=config.colors[0])
            
    elif config.unc:
        p = Plotter(ncols=config.ncols)

        if config.sdir is None:
            print("You should specify an experiment directory (use -sd option).")
            sys.exit(1)
            
        data = p.retrieveUncertainty(config)

        if len(data) == 0:
            print("Something is wrong with your command options. No data to plot")
            sys.exit(1)
        p.draw_uncertainty(data,config.ac_n,config.spread,config.title,config.kmu,config.plt_cluster)

    elif config.stats:
        p = Plotter(ncols=config.ncols)
        if config.sdir is None:
            print("You should specify an experiment directory (use -sd option).")
            sys.exit(1)

        if len(config.tmode) == 1:
            exp_type = os.path.join(config.sdir,config.tmode[0],config.tmode[0])
        elif len(config.tmode) != len(config.n_exp):
            print("Experiment types and sizes mismatch")
            sys.exit(1)
        else:
            exp_type = []
            for i in range(len(config.n_exp)):
                exp_type.append(os.path.join(config.sdir,config.tmode[i],config.tmode[i]))
        
        if config.ids is None:
            print("You should define a set of experiment IDs (-id).")
            sys.exit(1)

        u,c = np.unique(config.ids, return_counts=True)
        dup = u[c>1]
        if np.any(dup):
            print("Your are using duplicated IDs, change these: {}".format(dup))
            #sys.exit()
                
        data = p.parseResults(exp_type,config.ids,config.n_exp,config.maxx,config.concat)
        
        if isinstance(config.confidence,list):
            config.confidence = config.confidence[0]

        if config.confidence < 0.0 or config.confidence > 1.0:
            print("CI interval should be between 0.0 and 1.0")
            sys.exit(1)

        if config.multi:
            idx = 0
            c = []
            d = {}
            for z in range(len(config.n_exp)):
                i = config.n_exp[z]
                if i > 0:
                    print("\n***Calculating statistics for experiments {}".format(config.ids[idx:idx+i]))
                    tm = config.tmode[z]
                    if config.auc_only:
                        c.extend(p.calculate_stats({f"{tm}-{k}":data[f"{tm}-{k}"] for k in config.ids[idx:idx+i]},config.auc_only,config.confidence,config.metrics))
                    else:
                        d[z] = p.calculate_stats({f"{tm}-{k}":data[f"{tm}-{k}"] for k in config.ids[idx:idx+i]},config.auc_only,config.confidence,config.metrics)
                    idx += i
            if config.auc_only:
                data = c
            else:
                data = d
        else:
            data = p.calculate_stats(data,config.auc_only,config.confidence,config.metrics)
            
        if len(data) == 0:
            print("Something is wrong with your command options. No data to plot")
            sys.exit(1)

        if config.auc_only:
            p.draw_stats(data,config.xtick,config.auc_only,config.labels,config.spread,config.title,config.colors,yscale=config.yscale,maxy=config.maxy,miny=config.miny)
        else:
            p.draw_time_stats(data,config.xtick,config.auc_only,config.metrics,config.labels,config.spread,
                                  config.title,config.colors,yscale=config.yscale,maxy=config.maxy,miny=config.miny,merge=config.merge)

    elif config.debug:

        if config.sdir is None:
            print("Results dir path is needed (-sd option)")
            sys.exit(1)

        p = Plotter(path=config.sdir,ncols=config.ncols)
        data = {}
        if config.meta and config.clusters:
            full_data = p.retrieveUncertainty(config)
            data['labels'] = {k:full_data[k][3] for k in range(len(full_data))}
            data['unc'] = {k:full_data[k][2] for k in range(len(full_data))}
            data['ind'] = {k:full_data[k][0] for k in range(len(full_data))}
        elif config.clusters:
            full_data = p.parseSlurm()
            if 'cluster' in full_data:
                cldata = full_data['cluster']
                for cl in cldata:
                    for i in range(len(cldata[cl])):
                        data['labels'].setdefault(i,{})
                        data['labels'][i][cl] = cldata[cl][i]
                        
        else:
            data = p.generateDataFromPik()
        
        if config.multi:
            #In multi_plot, change the xvalues so that curves reflect the same acquisition
            d2 = p.parseSlurm()
            data['trainset'] = d2['trainset']
            #The debug function only  considers accuracy
            if 'auc' in d2:
                del d2['auc']
            mdata = {'AL selected':data,
                    'AL trained':d2}
            p.draw_multiline(mdata,config.title,config.xtick)
        elif config.clusters and not data is None:
            p.draw_cluster_distribution(data,config.spread,config.title)
        elif not data is None:
            p.draw_data(data,config.title,'Acquisition #')
        else:
            print("No data to plot.")

    elif config.wsi:
        if config.sdir is None:
            print("You should specify an experiment directory (use -sd option).")
            sys.exit(1)

        if config.metrics is None:
            print("Define WSI metrics to plot (at most 2)")
            sys.exit(1)
        elif len(config.metrics) > 2:
            config.metrics = config.metrics[:2]
            
        if len(config.tmode) == 1:
            exp_type = os.path.join(config.sdir,config.tmode[0],config.tmode[0])
        else:
            exp_type = [os.path.join(config.sdir,tmode,tmode) for tmode in config.tmode]
            
        p = Plotter(ncols=config.ncols)

        data = p.parseResults(exp_type,config.ids,config.n_exp,config.maxx,config.concat,wsi=True,ncf=config.ncf)

        p.draw_wsi_stat(data,config.title,config.labels,config.metrics,config.colors,config.ytick,config.xtick,config.err_bar)

