#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 08:39:15 2022

@author: kesson
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
HF=[ 2.80677337,  10.90159644,  29.97596055,  63.65834929,
       114.94254916, 171.9666131 , 248.43084898]
LiH=[ 13.97506367,  25.67692155,  64.20487913, 119.34797676,
       183.93633332, 404.94401783]
H6=[ 3.67880073,  19.02174547,  33.32975321,  61.22434267,
        84.2137248 , 116.77558529, 157.66303207]
H8=[ 6.63633219,  11.31309819,  31.35122097,  56.25171852,
       105.38991502, 146.13231036]
H4=[ 3.46372538,  14.16594896,  33.52886502,  62.87347664,
       102.93931972, 149.17881436]
H2=[  3.49365424,  12.26077549,  41.48724647,  80.31871301,
       112.81909865, 153.93319548]
E={}
E['HF']=HF
E['LiH']=LiH
E['H6']=H6
E['H8']=H8
E['H4']=H4
E['H2']=H2
name=['H2','H4','H6','H8']
color=['r','g','b','k']
k=0
for i in name:
    sd1=[]
    sd2=[]
    sd3=[]
    for j in range(len(E[i])):
        vt=np.loadtxt('H10/HDtest/HN'+i+str(j)+'_vt')
        ext=np.loadtxt('H10/HDtest/HN'+i+str(j)+'_ext')
        ex=np.loadtxt('H10/HDtest/Hall'+i+str(j)+'_ext')
        sd1.append(len(vt))
        sd2.append(len(ext))
        sd3.append(len(ex))
    plt.plot(E[i],sd1,c=color[k], label=i+'(No control)')
    plt.plot(E[i],sd2,c=color[k],linestyle=':',label=i+'(Control negative)')
    plt.plot(E[i],sd3,c=color[k],linestyle='--',label=i+'(Control all)')
    k+=1
    
plt.xscale('log')
plt.legend(fontsize=15)
plt.title('Control Hamiltonian Test',fontsize=30)
plt.xlabel('1/${\Delta E}$', fontsize=20)
plt.ylabel('Total time steps', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

name=['LiH','HF']
color=['r','b']
k=0
fig,ax = plt.subplots(1,2)
for i in name:
    sd1=[]
    sd2=[]
    sd3=[]
    for j in range(len(E[i])):
        vt=np.loadtxt('H10/HDtest/HN'+i+str(j)+'_vt')
        ext=np.loadtxt('H10/HDtest/HN'+i+str(j)+'_ext')
        ex=np.loadtxt('H10/HDtest/Hall'+i+str(j)+'_ext')
        sd1.append(len(vt))
        sd2.append(len(ext))
        sd3.append(len(ex))
    ax[k].plot(E[i],sd1,c=color[k], label=i+'(No control)')
    ax[k].plot(E[i],sd2,c=color[k],linestyle=':',label=i+'(Control negative)')
    ax[k].plot(E[i],sd3,c=color[k],linestyle='--',label=i+'(Control all)')
    ax[k].set_title('Control Hamiltonian Test ('+i+')',fontsize=30)
    ax[k].set_xscale('log')
    ax[k].legend(fontsize=15)
    ax[k].set_ylabel('Total time steps',fontsize=20)
    ax[k].set_xlabel('1/${\Delta E}$',fontsize=20)
    ax[k].tick_params(labelsize=15)
    k+=1