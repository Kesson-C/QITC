#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:18:10 2021

@author: kesson
"""

from scipy.sparse import csr_matrix
import numpy as np
from satmethod import system,method
import matplotlib.pyplot as plt
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import state_fidelity
from scipy.linalg import expm
from itertools import permutations,combinations
def label2Pauli(s): # can be imported from groupedFermionicOperator.py
    xs = []
    zs = []
    label2XZ = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    for c in s[::-1]:
        x, z = label2XZ[c]
        xs.append(x)
        zs.append(z)
    return Pauli(z = zs, x = xs)

def str2WeightedPaulis(s):
	s = s.strip()
	IXYZ = ['I', 'X', 'Y', 'Z']
	prev_idx = 0
	coefs = []
	paulis = []
	is_coef = True
	for idx, c in enumerate(s + '+'):
		if idx == 0: continue
		if is_coef and c in IXYZ:
			coef = complex(s[prev_idx : idx].replace('i', 'j'))
			coefs.append(coef)
			is_coef = False
			prev_idx = idx
		if not is_coef and c in ['+', '-']:
			label = s[prev_idx : idx]
			paulis.append(label2Pauli(label))
			is_coef = True
			prev_idx = idx
	return WeightedPauliOperator([[c, p] for (c,p) in zip(coefs, paulis)])

def anticommutator(A,B):
    return A.dot(B)+B.dot(A)

def comb(a,b,c,d):
    perm=[]
    for i in combinations(range(a),b):
        tmp=[d]*a
        for j in i:
            tmp[j]=c
        perm.append(''.join(tmp))
    return perm
Hd=[]
perms=[]        
perm=[]
# perms += comb(11,2,'Z','I')
perms += comb(11,1,'Z','I')
perms=set(perms)
for i in perms:
    Hd.append(Pauli(i).to_matrix(sparse=True))
    perm.append(i)
sample=16
n_qubit=11
case=1
dt=1e-1
S=1
roop=1000


print("sample",sample,"case",case)
resultcc=np.loadtxt('sat'+str(n_qubit)+'-result2.txt') 
resultc1=resultcc.tolist()
print("result",len(resultc1))
result=[int(x) for x in resultc1[sample]] 
HB,HP=system.satSystem(n_qubit,result)
T=300
nqubit=11
form=0
nq=2**nqubit
ax=[]
ay=[]
ay2=[]
ac=[]
ae=[]
aa1=[]
aa2=[]
aax=[]
dae=[]
Cut=5
 
N=2**nqubit

dtt=T/100
sd1=[]
sd2=[]
for ff in np.linspace(67,80):
    t=dtt*ff
    ax.append(t)
    
    if form==0:
        st=t/T
          
    Hp=(1-st)*HB+st*HP
    a1, bp1 = np.linalg.eig(Hp)
    idx = a1.argsort()[::1]   
    a1 = a1[idx]
    F=bp1[:,idx[0]]
    aa1.append(a1[0]) ##
    aa2.append(a1[1])
    aax.append(st)
    print(ff)
    Hp=csr_matrix(Hp)
    v=np.ones(2**n_qubit)
    # v=np.random.random(2**n_qubit)
    v=v/np.sqrt(v.conj().dot(v))
    ex=v.copy()
    vt=[]
    ext=[]
    # ext.append(state_fidelity(F,ex))
    # vt.append(state_fidelity(F,v))
    # ex=expm(-Hp*dt).dot(ex)
    # ex=ex/np.sqrt(ex.conj().dot(ex))
    E2=ex.conj().dot(Hp.dot(ex)).real
    # ext.append(state_fidelity(F,ex))
    ext.append(E2)
    vt.append(E2)
    beta=np.zeros(len(Hd))
    # print(' \n energy difference:',a1[1]-a1[0],
    #           '\n fidelity',state_fidelity(F,v),state_fidelity(F,ex))
    nI=0
    D=np.eye(2**n_qubit)
    Fe=1
    db=np.zeros(len(Hd))
    B=[]
    for i in range(roop):
        H=Hp.copy()
        HD=H.diagonal()
        beta=np.zeros(len(Hd))
        for k in range(len(Hd)):
            if db[k]<6:
                A=(ex.conj().dot(anticommutator(Hd[k],Hp).dot(ex))
               -2*ex.conj().dot(Hp.dot(ex))*ex.conj().dot(Hd[k].dot(ex))).real
                if abs(A)>1e-5:
                    # beta[k]=(2*S/(1+np.exp(-0.3*A.real))-S)
                    beta[k]=np.sign(A)*S
                    HD+=beta[k]*Hd[k].diagonal().real
                else:
                    db[k]+=1
        B.append(list(beta))
        if len(B)>2:
            db[(np.array(B[-1])*np.array(B[-2]))>0]=0
            db+=np.array(np.array(B[-1])*np.array(B[-2])<0)
        H.setdiag(HD)                   
        #ex=expm(-H*dt).dot(ex
        v1=H.dot(ex)*dt
        v2=H.dot(v1)*dt
        ex=ex-v1+0.5*v2
        ex=ex/np.sqrt(ex.conj().dot(ex))
        ext.append(E2)
        E2=ex.conj().dot(Hp.dot(ex)).real
        if abs(ext[-1]-ext[-2])<1e-2:
            S=0.05
        if abs(a1[0]-ext[-1])<1e-4:
            break
        print('\n',i)
        print('\n energy:', E2, '\n fidelity',state_fidelity(ex, F))
    break   
    H=Hp.copy()
    #H=expm(-H*dt)
    for i in range(roop):
        v1=H.dot(v)*dt
        v2=H.dot(v1)*dt
        v=v-v1+0.5*v2
        v=v/np.sqrt(v.conj().dot(v))
        Fe=1
        E1=v.conj().dot(Hp.dot(v)).real
        vt.append(E1)
        if abs(a1[0]-vt[-1])<1e-4:
            break
        print('\n',i)
        print('\n energy:', E1, '\n fidelity',state_fidelity(F,v))

    sd1.append(len(vt))
    sd2.append(len(ext))
    print(sd1[-1]/sd2[-1])
# dE=-1/(np.array(aa1)-np.array(aa2))
# # plt.plot(dE,SD)
"""
Final Result plot
"""
# dE=np.array([11.43389714, 10.89536515, 10.37208624,  9.86971287,  9.39174744,
#         8.94003257,  8.51518909,  8.11697818,  7.7445851 ,  7.39683262,
#         7.07233643,  6.76961478,  6.48716373,  6.22350685,  5.97722661,
#         5.74698288,  5.5315225 ,  5.32968275,  5.14039086,  4.96266104,
#         4.79558996,  4.63835138,  4.4901905 ,  4.35041825,  4.21840567,
#         4.09357864,  3.97541292,  3.86342955,  3.75719063,  3.65629545,
#         3.56037707,  3.46909903,  3.38215262,  3.29925419,  3.22014289,
#         3.14457854,  3.07233976,  3.00322226,  2.93703732,  2.87361039,
#         2.81277986,  2.75439596,  2.6983197 ,  2.64442197,  2.59258276,
#         2.54269034,  2.4946406 ,  2.44833648,  2.40368736,  2.36060854])
# sd1=np.array([439, 428, 416, 404, 392, 380, 368, 357, 346, 335, 324, 315, 305,
#         296, 287, 279, 271, 264, 257, 250, 244, 238, 232, 226, 221, 216,
#         211, 206, 202, 197, 193, 189, 185, 182, 178, 175, 172, 169, 166,
#         163, 160, 157, 155, 152, 150, 147, 145, 143, 141, 138])
# sd2=np.array([85, 78, 77, 77, 78, 78, 80, 81, 83, 83, 85, 86, 88, 88, 89, 92, 92,
#         92, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 92, 92, 92, 91, 91,
#         90, 90, 89, 89, 88, 88, 87, 87, 86, 85, 85, 84, 84, 83, 83, 82])
# from scipy.optimize import curve_fit
# def func(x, a, b):
#     return a *x+b
# def func2(x, a, b):
#     return a *np.log(x)+b
# t=np.linspace(2, 12)
# popt, pcov = curve_fit(func, dE, sd1)
# popt2, pcov = curve_fit(func, dE, sd2)
# plt.scatter(dE,sd1,c='blue', label='QITE')
# plt.scatter(dE,sd2,c='red', label='QITC')
# plt.title('3-SAT',size=30)
# plt.xlabel('1/ΔE',size=20)
# plt.ylabel('Total time steps',size=20)
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.grid()
# plt.plot(t, func(t, *popt), 'k--', label="Fitted Curve",linewidth=3,alpha=0.7)
# plt.plot(t, func(t, *popt2), 'k', label="Fitted Curve",linewidth=3,alpha=0.7)
# plt.legend(fontsize=20)
# plt.xscale('log')