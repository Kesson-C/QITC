#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:58:43 2022

@author: kesson
"""

from pyscf import gto
from pyscf import tdscf
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType, BasisType
from qiskit.chemistry import FermionicOperator
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import expm, eig
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli, state_fidelity
from itertools import permutations,combinations
import matplotlib.pyplot as plt
import time
from scipy.linalg import sqrtm
from scipy.sparse.linalg import norm
import heapq
map_type='jordan_wigner'
geometry=''
# n=4
# d=1.31
def VarH(ex):
    E=ex.conj().dot(Hp.dot(ex)).real
    E2=ex.conj().dot(Hp.dot(Hp).dot(ex)).real
    return E2-E**2
def label2Pauli(s): # can be imported from groupedFermionicOperator.py
    xs = []
    zs = []
    label2XZ = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    for c in s[::-1]:
        x, z = label2XZ[c]
        xs.append(x)
        zs.append(z)
    return Pauli(z = zs, x = xs)
def anticommutator(A,B):    return A.dot(B)+B.dot(A)
def comb(a,b,c,d):
    perm=[]
    for i in combinations(range(a),b):
        tmp=[d]*a
        for j in i:
            tmp[j]=c
        perm.append(''.join(tmp))
    return perm
def lshift(s,n):
    return s[n:]+s[:n]
def Fedtest(ex):
    SS=[]
    au=np.argsort(u)[:3]
    for x in au:
        SS.append(state_fidelity(ex,w[:,x]))
    SS=np.array(SS)
    # ww=[]
    # print(i,'---------------------')
    # for ms in heapq.nlargest(5, SS):
    #     if ms>1e-2:
    #         ww.append(np.where(SS==ms)[0][0])
    #         print(ms,u[au[ww[-1]]])
    # print('---------------------')
    return SS
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

C=0
na=3
nb=3
n=na*nb
dt=5e-2
ratio=[]
dist=[]
# h=0.8
# for i in range(5,10):
#     for j in range(5,10):
#         dist.append([i/10,j/10,j/10,j/10])
dist=[[.1,.9,.9,.9]]        
# dist=[[.2,.5,.5,.5]]  
# dist=[[1,5,5,5]]  
for nd in range(len(dist)):
    sf=[]
    cf=[]
    d=dist[nd][1:]
    h=dist[nd][0]
    J=d*np.array(h)
    print(n,d)
    edge=[]
    perms=[]
    for i in range(na):
        for j in range(nb-1):
            edge.append([i*nb+j,i*nb+np.mod(j+1,nb)])
    for i in range(na-1):
        for j in range(nb):
            edge.append([i*nb+j,np.mod((i+1),na)*nb+j])
    H2D=''
    for i in range(n):
        # tmpX='+'+str(h)
        # tmpY='+'+str(h)
        tmpZ='+'+str(h)
        for j in range(n):
            if j==i:
                # tmpX+='X'
                # tmpY+='Y'
                tmpZ+='Z'
            else:
                # tmpX+='I'
                # tmpY+='I'
                tmpZ+='I'
        H2D=H2D+tmpZ
    for i in edge:
        tmpX='+'+str(J[0])
        tmpY='+'+str(J[1])
        tmpZ='+'+str(J[2])
        for j in range(n):
            if j in i:
                tmpX+='X'
                tmpY+='Y'
                tmpZ+='Z'
            else:
                tmpX+='I'
                tmpY+='I'
                tmpZ+='I'
        H2D=H2D+tmpX+tmpY+tmpZ
      #  perms.append(tmpZ[5:])
    # H=str2WeightedPaulis('-20'+'I'*n+H2D)
    H=str2WeightedPaulis(H2D)
    # H.chop(1e-1)
    H2=H*H
    H2.chop(2e-1)
    # H4=H2*H2
    # zh2=[]
    # for i in H2.paulis:
    #     if sum(np.array(list(str(i[1])))=='Z')==3:
    #         zh2.append(str(i[1]))
    # # H2.chop(0.5e-1)
    # print(len(H2.paulis))
    # # H=str2WeightedPaulis('-30'+'I'*n+H2D)
    # H3=H2*H
    # H3.chop(2)
    # print(len(H3.paulis))
    # # H=str2WeightedPaulis('-40'+'I'*n+H2D)
    # H4=H3*H
    # H4.chop(50)
    # print(len(H4.paulis))
    # H2.paulis[0][0]=0
    # H3.paulis[0][0]=0
    # H4.paulis[0][0]=0 
    Hp=str2WeightedPaulis(H2D).to_opflow().to_spmatrix()
    # u,w=eigs(Hp,k=1,which='SR')
    # break
    u,w=np.linalg.eig(Hp.todense())
    w=np.array(w)
    
    # u,w=eigs(Hp,k=5,which='SR')
    gs=np.argsort(u)[0]
    GS=len(np.where(abs(u-u[gs])<1e-10)[0])
    fs=np.argsort(u)[1]
    roop=100000
    # 22
    # Check1='I'*(n-2)+'ZZ'    
    # 33
    Check1='I'*(n-4)+'ZIIZ'
    Check2='I'*(n-3)+'ZZZ'
    # 33 Worse
    # Check1='I'*(n-3)+'ZZZ'
    #44 
    # Check2='I'*(n-6)+'ZIZZZZ'
    # Check3='I'*(n-6)+'ZZIZZZ'
    #43
    # Check1='I'*(n-5)+'ZIIIZ'
    # Check2='I'*(n-3)+'ZZZ'
    # Check3='I'*(n-7)+'ZIIIIIZ'
    if nd==0:
        uHd=[]
        dHd=[]
        perm=[]
        perms=[]
        perms=comb(n,1,'Z','I')
        # perms=np.array(perms)
        for i in range(int(n)):
            perms.append(lshift(Check1,i))
            # perms.append(lshift(Check2,i))
            # perms.append(lshift(Check3,i))
        for i in perms:
            # if i in zh2:
            dHd.append(Pauli(i).to_matrix(sparse=1).diagonal())
                # perm.append(i)
        dHd=np.array(dHd)
        perm=np.array(perm)
    v=np.ones(2**n,dtype=np.complex128)
    v+=np.random.rand(2**n)*0.000
    v/=np.sqrt(v.conj().dot(v))
    cv=v.copy()
    #######################################################
    E=v.conj().dot(Hp.dot(v))
    vt=[E]
    pkk=[v]
    for i in range(roop):
        v1=Hp.dot(v)*dt
        # v2=Hp.dot(v1)*dt
        # v3=Hp.dot(v2)*dt
        v=v-v1#+0.5*v2#-1/6*v3
        # v=H1.dot(v)
        # v=expm(-Hp*dt).dot(v)
        v/=np.sqrt(v.conj().dot(v))
        E=v.conj().dot(Hp.dot(v))
        vt.append(E)
        # fid=state_fidelity(w[:,gs],v)#+state_fidelity(w[:,1],v)
        fid=sum(Fedtest(v)[:GS])
        sf.append(fid)
        if np.mod(i,100)==0:
            print(i,E,fid)
        if (1-sf[-1])<1e-2:
            break
        # if (vt[-1]-u[gs])<1e-4:
        #     break
        pkk.append(v)
    np.savetxt('Heisen2D/'+str(na)+str(nb)+'_vt',vt)  
    #######################################################

    v=cv.copy()
    E=v.conj().dot(Hp.dot(v))
    ext=[E]
    DR=10
    vk=Hp.dot(v)
    B=[]
    ssum=0
    on=0
    S=2
    ED=[]
    cf=[]
    pkk2=[v]
    k=1
    B1=[]
    B2=[]
    B3=[]
    B4=[]
    # H=Hp
    # H2=H.dot(H)
    # H2=sqrtm(H.todense())
    # H3=H2.dot(H)
    # H4=H3.dot(H)
    # Hd=dHd.copy()
    # Hd=[H2.to_opflow().to_spmatrix(),
    #     H3.to_opflow().to_spmatrix(),H4.to_opflow().to_spmatrix()]
    # Hd=[H2,H3,H4]
    from scipy.sparse import random 
    # D=np.zeros([2**n,2**n])
    dense=0
    D=random(2**n,2**n,density=dense).A
    # D[D!=0]=0.1
    D[gs,gs]=-5
    D[fs,fs]=5
    
    Hd=[csr_matrix(w.dot(D.dot(w.T.conj())))]
    for i in range(roop):
        beta=np.zeros(len(Hd))
        H1=Hp.copy()*1
        # A=np.zeros(len(Hd))
        # HD=H1.diagonal()
        t1=time.time()
        vv=Hp.dot(v)
        if S>1E-10:
            for j in range((len(Hd))):
                # vd=Hd[j]*(v)
                vd=Hd[j].dot(v)
                A=(vv.conj().dot(vd)+vd.conj().dot(vv)-2*E*v.conj().dot(vd)).real
                # beta[j]=S*np.sign(A)
                beta[j]=abs(2*S/(1+np.exp(-1*A.real))-S)
                # beta[j]=S
                H1+=beta[j]*Hd[j]
                # HD+=beta[j]*Hd[j]
        # B1.append(w[:,fs].conj().dot(H1.dot(w[:,fs]))-w[:,gs].conj().dot(H1.dot(w[:,gs])))
        # B2.append(w[:,fs].conj().dot(Hp.dot(w[:,fs]))-w[:,gs].conj().dot(Hp.dot(w[:,gs])))
        # B3.append(w[:,gs].conj().dot(H1.dot(w[:,gs])))
        # B4.append(norm(H1))
        # B1.append(w[:,0].conj().dot(H1.dot(w[:,0])))
        # B2.append(w[:,1].conj().dot(H1.dot(w[:,1])))
        # B3.append(w[:,2].conj().dot(H1.dot(w[:,2])))
        # B4.append(w[:,3].conj().dot(H1.dot(w[:,3])))
        B.append(beta)
        # H1.setdiag(HD)
        v1=H1.dot(v)*dt
        # v2=H1.dot(v1)*dt
        # v3=H1.dot(v2)*dt
        ex=v-v1#+0.5*v2#-1/6*v3
        v=ex.copy()
        v/=np.sqrt(v.conj().dot(v))
        vk=Hp.dot(v)
        E=v.conj().dot(vk).real
        fid=sum(Fedtest(v)[:GS])
        # fid=state_fidelity(w[:,gs],v)
        if np.mod(i,10)==0:# and i!=0:
            print(i,E,S,fid)
            if S>1e-5:
                S-=0.1
            # k=1
            # Fedtest(v)
        ext.append(E)
        cf.append(fid)
        pkk2.append(v)
        # if abs(ext[-1]-ext[-2])<1e-4:
            # S=0
        if (1-cf[-1])<1e-2:
            break
        # if (ext[-1]-u[gs])<1e-4:
        #     break
    print(len(ext))
    # ratio.append(len(vt)/len(ext))
    # plt.plot(cf,label='ITC matrix density='+str(int(dense*100))+'%',linewidth=2) 
        # if (ext[-1]-u[gs])<1e-4:
        #     break
    # np.savetxt('Heisen2D/P'+str(na)+str(nb)+'_ext',ext)  
    # ratio.append(len(vt)/len(ext))
# plt.plot(sf,label='ITE',linewidth=2)   
# plt.plot(w[:,gs],label='ground state')
# plt.plot(w[:,fs],label='1st excited state')
# plt.legend(fontsize=15)
# plt.title('Statevector',fontsize=20)

# plt.plot(cf,'QITC')
# plt.plot(sf,'QITE')
# plt.yscale('log')
# plt.xscale('log')
# plt.plot(np.ones(len(sf)))
# plt.xlabel('Total time steps(log)',fontsize=20)
# plt.ylabel('State Fidelity(log)',fontsize=20)
# plt.title('XXX model convergence test')