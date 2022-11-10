#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:58:43 2022

@author: kesson
"""
from recirq.qaoa.problems import get_all_sk_problems
import cirq
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.linalg import expm, eig
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli, state_fidelity
from itertools import permutations,combinations
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
import matplotlib.pyplot as plt
import time
from scipy.linalg import sqrtm
import heapq
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
    au=np.argsort(u)
    for x in au:
        SS.append(state_fidelity(ex,w[:,x]))
        # SS.append(ex)
    SS=np.array(SS)
    ww=[]
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
#%%
C=0
n=8
dt=3e-2
ratio=[]
var=[]
dist=list(range(100))
# dist=[ 132, 226, 205, 138, 104]
# dist=list(range(200,220))
dist=[53]
dE=[]
dif=[]
V=[]
C=[]
nselec=0
for nd in range(len(dist)):
    # V.append(len(np.loadtxt('spinglass/'+str(nd)+'_vt',dtype=np.complex)))
    # C.append(len(np.loadtxt('spinglass/'+str(nd)+'_ext',dtype=np.complex)))
    # continue
    np.random.seed(dist[nd])
    # np.random.seed(334)
    sf=[]
    cf=[]
    edge=[]
    perms=[]
    for i in range(n):
        for j in range(i):
            edge.append([j,i])
    J=(np.random.rand(len(edge))-0.5)
    J=np.round(J,5)
    H2D=''
    ZHd=''
    XHd=''
    YHd=''
    for i,k in enumerate(edge):
        if J[i]>=0:
            tmpX='+'+str(J[i])
            tmpY='+'+str(J[i])
            tmpZ='+'+str(J[i])
        else:
            tmpX=str(J[i])
            tmpY=str(J[i])
            tmpZ=str(J[i])
        for j in range(n):
            if j in k:
                tmpX+='X'
                tmpY+='Y'
                tmpZ+='Z'
            else:
                tmpX+='I'
                tmpY+='I'
                tmpZ+='I'
        H2D=H2D+tmpX+tmpY+tmpZ
           
    Hp=str2WeightedPaulis(H2D).to_opflow().to_spmatrix()
    u,w=eigs(Hp,k=10,which='SR')
    # u,w=np.linalg.eig(Hp.todense())
    gs=np.argsort(u)[0]
    fs=np.where(u==min(u[abs(u-u[gs])>1e-5]))[0]
    H=str2WeightedPaulis(H2D+'+20'+'I'*n)
    # C=[]
    # for i in H.paulis:
    #     C.append(i[0])
    # C=abs(np.array(C))
    # cp=C[np.argsort(-C)[10]]
    # H.chop(cp)
    H2=H*H
    # C=[]
    # for i in H2.paulis:
    #     C.append(i[0])
    # C=abs(np.array(C))
    # cp=C[np.argsort(-C)[10]]
    # coef=[]
    # for i in H2.paulis:
    #     coef.append(i[0])
    # coef=abs(np.array(coef))
    # cp=np.argsort(-coef)
    # H2.chop(cp)
    # H3=H2*H
    # C=[]
    # for i in H3.paulis:
    #     C.append(i[0])
    # C=abs(np.array(C))
    # cp=C[np.argsort(-C)[10]]
    # H3.chop(cp)
    # H4=H3*H
    # C=[]
    # for i in H3.paulis:
    #     C.append(i[0])
    # C=abs(np.array(C))
    # cp=C[np.argsort(-C)[10]]
    # H4.chop(cp)
    # H2.paulis[0][0]=0
    # H3.paulis[0][0]=0     
    # print(nd,(u[fs]-u[gs]).real)
    dE.append((u[fs]-u[gs]).real)
    # dE.append(np.var(u))
    # print(dE[-1])
    # if(u[fs]-u[gs]).real>.1:
    # continue
    Gsl=len(np.where(abs(u-u[gs])<1e-10)[0])
    roop=100000
    #4z
    Check1='ZZZZIIII'  
    Check2='ZIIIIIII'  
    # Check1='ZIZZIZII'  
    # Check2='ZZIZZIII'
    # Check3='ZIIZZZII'
    # Check4='ZZIIZZII'
    # Check5='ZIZZZIII'
    # Check6='ZIZZIIZI'
    # Check7='ZIIZZIZI'
    if 1:
        uHd=[]
        dHd=[]
        perm=[]
        perms=[]
        #perms+=comb(n-2,2,'XY','I')
        #perms+=comb(n-2,2,'XZ','I')
        # perms+=comb(n,1,'Z','I')
       # perms=np.array(perms)
        # perms=['ZZZI','ZZIZ','ZIZZ','IZZZ']
        # for i in cp:
        #     if sum(np.array(list(str(H2.paulis[i][1])))=='Z')==4:
        #     # and sum(np.array(list(str(H2.paulis[i][1])))=='Y')==0):
        #         dHd.append(H2.paulis[i][1].to_matrix(sparse=1))
        #         # print(H2.paulis[i][1])
        #     if len(dHd)==10:
        #         break
        for i in range(int(n)):
            perms.append(lshift(Check1,i))
            perms.append(lshift(Check2,i))
        #     perms.append(lshift(Check3,i))
        #     perms.append(lshift(Check4,i))
        #     perms.append(lshift(Check5,i))
        #     perms.append(lshift(Check6,i))
        #     perms.append(lshift(Check7,i))
        sperm=np.zeros(len(perms))
        for i in H2.paulis:
            if str(i[1]) in perms:
                lc=np.where(np.array(perms)==str(i[1]))[0]
                sperm[lc]=np.sign(i[0])
        for i in perms:
            dHd.append(Pauli(i).to_matrix(sparse=1))
            # dHd.append(str2WeightedPaulis(i).to_opflow().to_spmatrix())
            # perm.append(i)
        # dHd=np.array(dHd)
        perm=np.array(perm)
    
    # v=np.ones(2**n,dtype=np.complex128)
    # v*=0
    # v[0]=1
    # v[10]=100
    ratiotmp=[]
    diftmp=[]
    sp=10
    sample=0
    adj=5
    Sadj=1.5
    # H2=Hp.dot(Hp).toarray()
    # H2[abs(H2)<.5]=0
    # H2=csr_matrix(H2)
    # H3=H2.dot(Hp).toarray()
    # H3[abs(H3)<.5]=0
    # H3=csr_matrix(H3)
    # H4=H3.dot(Hp).toarray()
    # H4[abs(H4)<.5]=0
    # H4=csr_matrix(H4)
    # H5=H4.dot(Hp).toarray()
    # H5[abs(H5)<.5]=0
    # H5=csr_matrix(H5)
    # Hd=[0.5*H2,0.17*H3,0.04*H4,0.008*H5]
    while sample < sp:
        # np.random.seed())
        v=np.random.rand(2**n)
        v/=np.sqrt(v.conj().dot(v))
        cv=v.copy()
        ########################################################
        E=v.conj().dot(Hp.dot(v))
        vt=[E]
        pkk=[v]
        for i in range(roop):
            v1=Hp.dot(v)*dt
            v2=Hp.dot(v1)*dt
            # v3=Hp.dot(v2)*dt
            v=v-v1+0.5*v2#+1/6*v3
            # v=H1.dot(v)
            # v=expm(-Hp*dt).dot(v)
            v/=np.sqrt(v.conj().dot(v))
            E=v.conj().dot(Hp.dot(v))
            vt.append(E)
            # fid=state_fidelity(w[:,gs],v)#+state_fidelity(w[:,1],v)
            fid=sum(Fedtest(v)[:Gsl])
            # fid=sum(Fedtest(v)[:4])
            sf.append(fid)
            # Fedtest(v)
            # if np.mod(i,100)==0:
            #     print(i,E)
            # if (1-sf[-1])<1e-2:
            #     break
            if (vt[-1]-u[gs])<1e-4:
                break
            pkk.append(v)
        #######################################################
        v=cv.copy()
        E=v.conj().dot(Hp.dot(v))
        ext=[E]
        DR=10
        vk=Hp.dot(v)
        B=[np.zeros(len(dHd))]
        ssum=0
        on=0
        ED=[]
        pkk2=[v]
        k=1
        S=Sadj
        # H=csr_matrix(Hp)
        Hd=dHd.copy()
        for i in range(roop):
            beta=np.zeros(len(Hd))
            H1=Hp.copy()*k
            # A=np.zeros(len(Hd))
            # HD=H1.diagonal()
            t1=time.time()
            vv=Hp.dot(v)
            if S>1E-10:
                for j in range((len(Hd))):
                    # vd=Hd[j].diagonal()*(v)
                    vd=Hd[j].dot(v)
                    A=(vv.conj().dot(vd)+vd.conj().dot(vv)-2*E*v.conj().dot(vd)).real
                    beta[j]=(2*S/(1+np.exp(-adj*A.real))-S)
                    # if np.sign(sum(B)[j])==sperm[j]:
                    # beta[j]=S*np.sign(A)
                    H1+=beta[j]*Hd[j]
                    # HD+=beta[j]*Hd[j].diagonal()
            B.append(beta)
            # H1.setdiag(HD)
            v1=H1.dot(v)*dt
            v2=H1.dot(v1)*dt
            # v3=H1.dot(v2)*dt
            ex=v-v1+0.5*v2
            v=ex.copy()
            v/=np.sqrt(v.conj().dot(v))
            vk=Hp.dot(v)
            E=v.conj().dot(vk)
            # fid=sum(Fedtest(v)[:4])
            fid=sum(Fedtest(v)[:Gsl])
            if np.mod(i,100)==0 and i!=0:
                if S>1e-5:
                    S=0
                    # S=0
                    # k+=0.1
            # if np.mod(i,100)==0:
            #     print(i,E,S)
            ext.append(E)
            cf.append(fid)
            pkk2.append(v)
            # if abs(ext[-1]-ext[-2])<1e-4:
                # S=0
                # k=1
                # if S<0:
                # S=0
            # if (1-cf[-1])<1e-2:
            #     break
            if len(ext)>(len(vt)*3):
                sample=sp-2
                break
            if (ext[-1]-u[gs])<1e-4:
                break
        print(len(vt),len(ext))
        ratiotmp.append(len(vt)/len(ext))
        # ratiotmp[0]+=len(vt)
        # ratiotmp[1]+=len(ext)
        diftmp.append(len(vt)-len(ext))
        sample+=1
        if sample==sp-1:
            if np.mean(ratiotmp)<1:
                sample=0
                ratiotmp=[]
                Sadj*=0.8
        #     adj/=2
        #     sample-=1
        #     print(adj)
        # else:
            # ratiotmp.append(len(vt)/len(ext))
            # sample+=1
            # diftmp.append(len(vt)-len(ext))
    # if nd==len(dist):
    #     plt.plot(ext,color='r',alpha=1,label='QITC')
    #     plt.plot(vt,color='b',alpha=1,label='QITE')
    # else:
    #     plt.plot(ext,color='r',alpha=1)
    #     plt.plot(vt,color='b',alpha=1)
    # ratio.append(len(vt)/len(ext))
    # dif.append(len(vt)-len(ext))
    ratio.append(np.mean(ratiotmp))
    # ratio.append((ratiotmp[0])/(ratiotmp[1]))
    dif.append(sum(diftmp)/sp)
    var.append(ratiotmp)
    # if ratio[-1]>2:
    #     np.savetxt('spinglass/H'+str(nselec)+'_vt',vt)  
    #     np.savetxt('spinglass/H'+str(nselec)+'_ext',ext)  
    #     nselec+=1
    #     dE.append((u[fs]-u[gs]).real)
    #     V.append(len(vt))
    #     C.append(len(ext))
    print(ratio[-1],var[-1])
# dE=np.array(dE).reshape(-1)
# ratio=np.array(ratio)
# Esort=np.argsort(1/dE)
# plt.scatter(1/dE,V)
# plt.scatter(1/dE,C)
# plt.scatter(list(range(len(V))),V,label='ITE')
# plt.scatter(list(range(len(C))),C,label='ITC')
# plt.plot(np.mean(V)*np.ones(len(V)),label='Mean ITE',color='k')
# plt.plot(np.mean(C)*np.ones(len(C)),label='Mean ITC',color='blue')
# nvar=[]
# for i in var:
#     nvar.append(max(i))
# np.savetxt('spinglass/ratio_HZ',ratio)
# np.savetxt('spinglass/ratio_bHZ',nvar)

ratio1=np.loadtxt('spinglass/ratio_HZ')
br1=np.loadtxt('spinglass/ratio_bHZ')
ratio2=np.loadtxt('spinglass/ratio_Z')
br2=np.loadtxt('spinglass/ratio_bZ')
plt.plot(np.sort(ratio1),'r',label='p(H)_mean')
plt.plot(np.sort(br1),'r--',label='p(H)_best')
plt.plot(np.sort(ratio2),'b',label='$\sigma$(H)_mean')
plt.plot(np.sort(br2),'b--',label='$\sigma$(H)_best')
plt.plot(np.ones(len(dist)),label='QITE',color='k')
plt.legend(fontsize=15)
plt.xlabel('Index of Cases',fontsize=20)
plt.ylabel('$T_{QITE}$/$T_{QITC}$ (Total time steps)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Spin Glass (SK model)', fontsize=20)
plt.yscale('log')

