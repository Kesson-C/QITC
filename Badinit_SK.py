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
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import StatevectorSimulator
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
import warnings
warnings.filterwarnings("ignore")
backend = StatevectorSimulator(method="statevector")
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

def makewave(wavefunction,dele,name):
    n=wavefunction.num_qubits
    param = ParameterVector(name,int(3*n*(n-1)+6*n))
    t=0
    for i in range(n):
        if (t not in dele):
            wavefunction.rx(param[t],i)
        t+=1
    for i in range(n):
        if (t not in dele):
            wavefunction.rz(param[t],i)
        t+=1
    for i in range(n):
        for j in range(n):
            if i!=j and (t not in dele):
                wavefunction.cry(param[t],i,j)
            t+=1
    for i in range(n):
        if (t not in dele):
            wavefunction.rx(param[t],i)
        t+=1
    for i in range(n):
        if (t not in dele):
            wavefunction.rz(param[t],i)
        t+=1
    return wavefunction

def L(params,wavefunction):
    a={}
    t=0
    for i in wavefunction.parameters:
        a[i]=params[t]
        t+=1
        
    wavefunction = wavefunction.assign_parameters(a)
    u = execute(wavefunction, backend).result().get_statevector()
    # print((u.conj().dot(Hp.dot(u)).real)+shift)
    # E.append(u.conj().dot(Hp.dot(u)).real+shift)
    return u.conj().dot(Hp.dot(u)).real
    # return -state_fidelity(u,ext)

def dtheta(params,wavefunction,H):
    N=wavefunction.num_parameters
    A=np.zeros([N,N],dtype=np.complex128)
    C=np.zeros(N,dtype=np.complex128)
    dpdt=[]
    cp=1/2
    a=np.pi/2
    phi=Lv(params,wavefunction)
    for i in range(len(params)):
        ptmp1=params.copy()
        ptmp2=params.copy()
        ptmp1[i]+=a
        ptmp2[i]-=a    
        dp=cp*(Lv(ptmp1,wavefunction)-Lv(ptmp2,wavefunction))
        dpdt.append(dp)
    for i in range(len(params)):
        for j in range(len(params)):
            # phi=Lv(params,wavefunction)
            A[i,j]=(dpdt[i].conj().dot(dpdt[j])).real+dpdt[i].conj().dot(phi)*dpdt[j].conj().dot(phi)
    for i in range(len(params)):
        # phi=Lv(params,wavefunction)
        C[i]=(dpdt[i].conj().dot(H.dot(phi))).real
    dx=np.linalg.pinv(A.real).dot(-C)
    return dx.real

def Lv(params,wavefunction):
    a={}
    t=0
    for i in wavefunction.parameters:
        a[i]=params[t]
        t+=1
    wavefunction = wavefunction.assign_parameters(a)
    u = execute(wavefunction, backend).result().get_statevector()
    # u+=noise()
    u/=np.sqrt(u.conj().dot(u))
    return u

def label2Pauli(s): # can be imported from groupedFermionicOperator.py
    """
    Convert a Pauli string into Pauli object. 
    Note that the qubits are labelled in descending order: 'IXYZ' represents I_3 X_2 Y_1 Z_0
    
    Args: 
        s (str) : string representation of a Pauli term
    
    Returns:
        qiskit.quantum_info.Pauli: Pauli object of s
    """
    
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

d=np.array(range(50,200,5))/100
d=[0.74]
# Gamma=1-1e-3
vt=[]
cvt=[]
ex=[]
sc=[]
s=[]
dt=0.1
roop=1000
totalt=0
wavefunction = QuantumCircuit(4)
dele=[]
n=4
wavefunction.h(0)
wavefunction.h(1)
wavefunction.h(2)
wavefunction.h(3)
wavefunction = makewave(wavefunction, dele,1)
N=wavefunction.num_parameters
ds=[]
for dist in d:
    np.random.seed(0)
    edge=[]
    perms=[]
    for i in range(4):
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
    H2D=str2WeightedPaulis(H2D)
    ns=2**n
    Hp=H2D.to_opflow().to_matrix()
    u,w=np.linalg.eig(Hp)
    gs=np.argsort(u)[0]
    Hd=[]
    perms=['ZZZI','ZZIZ','ZIZZ','IZZZ']
    for i in perms:
        Hd.append(Pauli(i).to_matrix())
    pre=10
    ##########
    ###VQE#####
    #########
    optimizer = COBYLA(maxiter=pre)
    counts = []
    values = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
    x0=np.random.rand(N)*1
    a=VQE(H2D,wavefunction,optimizer,callback=store_intermediate_result,initial_point=x0)
    rs=a.run(backend)
    x1=np.array(list(rs.optimal_parameters.values()))
    E1=values.copy()
    ######################################
    #######QITC######
    #########################
    beta=np.zeros(len(Hd),dtype=np.complex128)
    cv=Lv(x0, wavefunction)
    S=2
    E2=[]
    for i in range(pre):
       C=(cv.conj().dot(Hp.dot(Hp)).dot(cv)-cv.conj().dot(Hp.dot(cv))**2)
       H=Hp.copy()
       cv=Lv(x0, wavefunction)
       for k in range(len(Hd)):
           A=(cv.conj().dot(anticommutator(Hd[k],Hp)).dot(cv)
                 -2*cv.conj().dot(Hp.dot(cv))*cv.conj().dot(Hd[k].dot(cv)))
           beta[k]=(2*S/(1+np.exp(-3*A.real))-S)
       for k in range(len(Hd)):
           H+=beta[k]*Hd[k]
           
       dx0=dtheta(x0, wavefunction, H)
       x0+=dx0*dt
       cv=Lv(x0, wavefunction)
       E2.append(cv.conj().dot(Hp.dot(cv)).real)
       print(i,E2[-1])
    ############################
    ###### ITE ################
    #####################
    cv=Lv(x0, wavefunction)
    v=Lv(x1, wavefunction)
    beta=np.zeros(len(Hd),dtype=np.complex128)
    # E1=[]
    # E2=[]
    ex=[u[gs]]*np.ones(roop)
    ext=w[gs]
    for i in range(roop):
        # C=(cv.conj().dot(Hp.dot(Hp)).dot(cv)-cv.conj().dot(Hp.dot(cv))**2)
        H=Hp.copy()
            
        dx0=dtheta(x0, wavefunction, H)
        x0+=dx0*dt
        cv=Lv(x0, wavefunction)
        
        H=Hp.copy()
        dx1=dtheta(x1, wavefunction, H)
        x1+=dx1*dt
        v=Lv(x1, wavefunction)
        
        E1.append(v.conj().dot(Hp.dot(v)).real)
        E2.append(cv.conj().dot(Hp.dot(cv)).real)
        
        if E1[-1]-ex[-1]<1e-3 and E2[-1]-ex[-1]<1e-3:
            break
        # print(i,'\n fidelity:',state_fidelity(ext, v),state_fidelity(ext, cv))
        print(i,E1[-1],E2[-1])
        # vt.append(E1)
        # cvt.append(E2)
        
ex=[u[gs]]*np.zeros(len(E1))   
t=np.array(list(range(len(E1))))
E1=E1-u[gs]
E2=E2-u[gs]

plt.plot(ex+10,np.linspace(min(E2), 10,len(ex)),'y--',label='Initial State Preparation',linewidth=2)
plt.plot(t,ex+1e-3,'g--',label='Chemical Accuracy',linewidth=2)
plt.plot(t,E1,'r-',label='Variational Imaginary Time (VQE initial)',linewidth=2)
plt.plot(t,E2,'b-',label='Variational Imaginary Time (Control initial)',linewidth=2)
plt.plot(t,ex+min(E2),'k-',linewidth=2)
plt.legend(fontsize=15)
plt.xlabel('Total time steps',fontsize=20)
plt.ylabel('Energy Difference From The Ground State',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Variational Ansatz Result(SK model)', fontsize=20)
plt.yscale('log')
plt.xlim([0,566])
plt.ylim([min(E2)-1E-7,10])