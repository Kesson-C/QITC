#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 01:58:49 2022

@author: kesson
"""
import openfermion as of
from openfermion import FermionOperator, MolecularData, QubitOperator
from openfermion import get_ground_state, get_sparse_operator
from openfermion.transforms import jordan_wigner, get_fermion_operator, reverse_jordan_wigner
import numpy as np
import fqe
from openfermionpyscf import run_pyscf
from openfermion.linalg import expectation
from openfermion.chem.molecular_data import spinorb_from_spatial
from qiskit.quantum_info.operators import Pauli
import matplotlib.pyplot as plt
import time
# geometry = [['H', [0, 0, 0]], ['H', [0, 0, 1.4]]]
# dist=[0.6,0.8,1.0,1.2,1.4,1.6,2.4,2.6]
dist=[1.47,1.445,1.31,1.305,1.19,1.192,1.08,1.06,1]
# dist=[1,1,1,1,1,1,1,1,1]
nelec=[2,3,4,5,6,7,8,9,10]
# dist=[1.305]
# nelec=[5]
step=[]
for nd in range(len(nelec)):
    d=dist[nd]
    n=nelec[nd]
    print(n,d)
    N=2*n
    cn=int(N+N*(N-1)/2)
    geometry=[]
    for i in range(n):
        geometry.append(('H', (0, 0, i*d)))
    multiplicity = 1
    basis = 'sto-3g'
    charge = np.mod(n,2)
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule = run_pyscf(molecule,run_fci = 1)
    E0=molecule.fci_energy
    dt=1e-2
    wfn = fqe.Wavefunction([[n-charge, 0, n]],broken=['number'])
    state=np.ones(2**(2*n),dtype=np.complex128)/np.sqrt(2**(2*n))
    fqe.transform.from_cirq(wfn, state)
    iop = molecule.get_molecular_hamiltonian()
    H=get_fermion_operator(iop)
    # t1=time.time()
    # Hp=get_sparse_operator(H)
    # t2=time.time()
    J=jordan_wigner(H)
    roop=100000
    Hd=[]
    V=np.array(list(J.terms.values()))
    print(2*n/len(V))
    NHD=int(len(V)/10)
    coefs=np.argsort(-abs(V[1:]))[:NHD]+1
    P=list(J.terms.keys())
    # for i in coefs:
    #     c=[]
    #     p=[]
    #     for j in P[i]:
    #         c.append(j[0])
    #         p.append(j[1])
    #     tmp=''
    #     for j in range(2*n):
    #         if j in c:
    #             s=np.where(np.array(c)==j)[0][0]
    #             tmp+=p[s]
    #         else:
    #             tmp+='I'
    #     A=Pauli(tmp)
    #     Hd.append(A.to_matrix(sparse=True))
    Hd=[]
    perms=[]
    for i in range(2*n):
        tmp='1'
        for j in range(2*n):
            if i==j:
                tmp+='Z'
            else:
                tmp+='I'
        perms.append(tmp)

    for i in perms:
        tmp=Pauli(i).to_spmatrix()
        Hd.append(tmp)
        
    Hp = fqe.build_hamiltonian(H,conserve_number=0)
    H1 = fqe.build_hamiltonian(H*dt,conserve_number=0)
    E=wfn.expectationValue(Hp)
    vt=[E]
    ext=[E]
    # vt=np.loadtxt('H10/FQE/Htest'+str(n)+'_vt')
    for i in range(roop):
        t1=time.time()
        v1=wfn.apply(H1)
        # v2=v1.apply(H1)
        v=fqe.to_cirq(wfn)-fqe.to_cirq(v1)#+0.5*fqe.to_cirq(v2)
        wfn = fqe.Wavefunction([[n-charge, 0, n]],broken=['number'])
        v/=np.sqrt(v.conj().dot(v))
        fqe.transform.from_cirq(wfn, v)
        E=wfn.expectationValue(Hp)
        print(len(vt),E)
        t2=time.time()
        vt.append(E)
        if abs(vt[-1]-E0)<1e-3:
            break
        
    # np.savetxt('H10/FQE/Htest'+str(n)+'_vt', np.array(vt).real)
    # S=abs(E0)
    S=1
    wfn = fqe.Wavefunction([[n-charge, 0, n]],broken=['number'])
    state=np.ones(2**(2*n),dtype=np.complex128)/np.sqrt(2**(2*n))
    fqe.transform.from_cirq(wfn, state)
    if n==10:
        break
    for i in range(roop):
        t1=time.time()
        beta=np.zeros(len(Hd))
        v1=wfn.apply(H1)
        # v2=v1.apply(H1) 
        ket=fqe.to_cirq(v1)
        bra=fqe.to_cirq(wfn)
        v=bra-fqe.to_cirq(v1)#+0.5*fqe.to_cirq(v2)
        for j in range(len(Hd)):
            A=2*(bra.conj().dot(Hd[j].dot(ket))-bra.conj().dot(ket)*bra.conj().dot(Hd[j].dot(bra))).real
            beta[j]=(2*S/(1+np.exp(-1*A.real))-S)
            # beta[j]=A
            v-=beta[j]*Hd[j].dot(bra)
        #     v+=beta[j]*Hd[j].dot(ket)
        # for j in range(len(Hd)):
        #     for k in range(len(Hd)):
        #         v+=0.5*(beta[j]*beta[k])*Hd[j].dot(Hd[k]).dot(bra)
        wfn = fqe.Wavefunction([[n-charge, 0, n]],broken=['number'])
        v/=np.sqrt(v.conj().dot(v))
        fqe.transform.from_cirq(wfn, v)
        E=wfn.expectationValue(Hp)
        print(len(ext),E)
        ext.append(E)
        t2=time.time()
        # print(t2-t1)
        cdt=t2-t1
        if (ext[-1]-E0)<1e-3:
            break

    # np.savetxt('H10/FQE/Htest'+str(n)+'_ext', np.array(ext).real)
    print('Step:',len(ext)-len(vt),'E:',ext[-1]-vt[-1])
    step.append((len(vt)-len(ext))/len(vt))
    print(step[-1])
# plt.scatter(nelec,step)
# plt.ylabel('time difference')
# plt.xlabel('bond length')
    # vt=np.loadtxt('H10/H10_'+str(d)+'(vt).txt')
    # ext=np.loadtxt('H10/H10_'+str(d)+'(ext).txt')
