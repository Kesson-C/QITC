# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 09:31:33 2021

@author: User
"""

from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType, BasisType
from qiskit.chemistry import FermionicOperator
from qiskit.aqua.algorithms import NumPyEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.providers.aer import StatevectorSimulator
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import state_fidelity
from scipy.linalg import expm
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
import matplotlib.animation as animation
# from modestga import minimize
from qiskit.visualization import plot_histogram
import time
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
import heapq
from itertools import permutations
map_type='jordan_wigner'
backend = StatevectorSimulator(method="statevector")
I  = np.array([[ 1, 0],
               [ 0, 1]])
Sx = np.array([[ 0, 1],
               [ 1, 0]])
Sy = np.array([[ 0,-1j],
               [1j, 0]])
Sz = np.array([[ 1, 0],
               [ 0,-1]])
PS=[I,Sz,Sx,Sy]
def get_qubit_op(dist):
    driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist)+";H .0 .0 "+str(dist*2)
                          +";H .0 .0 "+str(dist*3)#+";H .0 .0 "+str(dist*4)+";H .0 .0 "+str(dist*5)
                          #+";H .0 .0 "+str(dist*6)+";H .0 .0 "+str(dist*7)+";H .0 .0 "+str(dist*8)+";H .0 .0 "+str(dist*9)
                          , unit=UnitsType.ANGSTROM,hf_method=HFMethodType.UHF
                        , spin=0,charge=0, basis='sto3g')
    # driver = PySCFDriver(atom="H .0 .0 .0; F .0 .0 " + str(dist)
    #                       # +";H .0 " + str(np.sqrt(3)*dist/2)+" " +str(dist/2)#+";H .0 .0 "+str(dist*3)+";H .0 .0 "+str(dist*4)#+";H .0 .0 "+str(dist*5)
    #                     , unit=UnitsType.ANGSTROM,hf_method=HFMethodType.ROHF
    #                     , spin=0,charge=0, basis='sto3g')
    molecule = driver.run()
    repulsion_energy = molecule.nuclear_repulsion_energy
    freeze_list = [0,1]
    # freeze_list = []
    remove_list = []
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
    #g = groupedFermionicOperator(ferOp, num_particles)
    #qubitOp = g.to_paulis()
    # shift =  repulsion_energy
    shift = energy_shift + repulsion_energy
    return qubitOp, num_particles, num_spin_orbitals, shift

def makewave(wavefunction,dele,name):
    n=wavefunction.num_qubits
    param = ParameterVector(name,int(3*n*(n-1)+6*n))
    t=0
    for i in range(n):
        wavefunction.rz(param[t],i)
        t+=1
    for i in range(n):
        wavefunction.rx(param[t],i)
        t+=1
    for i in range(n):
        for j in range(n):
            if i!=j and (t not in dele):
                wavefunction.cry(param[t],i,j)
            t+=1
    for i in range(n):
        wavefunction.rz(param[t],i)
        t+=1
    for i in range(n):
        wavefunction.rx(param[t],i)
        t+=1
    return wavefunction

def L(params,wavefunction):
    wavefunction=Lw(params,wavefunction)
    u=np.zeros(2**wavefunction.num_qubits)
    u=execute(wavefunction, backend).result().get_statevector()
    return u

def dtheta2(params,wavefunction,H):
    N=wavefunction.num_parameters
    A=np.zeros([N,N],dtype=np.complex128)
    C=np.zeros(N,dtype=np.complex128)
    dpdt=[]
    cp=1/2
    a=np.pi/2
    for i in range(len(params)):
        ptmp1=params.copy()
        ptmp2=params.copy()
        ptmp1[i]+=a
        ptmp2[i]-=a    
        dp=cp*(Lv(ptmp1,wavefunction)-Lv(ptmp2,wavefunction))
        dpdt.append(dp)
    for i in range(len(params)):
        for j in range(len(params)):
            phi=Lv(params,wavefunction)
            A[i,j]=(dpdt[i].conj().dot(dpdt[j])).real+dpdt[i].conj().dot(phi)*dpdt[j].conj().dot(phi)
    for i in range(len(params)):
        phi=Lv(params,wavefunction)
        C[i]=(dpdt[i].conj().dot(H.dot(phi))).real
    dx=np.linalg.inv(A.real).dot(-C)
    return dx.real


def Lw(params,wavefunction):
    a={}
    t=0
    for i in wavefunction.parameters:
        a[i]=params[t]
        t+=1
    wavefunction = wavefunction.assign_parameters(a)
    return wavefunction

def Lv(params,wavefunction):
    u=np.zeros(2**wavefunction.num_qubits)
    wavefunction=Lw(params,wavefunction)
    u=execute(wavefunction, backend).result().get_statevector()
    un=u+noise()
    u=un/np.sqrt(un.conj().dot(un))
    return u

def anticommutator(A,B):
    return A.dot(B)+B.dot(A)

def commutator(A,B):
    return A.dot(B)-B.dot(A)

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

def noise():
    return np.random.normal(0,0,size=2**qubitOp.num_qubits)
                
                
# d=np.array(range(50,290,10))/100
# d=np.array(list(range(200,400,10)))/100
d=[1.7,1.75,1.8,1.85,1.9,2,2.05,2.1,2.2,2.3,2.33,2.36,2.39,2.42,2.44,2.46]
d+=[2.502,2.522,2.532,2.552]
# d=[1.7,1.8,1.9,2.0,2.1,2.2]
totalt=0
wavefunction = QuantumCircuit(8)
wavefunction.x(0)
wavefunction.x(1)
wavefunction.x(2)
wavefunction.x(3)
dele=[]
wavefunction = makewave(wavefunction, dele,1)
N=wavefunction.num_parameters
x0=np.random.rand(N)*0.01
# psif[5]=1
SD1=[]
SD2=[]
SD3=[]
K=10
scat=0
CEIPR=[]
IEIPR=[]
NUM=0
for dist in d:
    qubitOp, num_particles, num_spin_orbitals, shift=get_qubit_op(dist)
    ns=2**qubitOp.num_qubits
    Hp=qubitOp.to_opflow().to_matrix()
    # Hp=spinchain(10,1,1)
    Hd=[]
    perms=[]
    coefs=[]
   # for i in qubitOp.paulis:
    #    coefs.append(abs(i[0]))
    #perm=np.argsort(coefs)[:4]
    #print(perm)
    #for i in perm:
    #    perms.append(str(qubitOp.paulis[i][1]))
    #    print(perms[-1])
    #    Hd.append(str2WeightedPaulis(str(1)+perms[-1]).to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIIZIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIZIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IZIIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1ZIIIIIII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IIIIIIIZ').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IIIIIIZI').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IIIIIZII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IIIIZIII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IIIZIIII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IIZIIIII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1IZIIIIII').to_opflow().to_matrix())
    Hd.append(str2WeightedPaulis('1ZIIIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIIIIIIZ').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIIIIIZI').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIIIIZII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIIIZIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIIZIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIZIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IZIIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1ZIIIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IZIIIIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1ZIIIIIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIII').to_opflow().to_matrix())
    # Hd[0]=-Hd[0]
    # Hd[0][5]=1
    # Hd[0][10]=2
    # perms += list(set([''.join(p) for p in permutations('IIZZ')]))
    
    # Hd.append(str2WeightedPaulis('1IZZI').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIZZ').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIZZ').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1ZIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIIZ').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IIZI').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1IZII').to_opflow().to_matrix())
    # perms=['IIIZ','IIZI','IZII','ZIII']
    # co=[]
    # for i in qubitOp.paulis:
    #     # co2+=abs(i[0])
    #     if str(i[1]) in perms:
    #         co.append(np.sign(i[0]))
    #         Hd.append(str2WeightedPaulis('1'+str(i[1])).to_opflow().to_matrix())    
    # Hd.append(str2WeightedPaulis('1IZIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1ZIIIII').to_opflow().to_matrix())
    # Hd.append(str2WeightedPaulis('1ZZZZZZ').to_opflow().to_matrix())
    # Hd=[sum(np.array(Hd))]
    beta=np.zeros(len(Hd))
    F = NumPyEigensolver(qubitOp).run().eigenstates.to_matrix().reshape(-1)
    u,w=np.linalg.eig(Hp)
    K=np.where(u==min(u))[0][0]
    F = w[:,K]
    # plt.plot(F)
    us=np.sort(u)
    # K0=np.where(u==us[0])[0][0]
    # K1=np.where(u==us[1])[0][0]
    # K2=np.where(u==us[40])[0][0]
    # K3=np.where(u==us[150])[0][0]
    # fe2 = w[:,K3]
    # Fe2 = fe2.conj().dot(Hp.dot(fe2))+shift
    # fe1 = w[:,K2]
    # Fe1 = fe1.conj().dot(Hp.dot(fe1))+shift
    # fe = w[:,K1]
    # Fe = fe.conj().dot(Hp.dot(fe))+shift
    # fg = w[:,K0]
    # Fg = fg.conj().dot(Hp.dot(fg))+shift
    dE=[]
    for i in range(1,len(us)):
        dE.append(us[i]-us[i-1])
    
    # F = w[:,3]
    print('Distance: ',dist,'Smallest dE',dE[0])
    x1=x0.copy()
    x2=x0.copy()
    # v=Lv(x1, wavefunction)
    # ex=Lv(x2, wavefunction)
    v=np.ones(ns)
    # v[1]=1
    # v[]
    # v+=np.random.rand(ns)*0.5
    # v[0]=1*K
    v=v/np.sqrt(v.conj().dot(v))
    # K+=5
    ex=v.copy()
    # plt.plot(ex)
    dt=5e-1
    S=0.05
    vt=[]
    cvt=[]
    ext=[]
    s=[]
    si=[]
    roop=1000000
    # ext.append(ex.conj().dot(Hp.dot(ex)).real+shift)
    # vt.append(v.conj().dot(Hp.dot(v)).real+shift)
    ext.append(state_fidelity(F,ex))
    vt.append(state_fidelity(F,v))
    si.append(v)
    s.append(ex)
    # H0=[fg.dot(Hp.dot(fg))]
    # H1=[fe.dot(Hp.dot(fe))]
    # H2=[fe1.dot(Hp.dot(fe1))]
    # H3=[fe2.dot(Hp.dot(fe2))]
    # B=[np.mean(beta)]
    scat2=0
    H2=qubitOp*qubitOp
    # Hdp=str2WeightedPaulis('1IIIIIIIZ')*qubitOp
    # S1=len((H2+qubitOp).paulis)
    # S2=len(Hd)*(len((Hdp+qubitOp).paulis)-len(Hdp.paulis))+len(qubitOp.paulis)
    NM=wavefunction.num_parameters
    ST=2*NM+len(qubitOp.paulis)*NM+NM*NM
    sumC=0
    sumI=0
    IE=[]
    CE=[]
    SS=[]
    E1=[v.conj().dot(Hp.dot(v)).real+shift]
    E2=[ex.conj().dot(Hp.dot(ex)).real+shift]
    #for x in range(len(w)):
    #    SS.append(state_fidelity(ex,w[:,x]))
    #SS=np.array(SS)
    #CE.append(sum(SS[SS>1E-2]**4))
    
    #SS=[]
    #for x in range(len(w)):
    #    SS.append(state_fidelity(v,w[:,x]))
    #SS=np.array(SS)
    #IE.append(sum(SS[SS>1E-2]**4))
    
    print('initial \n energy:',vt[-1],ext[-1],'\n energy difference:',ext[-1]-vt[-1],
              '\n fidelity',state_fidelity(F,v),state_fidelity(F,ex))
    for i in range(roop):
        # Cp=(ex.conj().dot(anticommutator(Hp,Hp)).dot(ex)-2*ex.conj().dot(Hp.dot(ex))**2)
        for k in range(len(Hd)):
            etmp=Hd[k].diagonal()*(ex)
            A=(etmp.conj().dot(Hp.dot(ex))+ex.conj().dot(Hp.dot(etmp))
              -2*ex.conj().dot(Hp.dot(ex))*ex.conj().dot(etmp))
            # beta[k]=(2*S/(1+np.exp(-10*A.real))-S)
            if abs(A)>0.1:
                beta[k]=S/A
            else:
                beta[k]=S/0.1*np.sign(A)
        # sumC+=S2
        H=Hp.copy()
        for k in range(len(Hd)):
            H=H+beta[k]*Hd[k]
        # HD=H.diagonal()
        # if max(abs(beta))>1e-1:
        # for k in range(len(Hd)):
        #     HD=HD+beta[k]*Hd[k].diagonal()
            
        # np.fill_diagonal(H,HD)
        # sumC+=ST
        scat+=1
        # B.append(np.mean(beta))
        # H0.append(fg.dot(H.dot(fg)))
        # H1.append(fe.dot(H.dot(fe)))
        # H2.append(fe1.dot(H.dot(fe1)))
        # H3.append(fe2.dot(H.dot(fe2)))
        # print('\n',i)
        ex=expm(-H*dt).dot(ex)
        ex=ex/np.sqrt(ex.conj().dot(ex))
        
        # SS=[]
        # for x in range(len(w)):
        #     SS.append(state_fidelity(ex,w[:,x]))
        # SS=np.array(SS)
        # CE.append(sum(SS**4))
        
        ext.append(state_fidelity(F,ex))
        s.append(ex)
        Fe=1
        E2.append(ex.conj().dot(Hp.dot(ex)).real)
        if abs(E2[-2]-E2[-1])<1e-2:
            S=0
        if abs(E2[-1]-us[0])<1e-3:
            break
    
        # if E2[-1]-E2[-2]>-1e-6:
        #     break
        if np.mod(i,100)==0:
            print(i,'\n energy:', E2[-1])#, 'Total Measurement:',sumC)
    #print('E2:',len(E2))
    #np.savetxt('HF/HF_'+str(NUM)+'_ext',ext)
    H= expm(-Hp*dt)
    for i in range(roop):
        # H=Hp.copy()
        v=H.dot(v)
        v=v/np.sqrt(v.conj().dot(v))
        vt.append(state_fidelity(F,v))
        si.append(v)
        Fe=1
        E1.append(v.conj().dot(Hp.dot(v)).real)
      #  SS=[]
      #  for x in range(len(w)):
      #      SS.append(state_fidelity(v,w[:,x]))
      #  SS=np.array(SS)
      #  IE.append(sum(SS**4))
        
        if abs(E1[-1]-us[0])<1e-3:
            break
        # if E1[-1]-E1[-2]>-1e-6:
        #     break
        # sumI+=ST
        # print('\n',i)
        if np.mod(i,100)==0:
            print(i,'\n energy:', E1[-1])#, 'Total Measurement:',sumI)
    print('E1:',len(E2)/len(E1),E1[-1],E2[-1],us[0].real,dE[-1])
    #SD1.append(len(E2)/len(E1))
    SD2.append(1/dE[0])
    # np.savetxt('HF/HF_'+str(NUM)+'_vt',vt)
    SD1.append(len(vt))
    SD3.append(len(ext))
    NUM+=1
  
plt.plot(SD2,SD1,label='non control')
plt.plot(SD2,SD3,label='control')
# plt.yscale('log')
#plt.savefig('a.png')
    # SD1.append(len(E2))
    # SD2.append(dE[0])
    # plt.scatter(-dE[0],sumI,color='red')
    # plt.scatter(-dE[0],sumC,color='blue')
    # CEIPR.append(CE)
    # IEIPR.append(IE)
    # print(len(CE)-len(IE))
    # print(len(E1)-len(E2),dE[0])
    # plt.scatter(dE[0],(len(IE)-len(CE))*dt)
    # SD1.append([-dE[0],sumI])
    # SD2.append([-dE[0],sumC])
    
# Fe=F.conj().dot(Hp.dot(F)).real+shift
    # SD.append((sum(np.array(ext)-Fe>-1e-2)-sum(np.array(vt)-Fe>-1e-2)))
    # print(us[0]-us[1],SD[-1])
    # Fe=F.conj().dot(Hp.dot(F)).real+shift
    # t=np.array(range(len(ext)))*dt
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim([0,1.01])
    # plt.plot(t,np.ones(len(t))*Fe)
    # plt.plot(t,vt, label='Imaginary')#, color='red')
    # plt.plot(t,ext, label='Imaginary Control')#, color='blue')
    # plt.title('State Fidelity')
    # plt.legend()
    
# Fe=1
# # # Fe=F.conj().dot(Hp.dot(F)).real+shift
# # t=np.array(range(len(ext)))
# # plt.xscale("log")
# # plt.yscale("log")
# plt.ylim([0,1.05])
# plt.plot(t,Fe*np.ones(len(t)),label='Exact State')
# plt.plot(t,vt,label='Original')
# plt.plot(t,ext,label='Control')
# plt.title('State Fidelity')
# u,w=np.linalg.eig(Hp)
# ax=[]
# HIST_BINS = np.array(list(range(ns)))
# for i in s:
#     pb=[]
#     for j in range(len(w)):
#         pb.append(state_fidelity(i,w[:,j]))
#     ax.append(pb)
    
# def update_hist(num, data):
#     plt.cla()
#     plt.ylim([-0.03,1.03])
#     plt.title(num)
#     plt.xticks(np.arange(min(HIST_BINS), max(HIST_BINS)+1, 1.0))
#     plt.scatter(HIST_BINS,data[num])
#     n=data[num]
#     for i, txt in enumerate(n):
#         plt.annotate("{:.3f}".format(txt), (HIST_BINS[i], data[num][i]))
    
# data=ax[::1]
# fig = plt.figure()
# plt.ylim([-0.03,1.03])
# hist = plt.scatter(HIST_BINS,data[0])
# n=data[0]
# for i, txt in enumerate(n):
#     plt.annotate("{:.1f}".format(txt), (HIST_BINS[i], data[0][i]))
       
# animations = animation.FuncAnimation(fig, update_hist, len(data), fargs=(data, ), repeat='true')
# # animations.save('animationIC.gif', writer='imagemagick', fps=5)
# plt.show()

# oneptest(wavefunction,x2)
# from qiskit.tools.visualization import circuit_drawer
# circuit_drawer(wavefunction, output='mpl', style={'backgroundcolor': '#EEEEEE'})
# from scipy.optimize import curve_fit
# def func(x, a, b, c):
#     return a*x**2+b*x+c
# x=[0.0463,0.0388,0.0323,0.0269,0.0222,0.0151,0.0124,0.0101,0.00676,0.00447
#     ,0.00394,0.00347,0.00306,0.00269,0.00247,0.00227,0.00208,0.00199,0.00189,0.00174,0.00166]
# x=1/np.array(x)
# y=[49,61,76,95,119,189,241,308,511,865,1020,1208,1439,1726,1959,2240,2586,2796,3096,3819,4475]
# t=np.linspace(20,620)
# popt, pcov = curve_fit(func, x, y)
# plt.scatter(x,y,c='red', label='Simulation Results')
# plt.title('(b)',size=30)
# plt.xlabel('1/ΔE',size=20)
# plt.ylabel('Δτ',size=20)
# plt.xticks(size=20)
# plt.yticks(size=20)
# # plt.xlim([-0.051,0])
# plt.grid()
# plt.plot(t, func(t, *popt), 'k--', label="Fitted Curve",linewidth=3,alpha=0.7)
# plt.legend(fontsize=20)
# from scipy.optimize import curve_fit
# def func(x, a, b):
#     return a *x+b
# x=np.loadtxt('dE.txt')
# y=np.loadtxt('SD.txt')
# x=-1/x
# t=np.linspace(6, 15)
# popt, pcov = curve_fit(func, x, y)
# plt.scatter(x,y,c='red', label='Simulation Results')
# plt.title('3-SAT',size=30)
# plt.xlabel('1/ΔE',size=20)
# plt.ylabel('Δτ',size=20)
# # plt.xticks(size=20)
# # plt.yticks(size=20)
# # plt.grid()
# plt.plot(t, func(t, *popt), 'k--', label="Fitted Curve",linewidth=3,alpha=0.7)
# plt.legend(fontsize=20)
# plt.yscale('log')
# plt.xscale('symlog')

# plt.plot(t,t*0+H0[0],c='r',linewidth=2,label='Ground State')
# plt.plot(t,t*0+H1[0],c='b',linewidth=2,label='1st Excited State')
# plt.plot(t,t*0+H2[0],c='g',linewidth=2,label='40th Excited State')
# plt.plot(t,t*0+H3[0],c='k',linewidth=2,label='150th Excited State')
# plt.plot(t,np.array(H0),'r--',linewidth=2,alpha=0.8)
# plt.plot(t,np.array(H1),'b--',linewidth=2,alpha=0.8)
# plt.plot(t,np.array(H2),'g--',linewidth=2,alpha=0.8)
# plt.plot(t,np.array(H3),'k--',linewidth=2,alpha=0.8)
# plt.yticks(size=15,rotation=90)
# plt.xticks(size=15)
# plt.legend(fontsize='x-large')
# plt.ylabel('E',rotation=0,size=30,labelpad=20)
# plt.xlabel('τ',size=30)

# x=[0,1,2,3,4,7,10,12,15,20,23,26,31,36,41,46,49,50]
# y=[0.8288,0.9587,0.9839,0.9935,0.9954,0.9970,0.9975,0.9977,0.9979,0.9981,0.9981,0.9982,0.9982,0.9983,0.9983,0.9983,0.9983,0.9983]
# plt.plot(x,y,'r-o',label='ΔE=0.4169')
# plt.plot(np.zeros(50)+3,np.linspace(0, 1),'r-.',linewidth=2,alpha=0.5)
# plt.plot(x,np.ones(len(x))*y[-1],'r--',linewidth=2)
# x=[0,1,2,3,4,7,10,12,15,20,23,26,31,36,41,46,49,50]
# y=[0.2100,0.2906,0.4149,0.5805,0.7436,0.9184,0.9481,0.9503,0.9493,0.9482,0.9522,0.9541,0.9562,0.9578,0.9587,0.9584,0.9591,0.9600]
# plt.plot(x,y,'b-o',label='ΔE=0.0897')
# plt.plot(x,np.ones(len(x))*y[-1],'b--',linewidth=2)
# plt.plot(np.zeros(50)+36,np.linspace(0, 1),'b-.',linewidth=2,alpha=0.5)
# plt.yticks(size=15,rotation=90)
# plt.xticks(size=15)
# plt.ylabel('Fidelity',rotation=90,size=20,labelpad=20)
# plt.xlabel('Number of Control Step',size=20)

# x=[0,1,2,3,4,7,10,12,15,16,18,20,23,26,31,36,41,46,49,50]
# y=[0.0785,0.1067,0.1363,0.1708,0.2115,0.3404,0.4662,0.5194,0.5631,0.5775,0.5884,0.5830,0.5866,0.5818,0.5724,0.5841,0.5887,0.5950,0.5995,0.6010]
# plt.plot(x,y,'g-o',label='ΔE=0.0323')
# plt.plot(x,np.ones(len(x))*y[-1],'g--',linewidth=2)
# plt.plot(np.zeros(50)+50,np.linspace(0, 1),'g-.',linewidth=2,alpha=0.5)
# plt.legend(fontsize='xx-large')
# plt.plot(np.zeros(50),np.linspace(0, 1),'k-.',linewidth=2)
# plt.ylim([0,1.02])
# plt.scatter(SD2[:,0],SD1[:,1],label='Imaginary Time')
# plt.scatter(SD2[:,0],SD2[:,1],label='Imaginary Time Control')
# plt.fill_between(SD2[:,0], SD2[:,0]*0 , SD1[:,1]-SD2[:,1] , color='grey', alpha=0.5,label='Measurement Difference')
# plt.xlabel('ΔE',size=25)
# plt.ylabel('Number of Measurements',size=25)
# plt.xticks(size=25)
# plt.yticks(size=25)
# plt.legend(fontsize='xx-large')

# fig, ax = plt.subplots()
# x=[-0.046,-0.039,-0.032,-0.027,-0.022,-0.015,-0.012,-0.010,-0.0067,-0.0045,-0.0029,-0.0019]
# color=['k','rosybrown','red','darkorange','y','g','teal','deepskyblue','navy','blue','darkviolet','deeppink']
# for i in range(len(CEIPR)):
#     ax.plot(CEIPR[i], label='ΔE='+str(x[i]),linewidth=2,alpha=0.8,color=color[i])
#     ax.plot(IEIPR[i],linewidth=2,alpha=0.8,color=color[i],linestyle='--')
# f_legend=ax.legend(fontsize=20,loc='upper center',bbox_to_anchor=(0.5, 1.17),
#           fancybox=True, shadow=True, ncol=6)

# ax.add_artist(f_legend)
# ax.set_ylabel('EIPR',size=30)
# ax.set_xlabel('τ',size=30)
# ax.tick_params(axis='both', which='major', labelsize=25)
# import matplotlib.lines as mlines
# CEI = mlines.Line2D([], [], color='r', label='Quantum Imaginary Time Control')
# IEI =  mlines.Line2D([], [], color='r', linestyle='--', label='Quantum Imaginary Time')
# ax.legend(handles=[CEI,IEI],fontsize=20)

# x=[-0.046,-0.039,-0.032,-0.027,-0.022,-0.015,-0.012,-0.010,-0.0067,-0.0045,-0.0029,-0.0019]
# color=['k','rosybrown','red','darkorange','y','g','teal','deepskyblue','navy','blue','darkviolet','deeppink']
# for i in range(len(IEIPR)):
#     plt.plot(IEIPR[i], label='ΔE='+str(x[i]),linewidth=2,alpha=0.8,color=color[i])
    
# plt.legend(fontsize=15,loc='upper center',bbox_to_anchor=(0.5, 1.15),
#           fancybox=True, shadow=True, ncol=6)
# plt.xlabel('EIPR',size=20)
# plt.ylabel('τ',size=20)
# plt.xticks(size=20)
# plt.yticks(size=20)