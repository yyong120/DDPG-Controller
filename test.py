from casadi import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import torch.utils.data as Data
# from gekko import GEKKO
from math import exp
import sys
import os
from time import sleep, time
from tqdm import tqdm
from scipy.optimize import minimize
from collections import OrderedDict
import multiprocessing
from functools import reduce


e = 2.71828182846

T0 = 300.0
V = 1.0
k0 = 8.46e6
Cp = 0.231
sigma = 1000.0
# Ts = 400.0
Ts = 430.0
Qs = 0.0
F = 5.0
E = 5e4
delta_H = -1.15e4
R = 8.314
CAs = 2.0
CA0s = 4.0

P = np.array([[716.83, 0.0], [0.0, 1.0]])
gamma = 9.53    # equation 45g in "economic..."
x_size = 2
u_size = 2
c_hl_size = 8 # hidden layer for critic
a_hl_size = 8 # hidden layer for actor
u1_bd = 3.5
# u2_bd = 5e5
u2_bd = 5.0   # u2 is multified by 1e-5
u2_scale = 1e5
x1_bd = 1.0
x2_bd = 26.0
QMat = np.array([[3.0, 0.0], [0.0, 1.0]])
RMat = np.array([[1.0/20.0, 0.0], [0.0, 1.0/30.0]])

NUM_OF_X1 = 2
NUM_OF_X2 = 3

DISCOUNT = 0.99  # discount for q-learning rewards
C_LR = 1e-2     # learning rate for critic
A_LR = 1e-2     # learning rate for actor
C_MAX_EPOCH = 50
A_MAX_EPOCH = 50
MAX_Q_STEP = 2
BATCH_SIZE = 4

exception_num_0 = 0 # lyapunov constraints may conflict with u's bound
exception_num_1 = 0 # even not considering lyapunov constraints, solution still not found


DIR = "D:/WS/PythonWS/graduation_project/V2"
c_param_file = DIR + '/critic.pkl'
a_param_file = DIR + '/actor.pkl'


'''
x=0, u=0 -> dx/dt=0
'''
def getCA0s():
    return V/F * k0 * np.exp(-E/(R*Ts)) * (CAs**2) + CAs

def getQs():
    return sigma * Cp * V * (F/V * (Ts-T0) + delta_H/(sigma*Cp) * k0 * np.exp(-E/(R*Ts)) * (CAs**2))

CA0s = getCA0s()
Qs = getQs()

r1 = 3.0
r2 = 1.0
r3 = 1.0 / 20.0
r4 = 1.0 / 3e11

def get_xk1(xk, uk):
    xk1 = xk
    for _ in range(int(0.01 / 1e-4)):
        xk1[0] += 1e-4 * (F/V * (uk[0] + CA0s - xk1[0] - CAs) - k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2)
        xk1[1] += 1e-4 * (F/V * (T0 - xk1[1] - Ts) + (-delta_H)/sigma/Cp * k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2 + (uk[1] + Qs)/sigma/Cp/V)

    return xk1


def lmpc(inix):
    T = 0.1  # Time horizon
    N = 10 # number of control intervals，p
    
    # Declare model variables
    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x = vertcat(x1, x2)
    
    u1 = MX.sym('u1')
    u2 = MX.sym('u2')
    u = vertcat(u1, u2)
    
    # Model equations
    xdot = vertcat((F/V)*((u1+CA0s) - (x1+CAs)) - k0*(np.exp(-E/(R*(x2+Ts))))*((x1+CAs)**2),
                   (F/V)*(T0-x2-Ts) -((delta_H)/(sigma*Cp))*k0*(np.exp(-E/(R*(x2+Ts))))*((x1+CAs)**2) + (u2+Qs)/(sigma*Cp*V))
    
    # Objective term
    L = r1*(x1**2)+r2*(x2**2)+r3*(u1**2)+r4*(u2**2)#x.T*Q*x+u.T*R*u
    
    # Formulate discrete time dynamics
    # CVODES from the SUNDIALS suite
    dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
    opts = {'tf':T/N}#dekta
    Fc = integrator('Fc', 'cvodes', dae, opts)
    
    # Evaluate at a test point
    #Fk = Fc(x0=[inix[0],inix[1]],p=[10,51000])
    #print(Fk['xf'])#积分器求取的xk+1#[0.494658, 30.2413]
    #print(get_xk1([inix[0],inix[1]],[10,51000]))#自己写的xk+1的求解#[ 0.49837911 30.09165188]  
    #print(Fk['qf'])#区间的目标函数值
    
    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []
    
    # Formulate the NLP
    #initial x value

    # Xk = MX([inix[0], inix[1]])
    inix = [float(item) for item in inix]
    print(type(inix))
    print(inix)
    Xk = MX(np.array(inix))
    
    for k in range(N):
        # New NLP variable for the control
        #Uk = MX.sym('U_' + str(k),2)
        
        U1 = MX.sym('U1'+ str(k))
        U2 = MX.sym('U2'+ str(k))
        U = vertcat(U1, U2)
        
        #添加u的约束
        w += [U1]
        lbw += [-3.5]
        ubw += [3.5]
        w0 += [0]
        
        w += [U2]
        lbw += [-5e+5]
        ubw += [5e+5]
        w0 += [0]
        

         #Add inequality constraint            
        if k == 0:
            g += [(2*(Xk[1]))*((F/V)*(T0-Xk[1]-Ts) -((delta_H)/(sigma*Cp))*k0*(np.exp(-E/(R*(Xk[1]+Ts))))*((Xk[0]+CAs)**2) + (U2+Qs)/(sigma*Cp*V))+gamma*(Xk[1])**2]
            lbg += [-inf]
            ubg += [0]            

        # Integrate till the end of the interval
        Fk = Fc(x0=Xk, p=U)
        Xk = Fk['xf']
        #J=J+Fk['qf']
        J=J+r1*(Xk[0]**2)+r2*(Xk[1]**2)+r3*(U1**2)+r4*(U2**2)
        # Add inequality constraint
        #g += [Xk[0]]
        #lbg += [-.25]
        #ubg += [inf]
    
    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    # solver = nlpsol('solver', 'ipopt', prob,{'ipopt':{'max_iter':100}})
    solver = nlpsol('solver', 'ipopt', prob,{'ipopt':{'max_iter':100, 'print_level': 0}, 'print_time': 0})
    
    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x']
    
    fin_u1=w_opt[::2]
    fin_u2=w_opt[1::2]
    
    
    return fin_u1,fin_u2
    
    #print("解是",w_opt)


x0=[1.0,25.0] #自己设置一个初始值
U1=[]
U2=[]
# X0=[]
X1=[]
X2=[]
n=10#求解n个

for i in tqdm(range(n)):

    # X0.append(x0)
    X1.append(x0[0])
    X2.append(x0[1])
    
    u1,u2=lmpc(x0)
    x0=get_xk1(x0,[u1[0],u2[0]])#求下时刻的系统状态，更新x0为xk+1
    
    U1.append(u1[0])
    U2.append(u2[0])