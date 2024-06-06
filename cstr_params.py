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

'''
x=0, u=0 -> dx/dt=0
'''
def getCA0s():
    return V/F * k0 * np.exp(-E/(R*Ts)) * (CAs**2) + CAs

def getQs():
    return sigma * Cp * V * (F/V * (Ts-T0) + delta_H/(sigma*Cp) * k0 * np.exp(-E/(R*Ts)) * (CAs**2))

CA0s = getCA0s()
Qs = getQs()
