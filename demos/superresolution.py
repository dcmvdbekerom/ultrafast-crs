# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:17:17 2025

@author: bekeromdcmvd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x_max = 4.0
dx = 0.01

x_arr = np.arange(-x_max, x_max, dx)


G = lambda x, x0, w: np.exp(-((x-x0)/w)**2)

w_ref = 1.0



I_ref = G(x_arr, 0.0, w_ref)
I_ref /= np.max(I_ref)

def fitfun(x_arr, x0, dx, R, w=0.6, N=10):
    
    T = 1 - R
    
    A_list = [R] + [T**2*R**(i-1) for i in range(1,N)]
    x_list = [x0 + i*dx for i in range(N)]
    I_list = [Ai*G(x_arr, xi, w) for Ai, xi in zip(A_list, x_list)]
    
    I = np.sum(I_list,0)
    
    return I / np.max(I)

R0 = 0.38
w_new0 = w_ref*0.5
x00 = -w_new0*0.8
dx0 =  w_new0

popt = [x00, dx0, R0]
popt, pcov = curve_fit(fitfun, x_arr, I_ref, p0=popt)
print(popt)
I_new = fitfun(x_arr, *popt)
plt.axhline(0, c='k', lw=1)


plt.plot(x_arr, I_ref)
plt.plot(x_arr, I_new, '--')