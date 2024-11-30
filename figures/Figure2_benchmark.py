# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:35:07 2024

@author: bekeromdcmvd
"""

import sys
sys.path.append('../')
sys.path.append('../cython/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


folder = '../benchmarks/'

fnames = sorted([f for f in os.listdir(folder) if f.split('_')[0] == 'summary'])[::-1]

fit_limits = {
'ref-py':  1e1,
'ref-cpp':  1e1,
'ufa-py':  1e5,
'ufa-cpp':6.8e6,#3.5e6,
}

colors = {'ref-py':'tab:blue',
          'ref-cpp':'tab:orange',
          'ufa-py':'tab:green',
          'ufa-cpp':'tab:red',
          }
ax, fig = plt.subplots(1, figsize=(6,7))

N_max = 1.5e8
for key in fit_limits.keys():
    fname = 'summary_' + key + '.xlsx'
    approach, implementation = key.split('-')
    df = pd.read_excel(folder + fname) 
    
    x = df['N_lines']
    y = df['t_run (ms)']
    
    idx = x >= fit_limits[key] 
    x_fit = np.log10(x[idx])
    y_fit = np.log10(y[idx])
    p = np.polyfit(x_fit,y_fit,1)
    
    # N_max = 1.5e8 if key in ['apx-cpp', 'apx-simd'] else 1.5e7
    x_extp = np.array([fit_limits[key], N_max])
    y_extp = 10**np.polyval(p, np.log10(x_extp))
    style = '*' if key=='ufa-cpp' else {'ref':'s', 'ufa':'o'}[approach]
    ms = 11 if key=='ufa-cpp' else 7
    p, = plt.plot(x, y, style, c=colors[key], label=key, ms=ms, mfc='None')
    plt.plot(x_extp, y_extp, '--', c=p.get_color(), lw=1)
    
plt.legend(loc=2, title='Implementation:', framealpha=1.0)
plt.subplots_adjust(bottom=0.080,top=0.96,right=0.95)
plt.grid(alpha=0.3)

c = 'k'
size=8
plt.text(7.3e7, 1e3*1.2, '1 sec', c=c, alpha=0.5, va='bottom', size=size)
plt.text(7.3e7, 60e3*1.2, '1 min', c=c, alpha=0.5, va='bottom', size=size)
plt.text(7.3e7, 3600e3*1.2, '1 hour', c=c, alpha=0.5, va='bottom', size=size)
plt.text(1e8, 24*3600e3*1.1, '1 day', c=c, alpha=0.5, va='bottom', size=size)

plt.annotate('', xy=(1.5e7,8e0), xytext=(1.5e8,8e0), 
             arrowprops = dict(arrowstyle='|-|',
                               mutation_scale=4.0,
                               alpha=0.5,
                               ),
             )
plt.text(4.75e7, 6e0,'Synthetically\nextended\ndatabase',size=8,ha='center',va='top', alpha=0.5)
plt.axhline(1e3, c=c, alpha=0.5)
plt.axhline(60e3, c=c, alpha=0.5)
plt.axhline(3600e3, c=c, alpha=0.5)
plt.axhline(24*3600e3, c=c, alpha=0.5)

plt.ylim(1.0,3e8)

plt.xlabel('Number of spectral lines')
plt.ylabel('Computation time (ms)')

plt.xscale('log')
plt.yscale('log')
plt.gca().set_aspect('equal')
plt.savefig('output/Figure2_benchmark.png', dpi=300)
plt.savefig('output/Figure2_benchmark.pdf', dpi=300)
import pickle
with open('output/Figure2.fig', 'wb') as f:
    pickle.dump(fig, f)
