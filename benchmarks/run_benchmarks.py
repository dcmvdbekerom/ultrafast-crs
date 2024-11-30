
#%% Imports:
import sys
sys.path.append('../')
import numpy as np
from time import perf_counter
from scipy.constants import pi, c
c *= 100 #cm.s-1
from ultrafast_crs import (set_database, generate_axes, next_fast_aligned_len, Gaussian, 
                      calc_spectrum,
                      calc_spectrum_py,
                      calc_spectrum_ref,
                      calc_spectrum_ref_py)
import pandas as pd
from time import sleep
import os


def get_extended_database(db, file_path_ext, N_ext=10):

    if os.path.exists(file_path_ext):
        return np.load(file_path_ext)
    
    db_ext = {}
    
    for dbkey in db:
        print(dbkey+'... ', end='')
        data = db[dbkey]
        
        if dbkey == 'EvJ_data':
            db_ext[dbkey] = data
            continue
        
        data_ext = np.zeros((N_ext, *data.shape), dtype=data.dtype, order='F')
        for i in range(N_ext):
            data_ext[i] = data
        data2 = data_ext.reshape(*data.shape[:-1], (N_ext * data.shape[-1]), order='F')
        db_ext[dbkey] = data2
        print('Done!')
        
    np.savez(file_path_ext, **db_ext)
    print('Done!')

    return db_ext

data_path = '../data/CH4_v2/'

# key = 'ref-py'
# key = 'ref-cpp'
key = 'ufa-py'
# key = 'ufa-cpp'

save_output       = True



#%% Import data
N_max_dict = {'ref-py':     21544,
              'ref-cpp':    46415,
              'ufa-py': 100000000,
              'ufa-cpp':100000000,
              }

print('Loading database... ')
db = np.load(data_path + 'database.npz')

extended_database = (N_max_dict[key] > len(db['nu_data']))
if extended_database:
    print('Using extended database...')
    db = get_extended_database(db, data_path + 'database_ext.npz')
else:
    print('Using regular database...')
print('Done! [{:.1f}M lines loaded]'.format(len(db['nu_data'])*1e-6))


#%% Initialization parameters:

p = 1.0 #bar  
T = 296.0 #K   
# T = 800.0 #K

# MEG fitting parameters (placeholders) #####
a = 2                                                                         # species-specific constant set to 2 
alpha = 0.0445
beta = 1.52
delta = 1.0                                                                     # room-temperature value
n = 0.0     
params = np.array([a, alpha, beta, delta, n], dtype=np.float64)

N_w_0 = 40000
N_G = 4
v_min = 1100.0 #cm-1
v_max = 2000.0 #cm-1

dt_pr = 5e-12; #width of the probe (not FWHM!)
dt_FWHM = dt_pr * np.log(2)**-0.5
tau_arr = (np.arange(25, 200+1, 1))*1e-12;   
itau = 134
tau = tau_arr[itau]  


#%% Spectral axis:       
w_min = 2*pi*c*v_min
w_max = 2*pi*c*v_max   
N_w = next_fast_aligned_len(N_w_0)
dw = (w_max - w_min) / N_w

w_arr, t_arr = generate_axes(w_min, dw, N_w)
v_arr = w_arr / (2*pi*c)

dt = t_arr[1] - t_arr[0]

E_probe = Gaussian(t_arr, dt_FWHM)*dt


#%% Run tests:    

func_dict = {
    'ref-py': calc_spectrum_ref_py,
    'ref-cpp':calc_spectrum_ref,
    'ufa-py': calc_spectrum_py,
    'ufa-cpp':calc_spectrum,
    }


folder = './'
# os.makedirs(folder, exist_ok=True)

approach, imp = key.split('-') 

I_list = [w_arr]
  
tau_list = [20e-12, 50e-12, 100e-12]    
T_list = [296.0, 800.0, 1500.0] #TODO: rerun with T=296.0K


# N_list = np.logspace(3,7,13,dtype=int)
N_list = [
    10,
    22,
    46,
    100,
    215,
    464,
    1000,
    2154,
    4641,
    10000,
    21544, #<- ref-py up to here
    46415, #<- ref-cy up to here
    100000,
    215443,
    464158,
    1000000,
    2154434,
    4641588,
    10000000,
    21544346,  
    46415888, 
    100000000,
    ]



df = pd.DataFrame(columns=['N_lines','N_simd','T (K)','tau (ps)','t_run (ms)'])
df2 = pd.DataFrame(columns=['N_lines','t_run (ms)', 'std_dev (%)'])
print('sorting sigma_gRmin... ',end='')
sorted_sigma = sorted(db['sigma_gRmin_data'])[::-1]
print('Done!')

calc_func = func_dict[key]

db_trunc = {'EvJ_data':db['EvJ_data']}
for N_lines in N_list:
    if N_lines > N_max_dict[key]:
        break

    thresh = sorted_sigma[N_lines]
    idx = db['sigma_gRmin_data'] > thresh

    for dbkey in db:
        if dbkey == 'EvJ_data': continue
        db_trunc[dbkey] = db[dbkey][idx][:N_lines]

    set_database(**db_trunc)   

    sleep(1.0)

    w_arr, I_arr, times = calc_func(w_min, dw, N_w, p, T_list[1], tau_list[1], 
                                        E_probe, params, N_G=N_G)

    for T in T_list:
        for tau in tau_list:
            t0 = perf_counter()
            

            w_arr, I_arr, times = calc_func(w_min, dw, N_w, p, T, tau, 
                                                        E_probe, params, N_G=N_G)
            
            I_list.append(I_arr)
            t_run = perf_counter() - t0
            df.loc[len(df.index)] = [N_lines, N_lines, T, tau*1e12, t_run*1e3]

    current = df['N_lines']==N_lines
    mu   = df['t_run (ms)'][current].mean()
    std =  df['t_run (ms)'][current].std() / mu
    df2.loc[len(df.index)] = [N_lines, mu, std]
    print(f'{N_lines:10d} - mean runtime:{mu:10.1f} ms ({100*std:4.1f}% stdev)')
    
print(df)
if save_output:
    df.to_excel(folder + 'benchmark_' + key + '.xlsx', index=False)
    df2.to_excel(folder + 'summary_' + key + '.xlsx', index=False)
    np.save(folder + 'output_' + key + '.npy', I_list)
    
