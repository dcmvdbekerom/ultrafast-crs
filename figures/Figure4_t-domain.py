
#%% Imports:
import sys
sys.path.append('../')
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.constants import pi, c
c *= 100 #cm.s-1
from ultrafast_crs import (set_database, generate_axes, calc_spectrum, calc_spectrum_ref_py, 
                      next_fast_aligned_len, Gaussian,
                      calc_Gamma)


data_path = '../data/CH4_v2/'
save_output = True


#%% Import data
print('Loading database... ', end='')


db = dict(np.load(data_path + 'database.npz'))

branch = 'Q'

DJ = {'O':-2, 'P':-1, 'Q':0, 'R':1, 'S':2}[branch]
N_EvJ = db['EvJ_data'].shape[1]
idx = (db['J_clip_data'] >= N_EvJ*(DJ+2)) & (db['J_clip_data'] < N_EvJ*(DJ+3))

for key in db:
    if key != 'EvJ_data': db[key] = db[key][idx]

set_database(**db)


print('Done! [{:.1f}M lines loaded]'.format(len(db['nu_data'])*1e-6))


#%% Initialization parameters:

p = 1.0 #bar  
T =  296.0 #K   
# T =  800.0 #K
# T = 1500.0 #K

# MEG fitting parameters (placeholders) #####
a = 2                                                                         # species-specific constant set to 2 
alpha = 0.0445
beta = 1.52
delta = 1.0                                                                     # room-temperature value
n = 0.0     
params = np.array([a, alpha, beta, delta, n], dtype=np.float64)

N_w_0 = 20000
# N_w_0 = 60000
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

#%% Truncate:
    
N_lines = 250#len(nu_data)
sorted_sigma = sorted(db['sigma_gRmin_data'])[::-1]

if N_lines == len(db['nu_data']):
    idx = np.ones(len(db['nu_data']), dtype=bool)
else:
    thresh = sorted_sigma[N_lines]
    idx = db['sigma_gRmin_data'] > thresh


db_trunc = {'EvJ_data': db['EvJ_data']}

for key in db:
    if key == 'EvJ_data': continue
    db_trunc[key] = db[key][idx]
    
set_database(**db_trunc)   

#%% Plot 1:
print('Plot 1')
fig, ax = plt.subplots(3,2, sharex=True, figsize=(16,8))
ax[0,1].axhline(0,c='k',alpha=0.5)
I_dict = {}
t0 = perf_counter()


tau_arr = np.arange(20.0,200.0,0.50)*1e-12 #s
print('tdw:',tau_arr[-1]*dw)

N_G=4
Gamma_RPA, G_min, G_max = calc_Gamma(p, T, params) #TODO: provide pure python alternative?
dG = (G_max * 1.0001 - G_min) / (N_G - 1)
print('tdG:',tau_arr[-1]*dG)
# Direct:
styles = ['--', '-', '-']
colors = ['k', 'tab:blue', 'tab:red']
keys = [
        'ref-py-w', 
        'ufa-cpp-w', 'ufa-cpp-t']


ref_key = keys[0]

I0_list = []
I1_list = []

for key, sty, col in zip(keys, styles, colors):
    approach, imp, domain = key.split('-')
    
    if domain == 't':
        # We should never get here..
        # if approach == 'ref':
        #     tau_arr2, I_arr2, times = calc_spectrum_ref_py(w_min, dw, N_w, p, T, tau, E_probe, 
        #                                             params, implementation=imp, domain='t')
        
        if approach == 'ufa':
            tau_arr2, I_arr2, times = calc_spectrum(w_min, dw, N_w, p, T, tau_arr[-1], E_probe, 
                                                params, implementation=imp, domain='t',N_G=N_G, envelope_corr=1)

        I_arr = np.interp(tau_arr, tau_arr2[:N_w//2], I_arr2[:N_w//2])
    else: #domain=='w'
        # if approach == 'leg':
        #     spectrum_func = lambda tau: calc_spectrum_legacy(w_min, dw, N_w, p, T, tau, dt_FWHM, 
        #                                             params, implementation=imp)[1:]
        if approach =='ref':
            spectrum_func = lambda tau: calc_spectrum_ref_py(w_min, dw, N_w, p, T, tau, E_probe, 
                                                    params, implementation=imp)[1:]
        else: #approach == apx
            spectrum_func = lambda tau: calc_spectrum(w_min, dw, N_w, p, T, tau, E_probe, 
                                                params, implementation=imp)[1:]
        
        I_list = []
        times = {'total':0.0}
        for i, tau in enumerate(tau_arr):
            I_w, times2 = spectrum_func(tau)
            I_tot = np.sum(I_w)*dw
            I_list.append(I_tot)
            # print(i, tau*1e12)
            times['total'] += times2['total']
        I_arr = np.array(I_list)
    
    # I_arr /= I_arr[0]
    if key == 'ref-py-w': tlab = '{:.1f} s'.format(times['total']*1e-3)
    if key == 'ufa-cpp-w': tlab = '{:.1f} s'.format(times['total']*1e-3)
    if key == 'ufa-cpp-t': tlab = '{:.1f} ms'.format(times['total'])
    
    
    print('{:10s}: {:6.1f} ms'.format(key, times['total'])+(' [ref]' if key==ref_key else ''))
    I_dict[key] = I_arr

    # lw = 1.5 if approach == 'apx' else 3
    lw = 1.5
    ax[0,0].plot(tau_arr*1e12, I_arr, sty, c=col, lw=lw, label=key+' ['+tlab+']', 
                 zorder = (0 if key==ref_key else -10),
                 alpha = (0.75 if key==ref_key else 1.0),
                 )
    
    if key == ref_key:
        I_ref = I_arr
    else:
        err = (I_arr - I_ref) / np.abs(I_ref)
        ax[0,1].plot(tau_arr*1e12, 100*err, sty, c=col, label = key)


ax[0,0].text(20, 1.6e-24,'250 lines\n$T = {:.0f}K$\n$N_\\omega={:.0f}k$, $N_\\Gamma={:d}$'.format(T,N_w*1e-3,N_G), size=12)

ax[0,0].legend(loc=1)
ax[0,1].legend()
ax[0,0].set_ylabel('Intensity (a.u.)')
ax[0,1].set_ylabel('Error (%)')
ax[0,0].set_yscale('log')
ax[0,1].yaxis.tick_right()
ax[0,1].yaxis.set_label_position('right')

# plt.savefig('t_domain_1.png',dpi=150)
print('')
#%% Reload lines:
set_database(**db)   


#%% Plot 2
print('Plot 2')
# fig, ax = plt.subplots(2, sharex=True)
ax[1,1].axhline(0,c='k',alpha=0.5)
I_dict = {}

N_list = [
          (20000,2),
          (40000,4),
          (80000,6),
          ]

colors = ['tab:red','tab:purple', 'tab:blue']

for i in range(len(N_list)):
    
    N_w_0, N_G = N_list[i]
    N_w = next_fast_aligned_len(N_w_0)
    dw = (w_max - w_min) / N_w
    
    w_arr, t_arr = generate_axes(w_min, dw, N_w)
    v_arr = w_arr / (2*pi*c)
    dt = t_arr[1] - t_arr[0]
    
    E_probe = Gaussian(t_arr, dt_FWHM)*dt
    
    # Gamma_RPA, G_min, G_max = calc_Gamma(p, T, params) #TODO: provide pure python alternative?
    # dG = (G_max * 1.0001 - G_min) / (N_G - 1)
    
    # print('tdw:',tau_arr[-1]*dw)
    # print('tdG:',tau_arr[-1]*dG)

    if not i:
        spectrum_func = lambda tau: calc_spectrum(w_min, dw, N_w, p, T, tau, E_probe, params)[1:]
    
        I_list = []
        times = {'total':0.0}
        for tau in tau_arr:
            I_w, times2 = spectrum_func(tau)
            I_tot = np.sum(I_w)*dw
            I_list.append(I_tot)
            # print(i, tau*1e12)
            times['total'] += times2['total']
        I_ref = np.array(I_list)
        print('ref: {:6.1f} ms'.format(times['total']))
        ax[1,0].plot(tau_arr*1e12, I_ref, 'k--', lw=1.5, label='ufa-cpp-w [{:.1f} s]'.format(times['total']*1e-3), alpha=0.75)


    tau_arr2, I_arr2, times = calc_spectrum(w_min, dw, N_w, p, T, tau_arr[-1], E_probe, 
                                        params, implementation='simd', domain='t',N_G=N_G, envelope_corr=1)
    
    I_arr = np.interp(tau_arr, tau_arr2[:N_w//2], I_arr2[:N_w//2])

    print('{:6.1f} ms'.format(times['total']))
    # I_dict[key] = I_arr

    lw = 1.5 #if approach == 'apx' else 3
    ax[1,0].plot(tau_arr*1e12, I_arr, '-', c=colors[i], lw=1.5, label='$N_\\omega={:d}k$, $N_\\Gamma={:d}$ [{:.1f} ms]'.format(N_w//1000, N_G, times['total']),zorder=-10)
    err = (I_arr - I_ref) / np.abs(I_ref)
    ax[1,1].plot(tau_arr*1e12, 100*err, '-', c=colors[i], label = '$N_\\omega={:d}k$, $N_\\Gamma={:d}$'.format(N_w//1000, N_G))


ax[1,0].text(20, 1e-14,'2.5M lines\n$T = 296K$', size=12)
ax[1,0].legend(loc=1)
ax[1,1].legend(loc=1)
ax[1,0].set_ylabel('Intensity (a.u.)')
ax[1,1].set_ylabel('Error (%)')
ax[1,0].set_yscale('log')

ax[1,1].yaxis.tick_right()
ax[1,1].yaxis.set_label_position('right')

# plt.savefig('t_domain_2.png',dpi=150)
print('')
#%% Plot 3
print('Plot 3')
# fig, ax = plt.subplots(2, sharex=True)
ax[2,1].axhline(0,c='k',alpha=0.5)
I_dict = {}


T_list = [296.0, 800.0, 1500.0]

# colors = ['tab:blue','tab:orange', 'tab:red']



N_w_0, N_G = (40000,4)
N_w = next_fast_aligned_len(N_w_0)
dw = (w_max - w_min) / N_w

w_arr, t_arr = generate_axes(w_min, dw, N_w)
v_arr = w_arr / (2*pi*c)
dt = t_arr[1] - t_arr[0]

E_probe = Gaussian(t_arr, dt_FWHM)*dt


t_ref_list = []
for T, col in zip(T_list, colors[::-1]):
    
    delta = 0.47 if T > 300.0 else 1.0
    n = 2.67 if T > 300.0 else 0.0 
    params = np.array([a, alpha, beta, delta, n], dtype=np.float64)

    
    Gamma_RPA, G_min, G_max = calc_Gamma(p, T, params) #TODO: provide pure python alternative?
    dG = (G_max * 1.0001 - G_min) / (N_G - 1)
    
    print('tdw:',tau_arr[-1]*dw)
    print('tdG:',tau_arr[-1]*dG)


    spectrum_func = lambda tau: calc_spectrum(w_min, dw, N_w, p, T, tau, E_probe, 
                                        params, implementation='simd')[1:]

    I_list = []
    times = {'total':0.0}
    for tau in tau_arr:
        I_w, times2 = spectrum_func(tau)
        I_tot = np.sum(I_w)*dw
        I_list.append(I_tot)
        # print(i, tau*1e12)
        times['total'] += times2['total']
    I_ref = np.array(I_list)
    print('ref: {:6.1f} ms'.format(times['total']))
    t_ref_list.append(times['total']*1e-3)
    ax[2,0].plot(tau_arr*1e12, I_ref, 'k--', lw=1.5, alpha=0.75, label=('ufa-cpp-w \n[#.# s (avg.)]' if T<300.0 else None)) # A little sloppy to hardcode this number, but we don't have access yet to the other two benchmark times here
    

    tau_arr2, I_arr2, times = calc_spectrum(w_min, dw, N_w, p, T, tau_arr[-1], E_probe, 
                                        params, implementation='simd', domain='t',N_G=N_G, envelope_corr=1)
    
    I_arr = np.interp(tau_arr, tau_arr2[:N_w//2], I_arr2[:N_w//2])

    print('{:6.1f} ms'.format(times['total']))
    # I_dict[key] = I_arr

    lw = 1.5 #if approach == 'apx' else 3
    ax[2,0].plot(tau_arr*1e12, I_arr, '-', c=col, lw=1.5, label='T = {:.0f}K [{:.1f} ms]'.format(T, times['total']) , zorder=-10)
    err = (I_arr - I_ref) / np.abs(I_ref)
    ax[2,1].plot(tau_arr*1e12, 100*err, '-', c=col, label='T = {:.0f}K'.format(T))


ax[2,0].text(20, 1e-14,'2.5M lines\n$N_\\omega=40k$, $N_\\Gamma=4$', size=12)
leg = ax[2,0].legend(loc=1)

leg.get_texts()[0].set_text('ufa-cpp-w \n[{:.1f} s (avg.)]'.format(np.mean(t_ref_list)))

ax[2,1].legend(loc=1)
ax[2,0].set_ylabel('Intensity (a.u.)')
ax[2,1].set_ylabel('Error (%)')
ax[2,0].set_yscale('log')

ax[2,1].yaxis.tick_right()
ax[2,1].yaxis.set_label_position('right')

ax[2,0].set_xlabel('Pump-probe delay time $\\tau$ (ps)')
ax[2,1].set_xlabel('Pump-probe delay time $\\tau$ (ps)')

plt.tight_layout()


t = ax[0,0].text(0.02,0.95,'a)', fontsize=18, va='top', transform = ax[0,0].transAxes)
t.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.7))

t = ax[0,1].text(0.02,0.95,'b)', fontsize=18, va='top', transform = ax[0,1].transAxes)
t.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.7))

t = ax[1,0].text(0.02,0.95,'c)', fontsize=18, va='top', transform = ax[1,0].transAxes)
t.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.7))

t = ax[1,1].text(0.02,0.95,'d)', fontsize=18, va='top', transform = ax[1,1].transAxes)
t.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.7))

t = ax[2,0].text(0.02,0.95,'e)', fontsize=18, va='top', transform = ax[2,0].transAxes)
t.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.7))

t = ax[2,1].text(0.02,0.95,'f)', fontsize=18, va='top', transform = ax[2,1].transAxes)
t.set_bbox(dict(facecolor='w', edgecolor='none', alpha=0.7))


#%%

if save_output:
    plt.savefig('output/Figure4_t-domain.png',dpi=300)
    plt.savefig('output/Figure4_t-domain.pdf',dpi=300)
    
    import pickle
    with open('output/Figure4.fig', 'wb') as f:
        pickle.dump(fig, f)

print('')
