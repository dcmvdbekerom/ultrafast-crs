
#%% Imports:
import sys
sys.path.append('../')
sys.path.append('../cython/')
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
c *= 100 #cm.s-1
from ultrafast_crs import set_database, generate_axes, calc_spectrum, next_fast_aligned_len, Gaussian
from time import perf_counter

#%% Import data

labelsize=12
#TODO: make use of pre-processing
print('Loading database... ', end='')
data_path = '../data/CH4_v2/'
ref_path = '../ref/'


db = np.load(data_path + 'database.npz')
set_database(**db)    


print('Done! [{:.1f}M lines loaded]'.format(len(db['nu_data'])*1e-6))


#%% Initialization parameters:

p = 1.0 #bar  
T = 296.0 #K
# T = 1500.0 #K   

# MEG fitting parameters (placeholders) #####
a = 2                                                                         # species-specific constant set to 2 
alpha = 0.0445
beta = 1.52
delta = 1.0                                                                     # room-temperature value
n = 0.0    

# delta = 0.47
# n = 2.67
params = np.array([a, alpha, beta, delta, n], dtype=np.float64)

N_w_0 = 40000
v_min = 1190.0 #cm-1
v_max = 2010.0 #cm-1

dt_pr = 5e-12; 
dt_FWHM = dt_pr/np.log(2)**0.5 #FWHM of the probe
tau_arr = (np.arange(25, 200+1, 1))*1e-12;   
itau = 142
tau = tau_arr[itau]  


#%% Spectral axis:       
w_min = 2*pi*c*v_min
w_max = 2*pi*c*v_max   
N_w = next_fast_aligned_len(N_w_0)
dw = (w_max - w_min) / N_w

w_arr, t_arr = generate_axes(w_min, dw, N_w)
v_arr = w_arr/(2*pi*c)
dt = t_arr[1] - t_arr[0]
E_probe = np.exp(-(2*np.log(2)*(t_arr)/dt_pr)**2)*dt      
# E_probe *= (E_probe/np.max(E_probe) >= 1e-6)

#%% Plotting:
N_G = 4



# T = 296K

chi_ref_296K = np.load(ref_path + 'chi_arr_296.0K_all.npy')
I_ref_296K = np.zeros((len(tau_arr), len(chi_ref_296K)), dtype=np.float64)
E_probe_func = lambda t: Gaussian(t, dt_FWHM)*dt

from scipy.fft import fft
for i, tau_i in enumerate(tau_arr):    
    E_pr_i = E_probe_func(t_arr - tau_i)
    E_CRS = fft(E_pr_i * chi_ref_296K)
    I_ref_296K[i] = np.abs(E_CRS)**2

err_map_296K = np.zeros((len(tau_arr), len(w_arr)), dtype=np.float64)
delta = 1.0                                                                     # room-temperature value
n = 0.0    
params = np.array([a, alpha, beta, delta, n], dtype=np.float64)
for i, tau_i in enumerate(tau_arr):
    w_arr, I_CRS, times = calc_spectrum(w_min, dw, N_w, p, 296.0, tau_i, E_probe, params, N_G=N_G)
    Imax = np.max(I_ref_296K[i])
    err_map_296K[i] = (I_ref_296K[i] - I_CRS)/Imax * 100.0



# T = 1500K
chi_ref_1500K = np.load(ref_path + 'chi_arr_1500.0K_all.npy')
I_ref_1500K = np.zeros((len(tau_arr), len(chi_ref_1500K)), dtype=np.float64)
E_probe_func = lambda t: Gaussian(t, dt_FWHM)*dt

from scipy.fft import fft
for i, tau_i in enumerate(tau_arr):    
    E_pr_i = E_probe_func(t_arr - tau_i)
    E_CRS = fft(E_pr_i * chi_ref_1500K)
    I_ref_1500K[i] = np.abs(E_CRS)**2


err_map_1500K = np.zeros((len(tau_arr), len(w_arr)), dtype=np.float64)
delta = 0.47
n = 2.67
params = np.array([a, alpha, beta, delta, n], dtype=np.float64)
for i, tau_i in enumerate(tau_arr):
    w_arr, I_CRS, times = calc_spectrum(w_min, dw, N_w, p, 1500.0, tau_i, E_probe, params, N_G=N_G)
    Imax = np.max(I_ref_1500K[i])
    err_map_1500K[i] = (I_ref_1500K[i] - I_CRS)/Imax * 100.0

    
    
# I_ref_296K    = np.load('I_arr_296K.npy')
# I_ref_1500K   = np.load('I_arr_1500K.npy')
# err_map_296K  = np.load('err_map_296K.npy')
# err_map_1500K = np.load('err_map_1500K.npy')
colors = ['tab:blue', 'tab:purple', 'tab:red']

#t = 40, 110, 180
#i = 15,  85, 155

fig, ax = plt.subplots(3,2, sharex=True, figsize=(13.9,12))
ax00 = ax[1,0]
ax10 = ax[1,1]
ax01 = ax[0,0]
ax11 = ax[0,1]
ax20 = ax[2,0]
ax21 = ax[2,1]


idx = (v_arr>1400-5)&(v_arr<1800+5)
i_list = [25,75,125]
dist = [0, 7.5e-25, 12.5e-25]


delta = 1.0                                                                     # room-temperature value
n = 0.0    
params = np.array([a, alpha, beta, delta, n], dtype=np.float64)

for i,i0 in enumerate(i_list):
    w_arr, I, times = calc_spectrum(w_min, dw, N_w, p, T, tau_arr[i0], E_probe, params)
    v_arr = w_arr/(2*np.pi*c)
    label0 = '$\\tau = {:.0f}$ ps'.format(tau_arr[i0]*1e12)
    label1 = label0 + ' [{:.1f} ms]'.format(times['total'])
    ax00.axhline(dist[i], c='k', alpha=0.2, lw=1.0)
    ax00.plot(v_arr, I_ref_296K[i0]+ dist[i],'k--', alpha=0.75, label=('ref [~4h]' if i==0 else None))
    ax00.plot(v_arr, I + dist[i], c=colors[i],zorder=-i, label=label1)
    ax01.axhline(tau_arr[i0]*1e12, c=colors[i], alpha=0.5, ls='--')
    
    ax20.plot(v_arr, err_map_296K[i0], c=colors[i], label=label0)
    
ax20.set_ylim(-0.06,0.06)
ax20.legend(loc=1)

delta = 0.47
n = 2.67
params = np.array([a, alpha, beta, delta, n], dtype=np.float64)
# dist = [0, 5e-23, 12e-23]
dist = [0, 5e-23, 12.5e-23]

T= 1500.0#K

for i,i0 in enumerate(i_list):
    
    w_arr, I, times = calc_spectrum(w_min, dw, N_w, p, T, tau_arr[i0], E_probe, params)
    v_arr = w_arr/(2*np.pi*c)
    label0 = '$\\tau = {:.0f}$ ps'.format(tau_arr[i0]*1e12)
    label1 = label0 + ' [{:.1f} ms]'.format(times['total'])
    ax10.axhline(dist[i], c='k', alpha=0.2, lw=1.0)
    ax10.plot(v_arr, I_ref_1500K[i0]+ dist[i],'k--', alpha=0.75, label=('ref [~4h]' if i==0 else None))
    ax10.plot(v_arr, I + dist[i], c=colors[i],zorder=-i, label=label1)
    ax11.axhline(tau_arr[i0]*1e12, c=colors[i], alpha=0.5, ls='--')

    ax21.plot(v_arr, err_map_1500K[i0], c=colors[i], label=label0)

ax21.set_ylim(-0.06,0.06)
ax21.legend(loc=1)

cmap1 = ax01.imshow(err_map_296K[:,idx], 
           extent=(v_arr[idx][0],v_arr[idx][-1],tau_arr[-1]*1e12,tau_arr[0]*1e12), 
           aspect='auto', 
            vmin=-0.08,
            vmax= 0.08,
           cmap='seismic',#'RdBu',#'gist_ncar',
           interpolation='bicubic',
           )

cmap2 = ax11.imshow(err_map_1500K[:,idx], 
            extent=(v_arr[idx][0],v_arr[idx][-1],tau_arr[-1]*1e12,tau_arr[0]*1e12), 
            aspect='auto', 
            vmin=-0.08,
            vmax= 0.08,
            cmap='seismic',#'RdBu',#'gist_ncar',
            interpolation='bicubic',
            )


# ax[0,0].legend(bbox_to_anchor=(0.47,0.35))

ax00.legend(loc=1)
# t1 = ax00.text(1410,1.9e-24,'T = 296K',size=14)
# t1 = ax00.text(1405,1.925e-24,'c)',size=14)
# t1.set_bbox({'fc':'w', 'alpha':0.5, 'ec':'none'})
ax00.set_ylabel('Intensity (a.u.)', size=labelsize)
ax00.set_ylim(-0.1e-24,2.3e-24)
ax00.yaxis.tick_left()
ax00.yaxis.set_label_position('left')

ax10.legend(loc=1)
# t2 = ax10.text(1410,1.9e-22,'T = 1500K',size=14)
# t2 = ax10.text(1405,1.925e-22,'d)',size=14)
# t2.set_bbox({'fc':'w', 'alpha':0.5, 'ec':'none'})
ax10.set_ylabel('Intensity (a.u.)', size=labelsize)
ax10.set_ylim(-0.1e-22,2.3e-22)
ax10.yaxis.tick_right()
ax10.yaxis.set_label_position('right')

# ax01.text(1410,180,'T = 296K',size=14)
# ax01.text(1405,185,'a)',size=14)
ax01.set_ylim(tau_arr[0]*1e12, tau_arr[-1]*1e12)
ax01.set_title('T = 296K', fontsize=14)
ax01.set_ylabel('Probe delay time (ps)', size=labelsize)
ax01.yaxis.tick_left()
ax01.yaxis.set_label_position('left')

# ax11.text(1410,180,'T = 1500K',size=14)
# ax11.text(1405,185,'b)',size=14)
ax11.set_ylim(tau_arr[0]*1e12, tau_arr[-1]*1e12)
ax11.set_title('T = 1500K', fontsize=14)
ax11.set_ylabel('Probe delay time (ps)', size=labelsize)
ax11.yaxis.tick_right()
ax11.yaxis.set_label_position('right')


ax20.set_ylabel('Error w.r.t. reference (% of max.)', size=labelsize)
ax20.set_xlabel('Wavenumber (cm$^{-1}$)', size=labelsize)


ax21.yaxis.tick_right()
ax21.yaxis.set_label_position('right')
ax21.set_ylabel('Error w.r.t. reference (% of max.)', size=labelsize)
ax21.set_xlabel('Wavenumber (cm$^{-1}$)', size=labelsize)



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



fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.92, 0.7, 0.012, 0.26])
cb= fig.colorbar(cmap2, cax=cbar_ax)

# cb = plt.colorbar(mappable=cmap2, orientation='horizontal')
cb.set_label('Error relative to max intensity (%)', size=labelsize)
# plt.tight_layout()

plt.subplots_adjust(
top=0.965,
bottom=0.09,
left=0.065,
right=0.860,
hspace=0.085,
wspace=0.065)

plt.savefig('output/Figure3_err_map.png',dpi=300)
plt.savefig('output/Figure3_err_map.pdf',dpi=300)

#%%  Attempt to enable on bitmap interpolation
try:
    
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, BooleanObject

    reader = PdfReader("output/Figure3_err_map.pdf")
    writer = PdfWriter(reader)

    for img_obj in writer.pages[0].images:
        obj = writer.get_object(img_obj.indirect_reference)
        obj[NameObject('/Interpolate')] = BooleanObject(True)

    writer.write('output/Figure3_err_map_intp.pdf')

except(ModuleNotFoundError):
    pass

#%%
import pickle
with open('output/Figure3.fig', 'wb') as f:
    pickle.dump(fig, f)

print(' Done!')


