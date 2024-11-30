
#%% Imports:
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ultrafast_crs import next_fast_aligned_len, generate_axes, set_database, calc_spectrum, pi, c, Gaussian
import matplotlib.backends.backend_qt as qt
from matplotlib.backends.backend_qt import QtCore

#%% Import data
print('Loading database... ', end='')

data_path = '../data/CH4_v2/'
ref_path = '../ref/'

db = np.load(data_path + 'database.npz')
err = set_database(**db, set_cpp=True, force_alignment=False)   

print('Done! [{:.1f}M lines loaded]'.format(len(db['nu_data'])*1e-6))


#%% Initialization parameters:

p = 1.0 #bar
T = 296.0#K  
# T = 1500.0 #K   

# MEG fitting parameters (placeholders) #####
a = 2                                                                           # species-specific constant set to 2 
alpha = 0.0445
beta = 1.52

if T < 500.0:
    delta = 1.0                                                                     # room-temperature value
    n = 0.0     
else:
    delta = 0.47
    n = 2.67

params = np.array([a, alpha, beta, delta, n], dtype=np.float64)

N_w_0 = 40000
v_min = 1190.0 #cm-1
v_max = 2010.0 #cm-1
v_avg = 0.5*(v_min + v_max)

dt_pr = 5e-12; #FWHM of the probe
dt_FWHM = dt_pr/np.log(2)**0.5
tau_min = 25
tau_arr = (np.arange(tau_min, 200+1, 1))*1e-12;   
itau = 134
tau = tau_arr[itau]  

impl = 'simd'
N_G = 3


#%% Spectral axis:       
w_min = 2*pi*c*v_min
w_max = 2*pi*c*v_max   
N_w = next_fast_aligned_len(N_w_0)
dw = (w_max - w_min) / N_w

w_arr, t_arr = generate_axes(w_min, dw, N_w)
dt = t_arr[1] - t_arr[0]


E_probe = Gaussian(t_arr, dt_FWHM)*dt    


#%% Reference spectrum

# I_CRS_ref = np.load(data_path + 'I_ref_arr_{:.1f}K_all.npy'.format(T))
chi_ref = np.load(ref_path + 'chi_arr_{:.1f}K_all.npy'.format(T))
I_CRS_ref = np.zeros((len(tau_arr), len(chi_ref)), dtype=np.float64)
E_probe_func = lambda t: Gaussian(t, dt_FWHM)*dt

from scipy.fft import fft
for i, tau_i in enumerate(tau_arr):    
    E_pr_i = E_probe_func(t_arr - tau_i)
    E_CRS = fft(E_pr_i * chi_ref)
    I_CRS_ref[i] = np.abs(E_CRS)**2



skip = 40
err_map = np.zeros((len(tau_arr), len(w_arr[::skip])), dtype=np.float64)
for i, tau_i in enumerate(tau_arr):
    w_arr, I_CRS, times = calc_spectrum(w_min, dw, N_w, p, T, tau_i, E_probe, params, N_G=N_G, implementation=impl)
    Imax = np.max(I_CRS_ref[i])
    err_map[i] = ((I_CRS_ref[i,::skip] - I_CRS[::skip])/Imax * 100.0)

Imax = np.max(I_CRS_ref[itau])
print('Done!')


#%% Plotting:
    
# fig, ax = plt.subplots(2, sharex=True)

fig = plt.figure(figsize=(18,6))
ax0 = plt.subplot2grid(shape=(2, 2), loc=(0,0))
ax1 = plt.subplot2grid(shape=(2, 2), loc=(1,0), sharex=ax0)
ax2 = plt.subplot2grid(shape=(2, 2), loc=(0,1), sharex=ax0, rowspan=2)
ax = [ax0, ax1, ax2]
plt.subplots_adjust(left=0.05, right=0.95)




w_arr, I_CRS, times = calc_spectrum(w_min, dw, N_w, p, T, tau, E_probe, params, N_G=N_G, implementation=impl)


v_arr = w_arr / (2*pi*c)
p1, = ax[0].plot(v_arr,I_CRS, label='Ultrafast algorithm')    
p2, = ax[0].plot(v_arr,I_CRS_ref[itau,:], 'k--', label='Reference')

vb1 = ax[0].axvline(v_avg, c='gray', lw=1)

ax[0].set_ylim(-Imax*0.1,Imax*1.1)

ax[0].legend(loc=1)
ax[0].set_ylabel('Intensity (a.u.)')
# ax[0].set_xticklabels([])
plt.setp(ax[0].get_xticklabels(), visible=False) 


ax[0].axhline(0,c='k',lw=1,alpha=0.5)


rtol = 5e-4
ax[1].axhline( 100*rtol, c='k', ls='--', lw=1)
ax[1].axhline(-100*rtol, c='k', ls='--', lw=1)

err = (I_CRS_ref[itau,:] - I_CRS)/Imax  #<--

ax[1].axhline(0,c='k',lw=1,alpha=0.5)
p3,= ax[1].plot(v_arr, 100*err)

ax[1].set_xlabel('Raman shift (cm$^{-1}$)')
ax[1].set_ylabel('Error w.r.t. reference (% of max.)')

ax[1].set_xlim(v_min, v_max)
ax[1].set_ylim(-0.1, 0.1)


vb2 = ax[1].axvline(v_avg, c='gray', lw=1)



cmap = ax[2].imshow(err_map[::-1,:], 
            extent=(v_arr[0],v_arr[-1],tau_arr[0]*1e12,tau_arr[-1]*1e12), 
            aspect='auto', 
            vmin=-0.1,
            vmax= 0.1,
            cmap='seismic',#'RdBu',#'gist_ncar',
            interpolation='bicubic',
            )

ax[2].set_xlim(1400,1800)
ax[2].yaxis.tick_right()
ax[2].yaxis.set_label_position('right') 
ax[2].set_xlabel('Raman shift (cm$^{-1}$)')
ax[2].set_ylabel('Probe delay (ps)')


cbar_ax = fig.add_axes([0.93, 0.125, 0.015, 0.78])
cb= fig.colorbar(cmap, cax=cbar_ax)
cb.set_label('Error w.r.t. reference (% of max.)')
valbar0 = cbar_ax.axhline(0, c='w', lw=2, zorder=10)
valbar = cbar_ax.axhline(0, c='k', lw=1, zorder=10)

hbar = ax[2].axhline(tau*1e12, c='gray', lw=1)
hbarch, = ax[2].plot((0.5*(v_min+v_max),),(tau*1e12,), '|', c='gray', lw=1, ms=15)

plt.subplots_adjust(
    top=0.93,
    bottom=0.11,
    left=0.065,
    right=0.880,
    hspace=0.065,
    wspace=0.055)
    
def update(event):
    global last_x, err
    if not event.inaxes:
        last_x = None
        return
    
    if event.button == 2:
        
        if last_x is None:
            last_x = event.xdata
        else: 
            xstep = event.xdata - last_x
            xmin, xmax = event.inaxes.get_xlim()
            event.inaxes.set_xlim(xmin-xstep, xmax-xstep)
            fig.canvas.draw_idle()
            last_x = event.xdata - xstep
    
    elif event.button == 1:
    
        x = event.xdata    
        vb1.set_xdata((x,x))
        vb2.set_xdata((x,x))
        hbarch.set_xdata([x])
        
        if not ax[2].in_axes(event):
            iw = int((2*pi*c*x - w_min)/dw + 0.5)
            valbar0.set_ydata((100*err[iw], 100*err[iw]))
            valbar.set_ydata((100*err[iw], 100*err[iw]))
            fig.canvas.draw_idle()
            return
        
        y = event.ydata
        itau = int(y + 0.5) - tau_min
        tau = tau_arr[itau]
        w_arr, I_CRS, times = calc_spectrum(w_min, dw, N_w, p, T, tau, E_probe, params, N_G=N_G,
                                            implementation=impl, chunksize=16*1024)
    
        Imax = np.max(I_CRS_ref[itau])
        err = ((I_CRS_ref[itau] - I_CRS)/Imax) 
    
        p1.set_ydata(I_CRS)
        p2.set_ydata(I_CRS_ref[itau,:]) 
        p3.set_ydata(100*err)
        hbar.set_ydata((y,y))
        hbarch.set_ydata([y])

        iw = int((2*pi*c*x - w_min)/dw + 0.5)
        valbar0.set_ydata((100*err[iw], 100*err[iw]))
        valbar.set_ydata((100*err[iw], 100*err[iw]))

        t_tot = times['total']
        ax[0].set_title('τ = {:4.0f} ps,  t = {:5.1f} ms'.format(tau*1e12, t_tot))
        ax[0].set_ylim(-Imax*0.1,Imax*1.1) # <--
    
        fig.canvas.draw_idle()


def stop_drag(event):
    global last_x
    last_x = None
        
    
def zoom(event):
    if event.inaxes is None:
        return
    
    zoom = 1.2
    xmin, xmax = event.inaxes.get_xlim()

    if event.button == 'down':
        xmid = event.xdata
        new_xmin = max(xmid + (xmin-xmid) * zoom, v_min)
        new_xmax = min(xmid + (xmax-xmid) * zoom, v_max)
    if event.button == 'up':
        xmid = event.xdata
        new_xmin = xmid + (xmin-xmid) / zoom
        new_xmax = xmid + (xmax-xmid) / zoom
    event.inaxes.set_xlim(new_xmin, new_xmax)
    fig.canvas.draw_idle()



last_x = None
fig.canvas.mpl_connect('button_release_event', stop_drag)
fig.canvas.mpl_connect('button_press_event', update)
fig.canvas.mpl_connect('motion_notify_event', update)
fig.canvas.mpl_connect('scroll_event', zoom)

# s1.on_changed(update)
# s2.on_changed(update)
plt.show()
