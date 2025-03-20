
#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from numpy.random import random
from scipy.optimize import curve_fit
from functools import partial
from time import perf_counter
import sys
sys.path.append('../')
from ultrafast_crs import set_database, generate_axes, calc_spectrum, next_fast_aligned_len, pi, c


#%% User input

p = 1.0 #bar  
T = 293.0 #K   #initial value
scale = 1.0

# MEG fitting parameters
a = 2                                                                    
alpha = 0.0445
beta = 1.52
delta = 1.0                                                            
n = 0.0     

# Fitting limits
T_min, T_max = 293.0, 1500.0
scale_min, scale_max = 0.0, 2.0
tau_min, tau_max = 5.0, 100.0 #ps

# Spectral axis
N_w_0 = 40000
v_min = 1200.0 #cm-1
v_max = 2000.0 #cm-1

tau = 16.0 #s #initial value
dt_pr = 4.7e-12; #FWHM of the probe

data_path = '../data/CH4_v2/'


#%% Data initialization       
print('Initializing axes... ', end='')
it=0

cum_time = 0.0
cur_time = 0.0

w_min = 2*pi*c*v_min
w_max = 2*pi*c*v_max   
N_w = next_fast_aligned_len(N_w_0)
dw = (w_max - w_min) / N_w

w_arr, t_arr = generate_axes(w_min, dw, N_w)
v_arr = w_arr / (2*pi*c)
v_exp = (np.arange(v_min, v_max, 2.0))
w_exp = 2*pi*c*v_exp

params = np.array([a, alpha, beta, delta, n], dtype=np.float64)

t_max = np.max(t_arr)
dt = t_arr[1] - t_arr[0]
E_probe = np.exp(-(2*np.log(2)*(t_arr)/dt_pr)**2)*dt      
print('Done!')

print('Loading database... ', end='')

db = dict(np.load(data_path + 'database.npz')) # Convert to dict to support item assignment
set_database(**db)    

Nl = round(len(db['nu_data'])*1e-5)*0.1

print('Done! [{:.1f}M lines loaded]'.format(Nl))


#%% Callback functions:

def fitfun(w_in, T, scale, dt_pr, norm_before_intp=True):
    global cum_time
    t0 = perf_counter()
    E_probe = np.exp(-(2*np.log(2)*(t_arr)/dt_pr)**2)*dt   
    w_arr, I_CRS, times = calc_spectrum(w_min, dw, N_w, p, T, s3.val*1e-12, E_probe, params, N_G=2)
    cum_time += perf_counter() - t0
    if norm_before_intp: I_CRS /= np.max(I_CRS)
    I_arr = np.interp(w_in, w_arr, I_CRS)
    if not norm_before_intp: I_arr /= np.max(I_arr)
    return scale*I_arr


def update(obj, val):
    global last_changed
    last_changed = obj
    dt_pr = s4.val        
       
    I_CRS = fitfun(w_arr, s1.val, s2.val, s4.val)
    p_fit.set_ydata(I_CRS)
    fig.canvas.draw_idle()



def fit(event):
    global cum_time, cur_time
    popt = [s1.val, s2.val]
    b2.hovercolor= 'coral'
    b2._motion(event)
    # I_exp = p_exp.get_ydata()
    cum_time = 0.0
    # popt, pcov = curve_fit(fitfun, w_exp, I_exp, p0=popt, 
                           # bounds=([T_min, scale_min],[T_max, scale_max]))
    
    print('Fit values: {:6.1f}, {:5.3f}'.format(*popt))
    s1.set_val(popt[0])
    s2.set_val(popt[1])
    b2.hovercolor= 'whitesmoke'
    b2._motion(event)

    fig.canvas.draw()


def release_callback(event):
    global last_changed
    if last_changed == s3:
        last_changed = None

    
#%% Plot initialization
fig, ax = plt.subplots(1, sharex=True, figsize=(8,7))
plt.subplots_adjust(top=0.77,bottom=0.21)

s1_ax = plt.axes([0.15, 0.1,  0.55, 0.03])
s1  = Slider(s1_ax, '$T$ $(K)$', T_min, T_max, valinit=T)
s1.on_changed(partial(update,s1))

s2_ax = plt.axes([0.15, 0.05,  0.55, 0.03])
s2  = Slider(s2_ax, '$scale$', scale_min, scale_max, valinit=scale)
s2.on_changed(partial(update,s2))

s3_ax = plt.axes([0.25, 0.85,  0.50, 0.03])
s3  = Slider(s3_ax, '$\\tau$ $(ps)$', tau_min, tau_max, valinit=tau)
s3.on_changed(partial(update,s3))
last_changed = None
cid0 = fig.canvas.mpl_connect('button_release_event', release_callback)

s4_ax = plt.axes([0.25, 0.80,  0.50, 0.03])
s4  = Slider(s4_ax, '$\\Delta t_{pr}$', 0.1e-12, 20.0e-12, valinit=dt_pr)
s4.on_changed(partial(update,s4))

b2_ax = fig.add_axes([0.8, 0.05, 0.1, 0.03])
b2 = Button(b2_ax, 'Fit')
b2.on_clicked(fit)

ax.set_title('Methane $\\nu_2$ ({:.1f} M lines)\nFit temperature at known probe delay\n\n\n\n'.format(Nl), fontsize=14)
ax.set_xlabel('Probe delay time (ps)')
ax.set_ylabel('Normalized CRS signal')

ax.grid(True)

ax.axhline(0,c='k', lw=2)
p_fit, = ax.plot(v_arr, np.ones_like(v_arr), label='Fit')    
# p_exp, = ax.plot(v_exp, np.ones_like(v_exp), 'o', mec='k', label='Data',zorder=-10, ms=5)
ax.legend(loc=1)     
ax.set_ylim(-0.1,1.1)   

#update plots once:
update(None,0.0)

plt.show()
