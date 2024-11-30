
#%% Import modules
import matplotlib
#matplotlib.use('QtAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from numpy.random import random
from scipy.optimize import curve_fit
from functools import partial

import sys
sys.path.append('../')
from ultrafast_crs import set_database, generate_axes, calc_spectrum, next_fast_aligned_len, pi, c


#%% User input

p = 1.0 #bar  
T = 500.0 #K   
branch = 'Q'

# MEG fitting parameters
a = 2                                                                    
alpha = 0.0445 #initial value
beta = 1.52  #initial value
delta = 1.0                                                            
n = 0.0     

# Fitting limits
alpha_min, alpha_max = 0.002, 0.1
beta_min, beta_max = 0.0, 5.0
T_min, T_max = 296.0, 1500.0

# Spectral axis
N_w_0 = 40000
v_min = 1200.0 #cm-1
v_max = 2000.0 #cm-1

dt_pr = 5e-12; #FWHM of the probe

data_path = '../data/CH4_v2/'


#%% Data initialization       
print('Initializing axes... ', end='')

w_min = 2*pi*c*v_min
w_max = 2*pi*c*v_max   
N_w = next_fast_aligned_len(N_w_0)
dw = (w_max - w_min) / N_w

w_arr, t_arr = generate_axes(w_min, dw, N_w)
tau_idx = (t_arr>25e-12)&(t_arr<250e-12)
tau_exp = (np.arange(25, 200+1, 2))*1e-12

params = np.array([a, alpha, beta, delta, n], dtype=np.float64)

t_max = np.max(t_arr)
dt = t_arr[1] - t_arr[0]
E_probe = np.exp(-(2*np.log(2)*(t_arr)/dt_pr)**2)*dt      
print('Done!')
print('Loading database... ', end='')


db = dict(np.load(data_path + 'database.npz')) # Convert to dict to support item assignment

N_EvJ = db['EvJ_data'].shape[1]
DJ = {'O':-2, 'P':-1, 'Q':0, 'R':1, 'S':2}[branch]
idx = (db['J_clip_data'] >= N_EvJ*(DJ+2)) & (db['J_clip_data'] < N_EvJ*(DJ+3))
for key in db:
    if key != 'EvJ_data': db[key] = db[key][idx]
set_database(**db)    

Nl = round(len(db['nu_data'])*1e-5)*0.1

print('Done! [{:.1f}M lines loaded]'.format(Nl))


#%% Callback functions:

def fitfun(tau_in, alpha, beta):
    params = np.array([a, alpha, beta, delta, n], dtype=np.float64)
    tau_arr, I_PDS, times = calc_spectrum(w_min, dw, N_w, p, s3.val, t_max, E_probe, params, domain='t', N_G=4)
    I_arr = np.interp(tau_in, tau_arr[:N_w//2], I_PDS[:N_w//2])
    I_arr /= I_arr[0]
    return np.log(I_arr)


def update(obj, val):
    global last_changed
    last_changed = obj
    I_PDS = np.exp(fitfun(t_arr[tau_idx], s1.val, s2.val))
    p_fit.set_ydata(I_PDS)
    fig.canvas.draw_idle()


def new_data(event, resample=True):
    if resample:
        while True:
            alpha = (alpha_max - alpha_min)*random() + alpha_min
            beta = (beta_max - beta_min)*random() + beta_min
            I_exp = np.exp(fitfun(tau_exp, alpha, beta))
            if not np.all(np.isnan(I_exp)):
                I_exp = np.nan_to_num(I_exp, nan=np.nanmin(I_exp))
                break
    else:
        alpha = s1.valinit
        beta = s2.valinit
        I_exp = np.exp(fitfun(tau_exp, alpha, beta))

    ax.set_ylim(np.min(I_exp)*0.1, np.max(I_exp)*10)
    noise_val = 1e-2 * s4.val 
    noise = 1.0 + np.random.normal(scale=noise_val, size=len(I_exp))
    I_exp *= noise
    s1.valinit=alpha
    s1.vline.set_xdata([alpha])
    s2.valinit=beta
    s2.vline.set_xdata([beta])
    if resample:    
        print('\n            alpha  beta')
        print('New data:   {:5.3f}, {:5.3f}'.format(alpha, beta))
    p_exp.set_ydata(I_exp)
    fig.canvas.draw_idle()


def fit(event):
    popt = [s1.val, s2.val]
    b2.color= 'coral'
    b2.hovercolor= 'coral'
    b2._motion(event)
    I_exp = p_exp.get_ydata()
    popt, pcov = curve_fit(fitfun, tau_exp, np.log(I_exp), p0=popt, 
                           bounds=([alpha_min, beta_min],[alpha_max, beta_max]))
    print('Fit values: {:5.3f}, {:5.3f}'.format(*popt))
    s1.set_val(popt[0])
    s2.set_val(popt[1])
    b2.hovercolor= 'whitesmoke'
    b2.color= 'lightgray'
    b2._motion(event)

    fig.canvas.draw()


def release_callback(event):
    global last_changed
    if last_changed == s3:
        new_data(event, resample=False)
        last_changed = None

    
#%% Plot initialization
fig, ax = plt.subplots(1, sharex=True, figsize=(8,7))
plt.subplots_adjust(top=0.77,bottom=0.21)

s1_ax = plt.axes([0.15, 0.1,  0.55, 0.03])
s1  = Slider(s1_ax, '$\\alpha$', alpha_min, alpha_max, valinit=alpha)
s1.on_changed(partial(update,s1))

s2_ax = plt.axes([0.15, 0.05,  0.55, 0.03])
s2  = Slider(s2_ax, '$\\beta$', beta_min, beta_max, valinit=beta)
s2.on_changed(partial(update,s2))

s3_ax = plt.axes([0.25, 0.85,  0.50, 0.03])
s3  = Slider(s3_ax, 'T (K)', T_min, T_max, valinit=T)
s3.on_changed(partial(update,s3))
last_changed = None
cid0 = fig.canvas.mpl_connect('button_release_event', release_callback)

s4_ax = plt.axes([0.25, 0.80,  0.50, 0.03])
s4  = Slider(s4_ax, 'Noise (%)', 0, 20.0, valinit=5.0)
s4.on_changed(lambda event: new_data(event, resample=False))


b1_ax = fig.add_axes([0.8, 0.1, 0.1, 0.03])
b1 = Button(b1_ax, 'New data')
b1.on_clicked(new_data)

b2_ax = fig.add_axes([0.8, 0.05, 0.1, 0.03])
b2 = Button(b2_ax, 'Fit')
b2.on_clicked(fit)

ax.set_title('Methane $\\nu_2$ {:s}-branch ({:.1f} M lines)\nFit MEG parameters at known temperature\n\n\n\n'.format(branch, Nl), fontsize=14)
ax.set_xlabel('Probe delay time (ps)')
ax.set_ylabel('Normalized CRS signal')

ax.grid(True)
ax.set_yscale('log')

p_fit, = ax.plot(t_arr[tau_idx]*1e12, np.ones_like(t_arr[tau_idx]), label='Fit')    
p_exp, = ax.plot(tau_exp*1e12, np.ones_like(tau_exp), 'o', mec='k', label='Data',zorder=-10)
ax.legend(loc=1)        

#update plots once:
new_data(None)
update(None,0.0)

plt.show()
