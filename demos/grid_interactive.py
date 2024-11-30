import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
# mpl.rc('font',family='Times New Roman')

def gG(x):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*x**2)

def gL(x):
    return 2/((np.pi) * (4*x**2 + 1))

def gLc(x):
    return np.imag(1/(np.pi * (2j*x + 1)))


class LineObject:
    def __init__(self, k=0, l=0, A=1.0, g=gL, w=1.0, ax=None, show_labels=False):
        
        self.show_labels = show_labels
        self.A = A
        self.g = g
        self.w = w
        
        self.init_plots(ax)
        self.update_plots(k,l)    
        


    def init_plots(self, ax):
        
        self.pv, = ax[0].plot((0.5,0.5), (0,1),'k--', alpha=0.25, lw=1) # ch ver
        self.ph, = ax[0].plot((0,1), (0.5,0.5),'k--', alpha=0.25, lw=1) # ch hor
        self.psn, = ax[0].plot((0,0), (0,0), 'g-', alpha=0.25,lw=2, zorder=0) #snap line
        self.pch, = ax[0].plot((0.5,), (0.5,),'k+', ms=9, zorder=20) #ch center

        
        self.ppt = ax[0].scatter((0,0,1,1), (0,1,0,1), color='tab:red', marker='D', zorder=10) #quad points.
        c = self.ppt.get_ec()
        self.colors = np.vstack((c,c,c,c))
        
        w = self.w
        self.p00, = ax[1].plot(w_arr, w*self.g(w_arr/w), c='tab:red', label=('a00..a11' if self.show_labels else None))
        self.p01, = ax[1].plot(w_arr, w*self.g(w_arr/w), c='tab:red', label=(None if self.show_labels else None)) #'a01'
        self.p10, = ax[1].plot(w_arr, w*self.g(w_arr/w), c='tab:red', label=(None if self.show_labels else None)) #'a10'
        self.p11, = ax[1].plot(w_arr, w*self.g(w_arr/w), c='tab:red', label=(None if self.show_labels else None)) #'a11'
        self.pa0, = ax[1].plot(w_arr, w*self.g(w_arr/w),c='tab:blue',lw=2, label=('Approx.' if self.show_labels else None))
        self.pa1, = ax[1].plot(w_arr, w*self.g(w_arr/w),'w',lw=1, alpha=1)
        self.pex, = ax[1].plot(w_arr, w*self.g(w_arr/w),'k--',lw=1, label=('Exact' if self.show_labels else None))




    def update_plots(self, *vargs):
        
        if len(vargs):
            self.k, self.l = vargs
        k,l =self.k, self.l

        k0, l0 = int(k), int(l)
        k1, l1 = k0 + 1, l0 + 1
        la_w, la_G = k - k0, l - l0

        w0  = wmin + k0*dw
        w1  = wmin + k1*dw
        wex = wmin + k *dw
        G0  = Gmin * np.exp(l0 * dxG)
        G1  = Gmin * np.exp(l1 * dxG)
        Gex = Gmin * np.exp(l  * dxG)
        tau23 = 1e-10
        
        av = la_w
        aG = la_G
        av = (np.exp(-1j*wex*tau23) - np.exp(-1j*w0*tau23)) / (np.exp(-1j*w1*tau23) - np.exp(-1j*w0*tau23))
        aG = (np.exp(-Gex*tau23) - np.exp(-G0*tau23)) / (np.exp(-G1*tau23) - np.exp(-G0*tau23))

        a00 = (1 - av) * (1 - aG)
        a01 = (1 - av) * aG
        a10 = av * (1 - aG)
        a11 = av * aG

        w = self.w
        y00 = self.A * self.g((w_arr - w0  + 1j*G0 ) / w ) / w
        y01 = self.A * self.g((w_arr - w0  + 1j*G1 ) / w ) / w
        y10 = self.A * self.g((w_arr - w1  + 1j*G0 ) / w ) / w
        y11 = self.A * self.g((w_arr - w1  + 1j*G1 ) / w ) / w
        yex = self.A * self.g((w_arr - wex + 1j*Gex) / w ) / w
        yap = a00*y00 + a01*y01 + a10*y10 + a11*y11
        
        
        self.pv.set_data((k,k),(l0,l1)) # ver crosshair
        self.ph.set_data((k0,k1),(l,l)) # hor crosshair
        self.pch.set_data((k,),(l,))    # center crosshair
        self.ppt.set_offsets(np.c_[(k0,k0,k1,k1),(l0,l1,l0,l1)]) # pos quad points

        self.colors[:,3] = (np.abs(a00),np.abs(a01),np.abs(a10),np.abs(a11)) # alpha quad points
        self.ppt.set_color(self.colors) 

        self.ph.set_alpha((0.0 if not la_w else 0.25))
        self.pv.set_alpha((0.0 if not la_G else 0.25))
        
        self.p00.set_ydata(np.real(y00))
        self.p01.set_ydata(np.real(y01))
        self.p10.set_ydata(np.real(y10))
        self.p11.set_ydata(np.real(y11))
        self.pa0.set_ydata(np.real(yap))
        self.pa1.set_ydata(np.real(yex))
        self.pex.set_ydata(np.real(yex))
        
        self.p00.set_alpha(np.abs(a00))
        self.p01.set_alpha(np.abs(a01))
        self.p10.set_alpha(np.abs(a10))
        self.p11.set_alpha(np.abs(a11))
        

class WeightsPlot:
    def __init__(self, axe, axw, axG):
        self.axe = axe
        self.axw = axw
        self.axG = axG
        
    def init_plots(self):
        self.axe.xaxis.tick_top()
        # self.axe.yaxis.tick_right()
        self.axe.set_xticks([0,1], [0,1], size=14)
        self.axw.set_xticks([0,1], [0,1], size=14)
        self.axe.set_yticks([0,1], [0,1], size=14)
        self.axG.set_yticks([0,1], [0,1], size=14)

        size = 10
        self.axe.text( 1.0, 1.15,'$a_{11}=$\n$a_\\omega a_\\Gamma$',ha='center', size=size) #
        self.axe.text( 0.0, 1.15,'$a_{01}=$\n$(1-a_\\omega) a_\\Gamma$',ha='center', size=size) #
        self.axe.text( 1.15,-0.15,'$a_{10}=$\n$a_\\omega (1-a_\\Gamma)$', ha='center', va='top', size=size) #
        self.axe.text(0.05,-0.15,'$a_{00}=$\n$(1-a_\\omega) (1-a_\\Gamma)$',ha='center',va='top', size=size) #


        self.axe.grid()
        self.colors = np.zeros((4,4), dtype=float)
        self.colors[:,0] = 1.0 # set to red
        self.colors[:,3] = (1,0,0,0) # alpha quad points

        self.axepts = self.axe.scatter((0,0,1,1), (0,1,0,1), color=self.colors, marker='D', s=100, zorder=10) #quad points #color=colors,
        self.axech, = self.axe.plot((0,), (0,),'k+', ms=10, mew=1.7, zorder=100) #ch center
        
        #Annotations
        ap = dict(arrowstyle='-', ls='--', lw=1.5, alpha=0.5, ec='k')
        self.axeG = self.axe.annotate('', xytext=(0,0), xy=(0,0), xycoords  =self.axG.transData, textcoords=self.axe.transData, arrowprops=ap, zorder=10)
        self.axew = self.axe.annotate('', xytext=(0,1), xy=(0,0), xycoords  =self.axw.transData, textcoords=self.axe.transData, arrowprops=ap, zorder=10)
        

        # Gamma
        self.axG.sharey(self.axe)
        self.axG.yaxis.tick_right()
        self.axG.yaxis.set_label_position('right') 

        
        self.axG.set_xticks([0,0.25,0.5,0.75, 1.0], [0,0.25,0.5,0.75, 1.0], size=14)
        self.axG.set_xlim(1,0)
        # axG.set_ylim(lmin,lmax)
        self.axG.set_ylabel('$\\ell$', size=14)#
        
        self.axG.axhline(0,c='k',lw=0.5,alpha=0.5)#, zorder=-10)
        self.axG.axhline(1,c='k',lw=0.5,alpha=0.5)#, zorder=-10)
        
        # self.axG.axvline(0, c='k', lw=1)
        # self.axG.axvline(1, c='k', lw=1)
        
        self.axGbar = self.axG.barh([0,1], [1,0], height=.90,fc='cornsilk',ec='k')
        
        size=14
        self.axG.text(0.5,1,'$a_\\Gamma$',  fontsize = size,ha='center', va='center').set_bbox(dict(fc='w', alpha=0.8, ec='none'))#
        self.axG.text(0.5,0,'$1-a_\\Gamma$',fontsize = size,ha='center', va='center').set_bbox(dict(fc='w', alpha=0.8, ec='none'))#
        
        self.axGla = self.axG.text(0.2, 0.0,'$\\lambda_\\Gamma$', va='center',size=size)#  
        self.axGla.set_bbox(dict(fc='w', alpha=0.8, ec='none'))
        # axG.plot([0,0.1],[l0,l0],'k--',lw=1.5)
        self.axGat = self.axG.annotate('', xy=(0.05,0), xytext=(0.05,0), arrowprops=dict(arrowstyle='<->'), zorder=10)


        # omega
        self.axw.sharex(self.axe)
        self.axw.yaxis.tick_right()
        self.axw.yaxis.set_label_position('right') 
        
        self.axw.set_yticks([0,0.25,0.5,0.75, 1.0], [0,0.25,0.5,0.75,1.0], size=14)
        # axw.set_xlim(mkmin,mkmax)
        self.axw.set_ylim(0,1)
        self.axw.set_xlabel('$k$', size=14)#
        
        self.axw.axvline(0,c='k',lw=0.5,alpha=0.5)#, zorder=-10)
        self.axw.axvline(1,c='k',lw=0.5,alpha=0.5)#, zorder=-10)
        
        
        # self.axw.axhline(0, c='k', lw=1)
        # self.axw.axhline(1, c='k', lw=1)
        self.axwbar = self.axw.bar([0,1], [1,0], width=.90,fc='cornsilk',ec='k')
        
        size=14
        self.axw.text(1,0.5,'$a_\\omega$',fontsize = size, ha='center', va='center').set_bbox(dict(fc='w', alpha=0.8, ec='none'))#
        self.axw.text(0,0.5,'$1-a_\\omega$',fontsize = size, ha='center', va='center').set_bbox(dict(fc='w', alpha=0.8, ec='none'))#
        
        self.axwla = self.axw.text(0,0.1, '$\\lambda_\\omega$', ha='center',size=size)# 
        self.axwla.set_bbox(dict(fc='w', alpha=0.8, ec='none'))
        # axw.plot([k0,k0],[0,0.1],'k--',lw=1.5)
        self.axwat = self.axw.annotate('', xy=(0,0.05), xytext=(0,0.05), arrowprops=dict(arrowstyle='<->'),zorder=10)



        
    def update_plots(self, k, l, colors):
        
        k0 = int(k)
        l0 = int(l)
        k1 = k0 + 1
        l1 = l0 + 1
        la_w = k-k0
        la_G = l-l0
        
        self.axe.set_xticklabels([k0,k1])
        self.axw.set_xticklabels([k0,k1])
        self.axe.set_yticklabels([l0,l1])
        self.axG.set_yticklabels([l0,l1])
        self.axwbar[0].set_height(1-la_w)
        self.axwbar[1].set_height(la_w)        
        self.axGbar[0].set_width(1-la_G)
        self.axGbar[1].set_width(la_G)
        self.axech.set_data([la_w], [la_G])

        self.axew.set_x(la_w)
        self.axew.xy = (la_w,0)
        self.axeG.set_y(la_G)
        self.axeG.xy = (0, la_G)
        self.axwla.set_x(la_w*0.5)
        self.axGla.set_y(la_G*0.5)
        
        self.axwat.set_x(la_w)
        self.axGat.set_y(la_G)
        self.axepts.set_color(colors)






fig = plt.figure(figsize=(18,5.6))

Ncols = 7


axG = plt.subplot2grid(shape=(2, Ncols), loc=(0,Ncols-1))
axw = plt.subplot2grid(shape=(2, Ncols), loc=(1,Ncols-2))
axe = plt.subplot2grid(shape=(2, Ncols), loc=(0,Ncols-2))


ax0 = plt.subplot2grid(shape=(2, Ncols), loc=(0,0), colspan=Ncols-2)
ax1 = plt.subplot2grid(shape=(2, Ncols), loc=(1,0), colspan=Ncols-2)
ax=[ax0,ax1]

# plt.ion()
plt.subplots_adjust(left=0.04, wspace=0.250, hspace=0.250)
        

kmin, kmax = 0, 35
lmin, lmax = 0, 4

wmin = 1800.0 #cm-1
Gmin = 0.15 #cm-1

dw = 0.1 #cm-1
dxG = 0.2 #cm-1

wmax = wmin + (kmax - kmin)*dw
w_arr = np.arange(wmin,wmax+dw,dw)

line_objs = []
A = 0.20
#line_objs.append(LineObject(A=0.5, g=gL, ax=ax, show_labels=True))
# line_objs.append(LineObject(ax=ax, g=gL, A=A, k=33.0, l=2.4))

#for 30:
# line_objs.append(LineObject(ax=ax, g=gL, A=A, k=24.80, l=0.6))
# line_objs.append(LineObject(ax=ax, g=gL, A=A, k=14.50, l=1.5, show_labels=True))
# line_objs.append(LineObject(ax=ax, g=gL, A=A, k= 4.67, l=3.0))


w = 0.5

# line_objs.append(LineObject(ax=ax, g=gG, A=A, k= 28.8, l=1.65, w=0.5))
# line_objs.append(LineObject(ax=ax, g=gG, A=A, k=20.50, l=2.5, w=0.5))
# line_objs.append(LineObject(ax=ax, g=gG, A=A, k=12.0, l=0.7, w=0.5))
line_objs.append(LineObject(ax=ax, g=gG, A=A, k=4.67, l=3.0, w=0.5, show_labels=True))

weight_obj = WeightsPlot(axe, axw, axG)
weight_obj.init_plots()

ax[0].set_xlim(kmin, kmax)
ax[0].set_ylim(lmin, lmax)
ax[0].set_xlabel('$k\\quad(\\omega[k]=\\omega_0 + k\\Delta\\omega)$\n', fontsize=16)#
ax[0].set_ylabel('$\\ell$\n$\\quad(\\Gamma[\\ell]=\\Gamma_0 + \\ell\\Delta\\Gamma)$', fontsize=16)#
# ax[0].set_title('$S[k,l]$:')
ax[0].set_aspect(1)
ax[0].xaxis.tick_top()
ax[0].xaxis.set_label_position('top') 

# ax[0].yaxis.tick_right()
# ax[0].yaxis.set_label_position('right') 

ax[0].grid(True)

ax[0].set_xticks(np.arange(kmin,kmax+1),labels=np.arange(kmin,kmax+1), size=14)
ax[0].set_yticks(np.arange(lmin,lmax+1),labels=np.arange(lmin,lmax+1), size=14)
# ax[0].text(0.2,0.90,"Press 'Shift' to align axes", fontsize=14, va='top', ha='right', transform = ax[0].transAxes)



ax[1].set_xlim(wmin, wmax)
ax[1].set_ylim(-0.2, 1.2)

ax[1].set_xticks(w_arr[::10], labels=map('{:.1f}'.format,w_arr[::10]), size=14)
ax[1].set_xlabel('$\\nu$ (cm$^{-1}$)', fontsize=16)#
ax[1].set_ylabel('$I$ (a.u.)', fontsize=16)#
# ax[1].set_title('$I_{CRS}(\\nu)$:')

ax[1].set_xticks(w_arr)
ax[1].set_yticks([0.0, 1.0], labels=[0.0, 1.0], size=14)

ax[1].grid(True, which='major')
ax[1].grid(True, which='minor')

# ax[1].tick_params(which='both', width=2)
ax[1].tick_params(which='major', length=7)
ax[1].tick_params(which='minor', length=4, color='r')
# ax[1].yaxis.tick_right()
# ax[1].yaxis.set_label_position('right') 

ax[1].legend(loc=1, fontsize=12)

snapped = (-1, -1)

from matplotlib.patches import FancyBboxPatch
rbox = FancyBboxPatch((28,1), 1,1, boxstyle="round,pad=0.5", fc="none", ec='k', zorder=4, alpha=0.5, lw=1)
ax0.add_patch(rbox)
rboxan = ax0.annotate('', xy=(28+1.5,1.5), xytext=(36,1.5), arrowprops=dict(arrowstyle='<-', ec='k', alpha=0.5), zorder=4, alpha=0.5)



def update(event):
    if event.inaxes != ax[0]:
        return

    if event.xdata and event.ydata:
        k = (event.xdata if snapped[0] < 0 else snapped[0])
        l = (event.ydata if snapped[1] < 0 else snapped[1])
        
        rbox.set_x(int(k))
        rbox.set_y(int(l))
        rboxan.xy = (int(k)+1.5, int(l)+0.5)
        rboxan.set_y(int(l)+0.5)
    
        line_objs[-1].update_plots(k,l)
        weight_obj.update_plots(k,l, line_objs[-1].colors)
        ax[1].legend(loc=1, fontsize=14)
        fig.canvas.draw_idle()
            

def on_press(event):
    global snapped
    if event.key == 'shift':
        if event.xdata and event.ydata:
            k, l = event.xdata, event.ydata
            ki, li = int(k + 0.5), int(l + 0.5)
            if np.abs(ki - k) < np.abs(li - l):
                snapped = (ki, -1)
                line_objs[-1].psn.set_data((ki, ki), (0, lmax))
            else:
                snapped = (-1, li)
                line_objs[-1].psn.set_data((0, kmax), (li, li))
            update(event)

        
def on_release(event):
    global snapped
    if event.key == 'shift':
        snapped = (-1, -1)
        line_objs[-1].psn.set_data((0,0), (0,0))
        update(event)


lo = line_objs[0]
weight_obj.update_plots(lo.k, lo.l, lo.colors)


fig.canvas.mpl_connect('motion_notify_event', update)
fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('key_release_event', on_release)

# c = plt.Circle((27.5, 1.5), radius=1.2, fc='none', ec='k', lw=0.5)




plt.tight_layout()
plt.subplots_adjust(left=0.045, wspace=0.250, hspace=0.250)

plt.ion()
plt.show()


###############
 

# plt.savefig('Figure1_grid.png', dpi=150)
# plt.savefig('Figure1_grid.pdf', dpi=150)