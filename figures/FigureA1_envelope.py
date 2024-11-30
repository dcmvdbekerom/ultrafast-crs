# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:19:10 2024

@author: bekeromdcmvd
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fft, fftshift, ifftshift
pi = np.pi
from scipy.constants import c
c *= 100.0 #cm.s-1
import sys

v_max = 2000.0 #cm-1
dv = 0.1 #cm-1
v_arr = np.arange(-v_max, v_max, dv)
N_v =  len(v_arr)
tau23 = 2e-12
tau=tau23
# print(dv, N_v)

dt_pr = 5e-12 #Hz
# tau23 = 20e-12 #s
Gi = 1.0e10 #.s-1
vi = 3.0 + dv/4#cm-1


w_max = 2*pi*c*v_max
dw = 2*pi*c*dv
wi = 2*pi*c*vi
w_arr = 2*pi*c*v_arr

# dw = 0.05 #s-1
# w_max = 20.0 #s-1
# w_arr = np.arange(-w_max, w_max, dw)
Nw = len(w_arr)
N_w=Nw
Nt = Nw
t_arr = fftfreq(Nw, d=dw/(2*pi))
dt = t_arr[1]
t_max = t_arr[Nt//2-1] + dt


def G(t, a): #a is FWHM
    return np.exp(-4*np.log(2)*(t/a)**2)

# def G_FT(f, a):
#     return 0.5*a*(np.pi/np.log(2))**0.5 * np.exp(-(np.pi*a*f)**2/(4*np.log(2)))

def G_FT(w, a):
    return 0.5*a*(np.pi/np.log(2))**0.5 * np.exp(-(a*w)**2/(16*np.log(2)))


def G1_FT(w, a):
    A = -a**2/(8*np.log(2))
    return A*w * G_FT(w, a)

def G2_FT(w, a):
    A = -a**2/(8*np.log(2))
    return A*G_FT(w, a) + A*w*G1_FT(w, a)

def G3_FT(w, a):
    A = -a**2/(8*np.log(2))
    return 2*A*G1_FT(w, a) + A*w*G2_FT(w, a)



Wi = 1.0
chi_i = np.exp((1j*wi - Gi)*tau23)

# Approximate w:
k = (wi - w_arr[0]) / dw
k0 = int(k)
k1 = k0 + 1
la_w = k - k0

w0 = w_arr[k0]
w1 = w_arr[k1]

tau23 = t_max/2
tau=tau23

a1 = (np.exp(1j*la_w*dw*tau23) - 1) / (np.exp(1j*dw*tau23) - 1)
# a1 = 0.25 - 0.1j

a0 = 1-a1
ah = a1 - 0.5
la_h = la_w - 0.5


q0 = 0.5*dw*tau
q  = 0.5*dw*t_arr
p = q - q0/2

p0_arr = p/q0 + 0.5

fig, ax = plt.subplots(1,sharex=True, figsize=(8,3))


# for a1 in np.arange(0.0,0.3,0.01):
#     yc = np.abs(1 + 2j*a1*np.sin(0.5*dw*t_arr)*np.exp(0.5j*dw*t_arr))
#     plt.plot(p0_arr[:N_w//2], yc[:N_w//2])


yi = Wi*np.exp((1j* wi                  - Gi)*t_arr)
y0 = Wi*np.exp((1j*(wi -  la_w     *dw) - Gi)*t_arr)
y1 = Wi*np.exp((1j*(wi - (la_w-1.0)*dw) - Gi)*t_arr)
yh = Wi*np.exp((1j*(wi - (la_w-0.5)*dw) - Gi)*t_arr)

ya = a0*y0 + a1*y1
yb = yi*((1-a1)*np.exp(-1j* la_w     *dw*t_arr) + 
          a1   *np.exp(-1j*(la_w-1.0)*dw*t_arr))





# yb = np.sqrt(ya/ya.conjugate())*np.sign(ya) / ya




# yb = (0.5 - ah)*y0 + (0.5 + ah)*y1
# yb = 0.5*(y0 + y1) + ah*(y1 - y0)
# yb = 0.5*yh*(np.exp(0.5j*dw*t_arr) + np.exp(-0.5j*dw*t_arr)) + ah*yh*(np.exp(0.5j*dw*t_arr) - np.exp(-0.5j*dw*t_arr))
# yb = yh*(
#     np.cos(0.5*dw*t_arr) 
#     + ah*2j*np.sin(0.5*dw*t_arr)
#     )

# yb = yi*(
#           0.5*(np.exp(0.5j*dw*t_arr) + np.exp(-0.5j*dw*t_arr)) * np.exp(-1j*la_h*dw*t_arr) 
#          + ah*(np.exp(0.5j*dw*t_arr) - np.exp(-0.5j*dw*t_arr)) * np.exp(-1j*la_h*dw*t_arr) 
#         )

# yb = yi*(
#                  0.5 *(np.exp(0.5j*dw*t_arr) + np.exp(-0.5j*dw*t_arr)) 
#          + (a1 - 0.5)*(np.exp(0.5j*dw*t_arr) - np.exp(-0.5j*dw*t_arr))
#         )* np.exp(-1j*(la_w-0.5)*dw*t_arr) 

# yb = yi*(
#                       np.exp(-0.5j*dw*t_arr)
#          +        a1*(np.exp( 0.5j*dw*t_arr) - np.exp(-0.5j*dw*t_arr))
#         )* np.exp(-1j*(la_w-0.5)*dw*t_arr) 

# yb = yi*(1 + a1*(np.exp(1j*dw*t_arr) - 1)) * np.exp(-1j*la_w*dw*t_arr) 

# So we have an accumilating error in phase as well as in amplitude.
# the error in phase is given by * np.exp(-1j*la_w*dw*t_arr)
# the error in ampli is given by 1 + a1*(np.exp(1j*dw*t_arr) - 1)


yc = 1 + a1*(np.exp(1j*dw*t_arr) - 1)

yc = 1 + 2j*a1*np.sin(0.5*dw*t_arr)*np.exp(0.5j*dw*t_arr)


#we continue with a1=la_w

yc = np.abs(1 + a1*(np.exp(1j*dw*t_arr) - 1))

yc = np.abs(np.exp(-0.5j*dw*t_arr) + a1*(np.exp(0.5j*dw*t_arr) - np.exp(-0.5j*dw*t_arr)))

yc = np.abs(np.exp(-0.5j*dw*t_arr) + 2j*a1*np.sin(0.5*dw*t_arr))

yc = ( 
       (np.exp(-0.5j*dw*t_arr) + 2j*a1            *np.sin(0.5*dw*t_arr)) *
       (np.exp( 0.5j*dw*t_arr) - 2j*a1.conjugate()*np.sin(0.5*dw*t_arr))
      )**0.5

# #when la=0.5, we don't need to take the conjugates it seems; we only need the real part of:
# yc = (np.exp(-0.5j*dw*t_arr) + 2j*a1*np.sin(0.5*dw*t_arr))

# yc = (np.cos(0.5*dw*t_arr) - 2*np.imag(a1)*np.sin(0.5*dw*t_arr))


# yc = ( 
#           1.0
#         - 4*np.imag(a1*np.exp(0.5j*dw*t_arr))*np.sin(0.5*dw*t_arr)
#         + 4*np.abs(a1)**2                    *np.sin(0.5*dw*t_arr)**2
#         )**0.5



yc0 =0.5 + 0.5*(  np.sin(     la_w *q0) 
                + np.sin((1 - la_w)*q0)) / np.sin(q0)

# a1 = np.sin(la_w*q0)/np.sin(q0) * np.exp(1j*q0*(la_w-1)) 
# a0 = 1-a1


# yc = ( 
#           1.0
#         - 4*np.imag(a1*np.exp(1j*q))*np.sin(q)
#         + 4*np.abs(a1)**2           *np.sin(q)**2
#         )**0.5

yc = ( 
          1.0
        - 4*np.sin(la_w*q0)    * np.sin(q - q0*(1-la_w)) * np.sin(q)    / np.sin(q0)
        + 4*np.sin(la_w*q0)**2                           * np.sin(q)**2 / np.sin(q0)**2
        )**0.5



# #0.5*(cos(a+b)-cos(a-b)) = sin(a)*sin(b)


yc = ( 
          1.0
        - 4*np.sin(p + q0*(la_w-0.5)) * np.sin(p + 0.5*q0) * np.sin(la_w*q0) / np.sin(q0)
        + 4*np.sin(la_w*q0)**2 * np.sin(p + q0/2)**2                         / np.sin(q0)**2
        )**0.5




yc = ( 
          1.0
        - 4*(
            
              np.sin(p)**2    *np.cos(q0*(la_w-0.5))*np.cos(0.5*q0)
            + np.cos(p)**2    *np.sin(q0*(la_w-0.5))*np.sin(0.5*q0) 
                        
            ) * np.sin(la_w*q0) / np.sin(q0)
        
        + 4* (
              np.sin(p)**2    *np.cos(q0/2)**2 
            + np.cos(p)**2    *np.sin(q0/2)**2
            
            ) *np.sin(la_w*q0)**2 / np.sin(q0)**2
        )**0.5


yc = ( 
          1.0
        - 2*(np.cos((1-la_w)*q0) - np.cos(2*p)*np.cos(q0*la_w)) * np.sin(la_w*q0)/np.sin(q0)
        + 2*(1 - np.cos(2*p)*np.cos(q0)) * np.sin(la_w*q0)**2/np.sin(q0)**2
        )**0.5


yc = ( 
          1.0
        - 2*np.cos((1-la_w)*q0) * np.sin(la_w*q0)   /np.sin(q0)
        + 2                     * np.sin(la_w*q0)**2/np.sin(q0)**2

        + 2*np.cos(2*p)*np.cos(q0*la_w) * np.sin(la_w*q0)   /np.sin(q0)
        - 2*np.cos(2*p)*np.cos(q0)      * np.sin(la_w*q0)**2/np.sin(q0)**2
        
        )**0.5

yc = (           
        - 2*np.cos((1-la_w)*q0) * np.sin(la_w*q0)*np.sin(q0)
        + np.sin(q0)**2 + 2*np.sin(la_w*q0)**2

        + 2*np.cos(2*p) * np.sin((1-la_w)*q0)*np.sin(la_w*q0)
        
        )**0.5 / np.sin(q0)



yc = ( 
          
        - 2*np.cos((1-la_w)*q0) * np.sin(la_w*q0)*np.sin(q0)
        + np.sin(q0)**2 + 2*np.sin(la_w*q0)**2

        + 2*np.cos(2*p) * np.sin((1-la_w)*q0)*np.sin(la_w*q0)
        
        )**0.5 / np.sin(q0)


yc = ( 
          1 - np.cos(q0)*np.cos((1 - 2*la_w)*q0)
        + 2*np.cos(2*p)*np.sin((1 - la_w)*q0)*np.sin(la_w*q0)
        
        )**0.5 / np.sin(q0)



yc = ( 
        np.sin(q0)**2 
        + np.sin(la_w*q0)*(np.sin(la_w*q0) - np.sin(2*q0 - la_w*q0))

          
        + 2*np.cos(2*p)*np.sin((1 - la_w)*q0)*np.sin(la_w*q0) 
        )**0.5 / np.sin(q0)



yc = ( 
        np.sin(q0)*np.sin(q0) 
        -2*np.sin(la_w*q0)*np.sin(q0)*np.cos((1-la_w)*q0)
        +2*np.sin(la_w*q0)*np.sin(la_w*q0)
          
        + 2*np.cos(2*p)*np.sin((1 - la_w)*q0)*np.sin(la_w*q0) 
        )**0.5 / np.sin(q0)



yc = ( 
        - 2*np.sin(q0)*np.cos(q0)*np.cos(la_w*q0)*np.sin(la_w*q0)
        +   np.sin(q0)*np.sin(q0) 
        - 2*np.sin(q0)*np.sin(q0)*np.sin(la_w*q0)*np.sin(la_w*q0)
        + 2                      *np.sin(la_w*q0)*np.sin(la_w*q0)
          
        + 2*np.cos(2*p)*np.sin((1 - la_w)*q0)*np.sin(la_w*q0) 
        )**0.5 / np.sin(q0)




yc = ( 

      + 1.0
      - np.sin(q0)*np.cos(q0)*np.sin(2*la_w*q0)
      - np.cos(q0)*np.cos(q0)*np.cos(2*la_w*q0)

        + 2*np.cos(2*p)*np.sin((1 - la_w)*q0)*np.sin(la_w*q0) 
        )**0.5 / np.sin(q0)

yc = ( 

      + 1.0
      - 0.5*np.cos(2*(1-la_w)*q0)
      - 0.5*np.cos(2*la_w*q0)
      
      + 2*np.cos(2*p)*np.sin((1 - la_w)*q0)*np.sin(la_w*q0) 
        )**0.5 / np.sin(q0)


yc = ( 
      + np.sin(     la_w *q0)**2
      + np.sin((1 - la_w)*q0)**2 
      + np.sin((1 - la_w)*q0)*np.sin(la_w*q0) * 2*np.cos(2*p)
        )**0.5 / np.sin(q0)


yc = np.abs( 
      + np.exp( 1j*p)*np.sin(     la_w *q0) 
      + np.exp(-1j*p)*np.sin((1 - la_w)*q0)
        ) / np.sin(q0)





yc = np.abs(  np.exp( 1j*p)*np.exp( 1j*la_w*q0)*np.exp( 0.0j*q0)/2j 
            - np.exp( 1j*p)*np.exp(-1j*la_w*q0)*np.exp(-0.0j*q0)/2j 
            - np.exp(-1j*p)*np.exp( 1j*la_w*q0)*np.exp(-1.0j*q0)/2j
            + np.exp(-1j*p)*np.exp(-1j*la_w*q0)*np.exp( 1.0j*q0)/2j
  
            ) / np.sin(q0)


yc = np.abs(  np.exp( 1j*p)*np.abs(a1) 
            + np.exp(-1j*p)*np.abs(a0)
        )
# yc = np.abs(1 + 2j*a1*np.sin(0.5*dw*t_arr)*np.exp(0.5j*dw*t_arr))


yc00 = (   np.exp( 1j*la_w*q0)*np.exp( 0.0j*q0)/2j 
          -np.exp(-1j*la_w*q0)*np.exp(-0.0j*q0)/2j 
          -np.exp( 1j*la_w*q0)*np.exp(-1.0j*q0)/2j
          +np.exp(-1j*la_w*q0)*np.exp( 1.0j*q0)/2j
            ) / np.sin(q0)


yc00 = a1**0.5*a1.conjugate()**0.5 + (1 - 2*np.real(a1) + a1*a1.conjugate())**0.5
# yc00inv = 1/yc00

x = la_w - 0.5
#approximate:

yc00inv = 1 + 0.5*(x**2  - 0.25)*q0**2
# y_corr = 0.5 + 0.5*yc00inv

q0 = 0.5*dw*tau
y_corr = 1 + 0.25*(x**2  - 0.25)*q0**2
y_corr = 1 - q0**2/16 + q0**2/4 * x**2  


y_corr = 1 - la_w*(1-la_w)*q0**2/4

# yc00inv = .1
# a1 = (np.exp(1j*la_w*q0) - np.exp(-1j*la_w*q0))  * np.exp(1j*q0*(la_w-1)) 




# la_h = la_w - 0.5
# yc0 =0.5 + 0.5*(  np.sin((la_h+0.5)*q0) 
#                 + np.sin((la_h-0.5)*q0)) / np.sin(q0)




yd = 1-4*(q-q0)*q/q0**2 * (1/(2*(y_corr-0.5))-1)

( 
      + np.sin(     la_w *q0)**2
      + np.sin((1 - la_w)*q0)**2 
      + np.sin((1 - la_w)*q0)*np.sin(la_w*q0) * 2*np.cos(2*p)
        )**0.5 / np.sin(q0)


t = t_arr

la = la_w
tau = tau23
t_ = t/tau
q = 1j*dw*tau23
mu = la - 0.5
t0 = t/tau + 0.5
ds = 1j*dw
dx = ds*tau
dx2 = 0.5*dx

# ya = (  a0 * np.exp((1j*wi - (mu + 0.5)*ds - Gi)*(t_arr + tau)) * G(t_arr, dt_pr)
#       + a1 * np.exp((1j*wi - (mu - 0.5)*ds - Gi)*(t_arr + tau)) * G(t_arr, dt_pr))

# ya = (  a0 * np.exp((1j*wi - (mu + 0.5)*ds - Gi)*(t_arr + tau))
#       + a1 * np.exp((1j*wi - (mu - 0.5)*ds - Gi)*(t_arr + tau))) * G(t_arr, dt_pr)

# ya = (  a0 * np.exp(-2*(mu + 0.5)*dx2*(t_ + 1))
#       + a1 * np.exp(-2*(mu - 0.5)*dx2*(t_ + 1))) * yi

# a0 = -(np.exp(2*mu*dx2) - np.exp( dx2)) / (np.exp(dx2) - np.exp(-dx2))
# a1 =  (np.exp(2*mu*dx2) - np.exp(-dx2)) / (np.exp(dx2) - np.exp(-dx2))

# ya = (- np.exp(( 2*mu - 2*(mu + 0.5)*(t_ + 1))*dx2)
#       + np.exp(( 1    - 2*(mu + 0.5)*(t_ + 1))*dx2)
#       + np.exp(( 2*mu - 2*(mu - 0.5)*(t_ + 1))*dx2)
#       - np.exp((-1    - 2*(mu - 0.5)*(t_ + 1))*dx2)
#       )  / (np.exp(dx2) - np.exp(-dx2)) * yi

# ya = (+ np.exp((-2*(mu + 0.5)*t_ - 2*mu)*dx2)
#       - np.exp((-2*(mu + 0.5)*t_ - 1   )*dx2)
#       + np.exp((-2*(mu - 0.5)*t_ + 1   )*dx2)
#       - np.exp((-2*(mu - 0.5)*t_ - 2*mu)*dx2)
#       )  / (np.exp(dx2) - np.exp(-dx2)) * yi

# ya = (+ np.exp((-(2*mu + 1)*t0 - (mu - 0.5))*dx2)
#       - np.exp((-(2*mu + 1)*t0 + (mu - 0.5))*dx2)
#       + np.exp((-(2*mu - 1)*t0 + (mu + 0.5))*dx2)
#       - np.exp((-(2*mu - 1)*t0 - (mu + 0.5))*dx2)
#       )  / (np.exp(dx2) - np.exp(-dx2)) * yi


# {0:0,
#  1:2,
#  2:0,
#  3:6*mu**2 - 24*t0**2*(mu**2 - 0.25) + 0.5,
#  4:128*mu*t0*(mu**2 - 0.25)*(t0**2 - 0.25),
#  5:10*mu**4 + 5.0*mu**2 - 480*t0**4*(1.0*mu**4 - 1/6*mu**2 - 1/48) - 480*t0**2*(-1/6*mu**4 + 1/12*mu**2 -1/96) + 1/8
#  }

# ya = (2*dx2 
#       +(6*mu**2 - 24*t0**2*(mu**2 - 0.25) + 0.5)*dx2**3/6
#       +(128*mu*t0*(mu**2 - 0.25)*(t0**2 - 0.25))*dx2**4/24        
#       ) / (2*dx2 + 2*dx2**3/6 + 2*dx2**5/120) * yi

# ya = yi + (
#         -        (mu**2 - 0.25)*(t0**2 - 0.25)*dx**2/2
#         +  mu*t0*(mu**2 - 0.25)*(t0**2 - 0.25)*dx**3/3        
#       ) * yi

# ya = yi + (
#         -    (mu**2 - 0.25)*t_*(t_ + 1)           *dx**2/2
#         + mu*(mu**2 - 0.25)*t_*(t_ + 1)*(t_ + 0.5)*dx**3/3        
#       ) * yi

# dwt = dw*tau


# ya = yi + dwt**2/2*t_*yi*(+             (mu**2 - 1/4)*(            t_ + 1) 
#                           - 1j*dwt/3*mu*(mu**2 - 1/4)*(2*t_**2 + 3*t_ + 1)) 






# ya = a0*y0 + a1*y1

# ya = ((
#         np.exp(0.5j*dw*tau23) 
#       - np.exp(1j*(la_w-0.5)*dw*tau23) 
#       + np.exp(1j*dw*((la_w + 0.5)*tau23 + t_arr)) 
#       - np.exp(1j*dw*(t_arr + 0.5*tau23)))
     
    
#       / (np.exp(0.5j*dw*tau23) - np.exp(-0.5j*dw*tau23))
 
#     * np.exp(-1j*la_w*dw*(t_arr + tau23)) * yi)




# ya = ((
#         (1 - np.exp(q*(la-1)))
#       - (1 - np.exp(q* la   )) * np.exp(q*t/tau)
#       ) * np.exp(-q*la*(t/tau + 1))
     
#       / (1 - np.exp(-q))
 
#       * yi)


# ya = ((
#           np.exp(-q*(la +         la      * t/tau))
#         - np.exp(-q*(la +        (la - 1) * t/tau))
        
#         + np.exp(-q*(0.5 - 0.5 + (la - 1) * t/tau))
#         - np.exp(-q*(0.5 + 0.5 +  la      * t/tau))
#       ) 
     
#       / (1 - np.exp(-q))
 
#       * yi)





# ya = ((
#           np.exp(-q*((la - 0.5) +  la      * t/tau))
#         - np.exp(-q*((la - 0.5) + (la - 1) * t/tau))
        
#         + np.exp(-q*(-0.5 + (la - 1) * t/tau))
#         - np.exp(-q*( 0.5 +  la      * t/tau))
#       ) 
     
#       / (np.exp(0.5*q) - np.exp(-0.5*q))
 
#       * yi)


sinh = lambda x: 0.5*(np.exp(x) - np.exp(-x))
cosh = lambda x: 0.5*(np.exp(x) + np.exp(-x))
# ya = ((
#         + np.exp(-q*(mu*(1 + t/tau) + 0.5 * t/tau))
#         - np.exp(-q*(mu*(1 + t/tau) - 0.5 * t/tau))
        
#         + np.exp(-q*(-0.5*(1 + t/tau) + mu * t/tau))
#         - np.exp(-q*( 0.5*(1 + t/tau) + mu * t/tau))
#       ) 
     
#       / (np.exp(0.5*q) - np.exp(-0.5*q))
 
#       * yi)







# ya = ((
#         + np.exp(-q*(mu*(t0 + 0.5) + 0.5*(t0 - 0.5)))
#         - np.exp(-q*(mu*(t0 + 0.5) - 0.5*(t0 - 0.5)))
        
#         + np.exp(-q*(mu*(t0 - 0.5) - 0.5*(t0 + 0.5)))
#         - np.exp(-q*(mu*(t0 - 0.5) + 0.5*(t0 + 0.5)))
#       ) 
     
#       / (np.exp(0.5*q) - np.exp(-0.5*q))
 
#       * yi)






# ya = ((
#         + np.exp(-q*(mu*(t0 + 0.5) + 0.5*(t0 - 0.5)))
#         + np.exp(-q*(mu*(t0 - 0.5) - 0.5*(t0 + 0.5)))

#         - np.exp(-q*(mu*(t0 + 0.5) - 0.5*(t0 - 0.5)))        
#         - np.exp(-q*(mu*(t0 - 0.5) + 0.5*(t0 + 0.5)))
#       ) 
     
#       / (np.exp(0.5*q) - np.exp(-0.5*q))
 
#       * yi)

# sinh(x)*sinh(y) = (exp(x) - exp(-x))*(exp(y) - exp (-y)) = (exp(x+y) - exp(x-y) - exp(-x+y) + exp(-x-y))
# sinh(x)*cosh(x) = (exp(x) - exp(-x))*(exp(y) + exp (-y)) = (exp(x+y) - exp(x-y) + exp(-x+y) - exp(-x-y))

# ya = (np.exp(-q*mu*t0) * (
#         + np.exp(-q*(+0.5*mu + 0.5*(t0 - 0.5)))
#         - np.exp(-q*(+0.5*mu - 0.5*(t0 - 0.5)))
        
#         - np.exp(-q*(-0.5*mu + 0.5*(t0 + 0.5)))
#         + np.exp(-q*(-0.5*mu - 0.5*(t0 + 0.5)))
        
#       ) 
     
#       / (np.exp(0.5*q) - np.exp(-0.5*q))
 
#       * yi)
# sinh(x + y) - sinh(x - y) = (exp(x+y) - exp(-x-y))(exp(x-y) - exp(-x+y)) = cosh(2x) - cosh(2y)

# ya = (np.exp(-q*mu*t0) * (
#         + np.exp(-0.5*q*mu) * sinh(-q*0.5*(t0 - 0.5))
#         - np.exp( 0.5*q*mu) * sinh(-q*0.5*(t0 + 0.5))
        
#       ) / sinh(0.5*q)
       
#              * yi)




# ya = ((
#         + np.exp(-q*(mu*(t0 + 0.5) - 0.25 + 0.5*t0))
#         + np.exp(-q*(mu*(t0 - 0.5) - 0.25 - 0.5*t0))
        
#         - np.exp(-q*(mu*(t0 + 0.5) + 0.25 - 0.5*t0))        
#         - np.exp(-q*(mu*(t0 - 0.5) + 0.25 + 0.5*t0))
#       ) 
     
#       / (np.exp(0.5*q) - np.exp(-0.5*q))
 
#       * yi)



# t0 = t/tau + 0.5

# ya = (( + np.exp(-q*mu*(t0 + 0.5))*sinh(-q*0.5*(t0 - 0.5))
#         - np.exp(-q*mu*(t0 - 0.5))*sinh(-q*0.5*(t0 + 0.5))
#       ) 
     
#       / sinh(0.5*q)
 
#       * yi)


y_err = ya - yi


# ya = ((1 
#         - q/2
#         - q**2*((mu**2 - 0.25)*t_*(t_ + 1)  - 1/3)/2     
      
#         +q**3*(
#               t_**3*(2*mu**3/3           - mu/6       ) 
#             + t_**2*  (mu**3   + mu**2/2 - mu/4  - 1/8) 
#             + t_   *(  mu**3/3 + mu**2/2 - mu/12 - 1/8) 
#             - 1/12
#             )/2  
#       ) 
     
#       / (1 - q/2 + q**2/6 - q**3/24)
 
#       * yi)




# ya = yi - q**2/2 * (( 
        
#         + (1 - q/2) * (mu**2 - 0.25)*t_*(t_ + 1)       / (1 - q/2 + q**2/6 - q**3/24)
#         - q*(mu**3/3 - mu/12) *t_*(1 + 3*t_ + 2*t_**2) / (1 - q/2 + q**2/6 - q**3/24)
#       ) * yi)



q = 1j*dw*tau23
dwt = dw*tau23



#%% further processing:

plt.subplots_adjust(top=0.92, bottom=0.2,left=0.08,right=0.96,)


q0 = 0.5*dw*tau
q  = 0.5*dw*t_arr
# p = q - q0/2v

p0_max = np.pi/(dw*tau)



p0_arr = q/q0

yc0inv=1.0
ax=[ax]
t_arr2 = ifftshift(t_arr)
ax[0].axvline(0,c='k',alpha=0.25)
ax[0].axvline(1,c='k',alpha=0.25)
ax[0].axvline(p0_max,c='k',alpha=0.25)
ax[0].axhline(0,c='k',alpha=0.5)
ax[0].axhline(1,c='k',alpha=0.5)
ax[0].plot(p0_arr[:Nw//2], np.clip((yi.real/np.exp(-Gi*t_arr))[:Nw//2],-np.inf,np.inf),  c='tab:blue', lw=1.5, label='$\\chi_i$', zorder=-100)
# ax[0].plot(t_arr, yi.imag,  'b', lw=1)

ax[0].plot(p0_arr[:Nw//2], np.clip((yb.real/np.exp(-Gi*t_arr))[:Nw//2]*yc0inv,-np.inf,np.inf),  c='tab:red', lw=1.5, label = '$\\chi_{apx}$',zorder=-100)
# ax[0].plot(p0_arr[:Nw//2], yb[:Nw//2],  'r-', lw=1.5, label = '$\\chi_{apx}$')
# ax[0].plot([1],[yc0inv],'k+',zorder=10,mfc='none',ms=12, mew=2, label='$\\tau_0=\\tau_{last}$') #,yc0inv/((y_corr-0.5)*2)


plt.text(0,1.05,'$\\tau=0$')
plt.text(1,1.05,' $\\tau=\\tau_0$', ha='center')
plt.text(p0_max,1.05,'$\\tau=\\tau_{max}$ ',ha='right',zorder=1000)


# ax[0].plot(p0_arr[:Nw//2], np.clip((ya.real/np.exp(-Gi*t_arr))[:Nw//2]*yc0inv,-np.inf,np.inf),  'k--', lw=1.5, label = '$\\chi_{apx}$ (alt)')
# ax[0].plot(t_arr, ya.imag,  'k--', lw=1)

ax[0].plot(p0_arr[:Nw//2], yc[:Nw//2],  '-', c='tab:purple', lw=1.5, label='envelope', zorder=-100)


idx = (0.0 <= p0_arr)&(p0_arr <= 1.0)
ax[0].plot(p0_arr[idx], yc[idx]*y_corr,  'k--', lw=1.5, label='envelope\nafter scaling')
ax[0].fill_between(p0_arr[idx], yc[idx]*y_corr, 1.0, alpha=0.2, zorder=100)
# ax[0].axhline(yc00)

ax[0].set_xlabel('$\\tau / \\tau_{0}$')
ax[0].legend(loc=3)
ax[0].set_ylim(0.6,1.1)
ax[0].set_xlim(-0.05,p0_max + 0.05)
ax[0].fill_betweenx([0,1.0], 1.0, p0_max*1.1, zorder=-10, fc='w', alpha=0.75)


plt.savefig('output/FigureA1_envelope.png', dpi=150)
plt.savefig('output/FigureA1_envelope.pdf', dpi=150)