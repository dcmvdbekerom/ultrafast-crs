# -*- coding: utf-8 -*-

# This file is part of Ultrafast CRS

# Copyright (C) 2024  Dirk van den Bekerom - dcmvdbekerom@gmail.com

# Ultrafast CRS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


#%% Imports:
import numpy as np
from time import perf_counter
from os import cpu_count                      #for py functions
from scipy.fft import next_fast_len, fftfreq, ifft, fft, rfft, irfft, ifftshift  #for py functions
from functools import partial

def ptr(arr): #quickly get pointer for ndarrays
    return arr.ctypes.data

pi  = np.pi
c   = 29979245800.0 #cm.s-1
k_B = 1.380649e-23 #J.K-1
h   = 6.62607015e-34 #J.s
log2 = np.log(2)

REQUIRED_BYTE_ALIGNMENT = 32


#%% define Lineshape functions

def Gaussian(t, dt_FWHM):
    return np.exp(-4*log2*(t/dt_FWHM)**2)

def Gaussian_FT(w, dt_FWHM):
    return 0.5*dt_FWHM*(pi/log2)**0.5 * np.exp(-(dt_FWHM*w)**2/(16*log2))


class Database:
    def __init__(self, *vargs):
        
        self.J_min = -1
        self.J_max = -1
        self.EvJ1 = np.array([], dtype=np.float64)
        self.EvJ0 = np.array([], dtype=np.float64)
        self.N_EvJ = 0
        self.nu = np.array([], dtype=np.float64)
        self.sigma_gRmin = np.array([], dtype=np.float64)
        self.E0 = np.array([], dtype=np.float64)
        self.J_clip = np.array([], dtype=np.int32)
        self.N_lines = 0
        self.current_line = 0

        self.logger_thread = None
        self.jiggle_mouse = None
        self.jiggle_dir = 0        

        self.initialized = False


    def set_data(self, EvJ_data=None,
                        nu_data=None, sigma_gRmin_data=None, E0_data=None, J_clip_data=None, set_cpp=True, force_alignment=True):
        
        
        self.J_max = EvJ_data.shape[1] - 1
        self.J_min = np.argwhere(~np.isnan(EvJ_data[0]))[0,0]
        #TODO: check if all data arrays have been provided
    
        #TODO: check if sizes match J_max
        self.EvJ1 = EvJ_data[1]
        self.EvJ0 = EvJ_data[0]
        self.N_EvJ = len(self.EvJ1) 

        byte_align = REQUIRED_BYTE_ALIGNMENT if force_alignment else 1
        self.nu          = self.align_array(nu_data, nu_data[-1],         byte_align)
        self.sigma_gRmin = self.align_array(sigma_gRmin_data, 0.0,        byte_align)
        self.E0          = self.align_array(E0_data, E0_data[-1],         byte_align)
        self.J_clip      = self.align_array(J_clip_data, J_clip_data[-1], byte_align//2)
        self.N_lines = len(self.nu)
        
        self.aligned = self.is_aligned()
        self.set_cpp_database_refs()
        self.initialized = True
        return self.aligned
    
    
    def store_data(self, fname):
        np.save(fname, np.array([self.J_min, self.J_max,
                            self.EvJ1, self.EvJ0,
                            self.nu,
                            self.sigma_gRmin,
                            self.E0,
                            self.J_clip,
                            ], dtype=object))
        
        
    def set_cpp_database_refs(self):
        alignment_error = cpp.set_database_refs(
            self.J_min, self.J_max, self.EvJ1, self.EvJ0,
            self.nu, self.sigma_gRmin, self.E0, self.J_clip)

        return alignment_error

    def start_logger(self, t=0.0, jiggle_px=0):
        import time, threading, datetime, ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
        
        self.logger_t_start = time.perf_counter()
        self.logger_now = datetime.datetime.now()
        # if jiggle_px:        
        #     from pynput.mouse import Controller as MouseController
        #     self.jiggle_mouse = MouseController()
        #     self.jiggle_dir = jiggle_px

        def jiggle_func():
         
            # if jiggle_px:
            #     self.jiggle_mouse.move(self.jiggle_dir,0)
            #     self.jiggle_dir = -self.jiggle_dir

            if self.current_line < self.N_lines - 1:
                t0 = perf_counter()
                dt = t0 - self.logger_t_start
                tc = dt / (self.current_line+1) * self.N_lines
                complete = (self.logger_now + datetime.timedelta(seconds=tc)).strftime('%H:%M:%S')
                
                progress = (self.current_line + 1) / self.N_lines
                print('{:5.1f}% ({:.1f}s) - ETC: {:s}'.format(100*progress, dt, complete))
                
                self.logger_thread = threading.Timer(t, jiggle_func)
                self.logger_thread.start()   
            else:
                self.stop_logger()
        if t>0.0:
            self.printlog = True
            jiggle_func()
        else:
            self.printlog = False
        
    def stop_logger(self):
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if self.logger_thread:
            self.logger_thread.cancel()
        if self.printlog:
            t0 = perf_counter()
            dt = t0 - self.logger_t_start
            progress = (self.current_line + 1) / self.N_lines
            print('{:5.1f}% ({:.1f}s) Done!'.format(100*progress, dt))

    def is_aligned(self, byte_align=32, return_error_code=False):
        err = 0
        elem_align = byte_align // np.dtype(np.float64).itemsize
        if self.N_lines % elem_align: err |= 1
        for i, arr in enumerate([self.nu, self.sigma_gRmin, self.E0, self.J_clip]):
            if ptr(arr) % elem_align: err |= (i+1)
        return err if return_error_code else not err


    def align_array(self, arr, default=0.0, byte_align=32):
        
        # If the input array is already aligned, return the array itself.
        # If not, return a new array that is aligned.
        assert len(arr.shape) == 1 #TODO: only for vectors at the moment
    
        arr2 = self.zeros_aligned(arr.shape, dtype=arr.dtype, byte_align=byte_align, shape_warning=False)
        elem_align = byte_align // arr.dtype.itemsize
        if arr.shape[0] > elem_align:
            arr2[-elem_align:] = default        
        else:
            arr2[:] = default
        arr2[:arr.shape[0]] = arr
            
        return arr2
    
    
    @staticmethod
    def zeros_aligned(requested_shape, dtype=float, order='C', byte_align=32, shape_warning=True):
        try:    
            shape = [*requested_shape]
        except TypeError:
            shape = [requested_shape]
            
        itemsize = np.dtype(dtype).itemsize
        
        # If itemsize is larger than the alignment size, alignment is guaranteed
        if  itemsize >= byte_align:
            return np.zeros(shape, dtype=dtype, order=order)
        
        # The innermost axis size is checked for alignment, both start and end.
        # If the end doesn't align, either a warning is triggered of the axis is resized.
        aligned_axis = {'C':len(shape)-1, 'F':0}[order]
        elem_align = byte_align // itemsize
        rem = shape[aligned_axis] % elem_align
        
        if rem:
            if shape_warning:
                import warnings
                warnings.warn(f'Axis {aligned_axis:d} has length {shape[aligned_axis]:d}, which is not aligned with item alignemnt {elem_align:d} (remainder {rem:d})')
                return
            else:
                shape[aligned_axis] = shape[aligned_axis] + elem_align - rem
        
        nbytes = np.prod(shape)*itemsize
        buffer = np.zeros(nbytes + byte_align, dtype=np.uint8)
        rem = buffer.ctypes.data % byte_align
        offset = byte_align - rem if rem else 0
        arr = buffer[offset:offset + nbytes].view(dtype).reshape(shape, order=order)
        del buffer
        
        return arr



class DLLWrapper:
    def __init__(self):
        import os
        from ctypes import cdll, c_int, c_double, c_void_p, c_size_t, c_bool

        self.libname = "ultrafast_crs.dll" if os.name == "nt" else "ultrafast_crs.so"
        self.libpath = os.path.join(os.path.dirname(__file__), 'cpp', self.libname)
        self.lib = cdll.LoadLibrary(self.libpath)
    
        self.lib.cpp_get_J_max.restype = c_int
        
        self.lib.cpp_get_N_lines.restype = c_size_t
        
        self.lib.cpp_next_fast_aligned_len.argtypes = [c_size_t, c_int]
        self.lib.cpp_next_fast_aligned_len.restype = c_size_t
        
        self.lib.cpp_generate_axes.argtypes = [c_double, c_double, c_size_t, c_void_p, c_void_p]
        self.lib.cpp_generate_axes.restype = c_int
        
        self.lib.cpp_copy_database.argtypes = [c_int, c_int, c_size_t, c_void_p, c_void_p,
                                            c_size_t, c_void_p, c_void_p, c_void_p, c_void_p]
        self.lib.cpp_copy_database.restype = c_int
        
        self.lib.cpp_set_database_refs.argtypes = [c_int, c_int, c_size_t, c_void_p, c_void_p,
                                            c_size_t, c_void_p, c_void_p, c_void_p, c_void_p]
        self.lib.cpp_set_database_refs.restype = c_int
        
        self.lib.cpp_calc_Gamma.argtypes = [c_double, c_double, c_void_p, c_void_p, 
                                       c_void_p, c_void_p, c_double]
        
        self.lib.cpp_calc_matrix.argtypes = [c_double, c_double, c_double, c_void_p, c_double, 
                                        c_double, c_size_t, c_double, c_double, c_size_t, 
                                        c_void_p, c_int, c_bool, c_bool]
        
        # self.lib.cpp_calc_transform.argtypes = [c_double, c_double, c_double, c_size_t, 
        #                                         c_double, c_double, c_size_t, 
        #                                         c_void_p, c_void_p, c_int, c_int, c_bool]
        
        self.lib.cpp_calc_spectrum.argtypes = [c_double, c_double, c_size_t, c_double, 
                                               c_double, c_double, c_void_p, c_void_p, 
                                               c_void_p, c_void_p, c_size_t, c_double, 
                                               c_int, c_int, c_int, c_int, c_bool, c_bool ]
    
#         self.lib.cpp_calc_lines_direct.argtypes = [ c_void_p, c_double, c_void_p,
# 						                            c_void_p, c_void_p]

#         self.lib.cpp_add_lines_direct.argtypes = [c_double, c_double, c_size_t, c_void_p, c_void_p, 
#                                                   c_void_p, c_void_p, c_void_p]


        self.lib.cpp_calc_spectrum_direct.argtypes = [c_double, c_double, c_size_t, 
                        c_double, c_double, c_double, c_double, c_void_p, c_void_p, c_void_p, c_int, c_int]
			

    def next_fast_aligned_len(self, n_in):
        # We require alignment of 4 doubles (32 bytes) for SIMD AVX2
        # This aligment needs to survive a /2 for the calculation of t,
        # so it should be 8
        return self.lib.cpp_next_fast_aligned_len(n_in, 8)

    
    def generate_axes(self, w_min, dw, N_w):
        w_arr = np.zeros(N_w, dtype=np.float64)
        t_arr = np.zeros(N_w, dtype=np.float64)
        alignment_error = self.lib.cpp_generate_axes(w_min, dw, N_w, ptr(w_arr), ptr(t_arr))
        if alignment_error:
            import warnings
            warnings.warn('WARNING: generated arrays are not SIMD-aligned!')
            warnings.warn(alignment_error)
        return w_arr, t_arr


    def copy_database(self, J_min_in, J_max_in, EvJ1_in, EvJ0_in,
                    nu_in, sigma_gRmin_in, E0_in, J_clip_in, **kwargs):
        err = self.lib.cpp_copy_database(J_min_in, J_max_in, len(EvJ1_in), ptr(EvJ1_in), ptr(EvJ0_in),
                                len(nu_in), ptr(nu_in), ptr(sigma_gRmin_in), 
                                ptr(E0_in), ptr(J_clip_in) )    
        return err
    
    def set_database_refs(self, J_min_in, J_max_in, EvJ1_in, EvJ0_in,
                    nu_in, sigma_gRmin_in, E0_in, J_clip_in, **kwargs):
        err = self.lib.cpp_set_database_refs(J_min_in, J_max_in, len(EvJ1_in), ptr(EvJ1_in), ptr(EvJ0_in),
                                len(nu_in), ptr(nu_in), ptr(sigma_gRmin_in), 
                                ptr(E0_in), ptr(J_clip_in) )
        return err
    
    def calc_Gamma(self, p, T, params, T0=296.0):
        from ctypes import c_double, byref
        J_max = self.lib.cpp_get_J_max()
        if J_max < 0:
            print('Database not set! First set the database with set_database(set_cpp=True)...')
            return
        Gamma_RPA = np.zeros(5*(J_max+1), dtype=np.float64)
        G_min = c_double()
        G_max = c_double()
        self.lib.cpp_calc_Gamma(p, T, ptr(params), ptr(Gamma_RPA), byref(G_min), byref(G_max), T0)
        return Gamma_RPA, G_min.value, G_max.value
    
    
    def calc_matrix(self, p, T, tau, Gamma_RPA, w_min, dw, N_w, G_min, dG, N_G,
                                 chunksize=1024*128, envelope_corr=True, simd=True, **kwargs):
        W_kl_arr = db.zeros_aligned((N_G, N_w), dtype=np.complex128, byte_align=REQUIRED_BYTE_ALIGNMENT, shape_warning=True)
        self.lib.cpp_calc_matrix(p, T, tau, ptr(Gamma_RPA), w_min, dw, N_w, 
                            G_min, dG, N_G, ptr(W_kl_arr), chunksize, 
                            envelope_corr, simd)
        return W_kl_arr
    
    # TODO: match arguments wiht CPP
    # def calc_transform(self, tau, w_min, dw, G_min, dG, W_kl, E_probe, 
    #                         domain='w', FT_workers=0, simd=True, **kwargs):
    #     N_G, N_w = W_kl.shape
    #     x_arr = np.zeros(N_w, dtype=np.float64)
    #     I_arr = np.zeros(N_w, dtype=np.float64)
    #     self.lib.cpp_calc_transform(tau, w_min, dw, N_w, G_min, dG, N_G, ptr(W_kl), 
    #                             ptr(E_probe), ptr(x_arr), ptr(I_arr), 
    #                             domain=='t', FT_workers, simd)
    #     return x_arr, I_arr

    def calc_spectrum(self, w_min, dw, N_w, p, T, tau, E_probe, params, N_G=2, eps=1e-4, 
                      algo='ufa', domain='w', chunksize=1024*16, FT_workers=0, implementation='simd', envelope_corr=True, **kwargs):

        x_arr = np.zeros(N_w, dtype=np.float64)
        I_arr = np.zeros(N_w, dtype=(np.complex128 if domain=='chi' else np.float64))
        dom_in = {'w':0,  't':1, 'chi':2}[domain]
        t0 = perf_counter()
        self.lib.cpp_calc_spectrum(w_min, dw, N_w, p, T, tau, ptr(E_probe), ptr(params), ptr(x_arr), ptr(I_arr),
                                   N_G, eps, algo=='ref', dom_in, chunksize, FT_workers, envelope_corr, implementation=='simd')
        # print(err)
        
        t1 = perf_counter()
        times = {'total': (t1-t0)*1e3}
        
        return x_arr, I_arr, times


    # TODO: match arguments wiht CPP
    # def calc_lines_direct(self, Gamma_RPA, T):
        
    #     N_lines = self.lib.cpp_get_N_lines()
    #     wi_arr = np.zeros(N_lines, dtype=np.float64)
    #     Gi_arr = np.zeros(N_lines, dtype=np.float64)
    #     Wi_arr = np.zeros(N_lines, dtype=np.float64)
        
    #     self.lib.cpp_calc_lines_direct(ptr(Gamma_RPA), T, ptr(wi_arr), ptr(Gi_arr), ptr(Wi_arr))
        
    #     return wi_arr, Gi_arr, Wi_arr

    # TODO: match arguments wiht CPP
    # def add_lines_direct(self, a, tau, w_arr, wi_arr, Gi_arr, Wi_arr):
        
    #     N_w = len(w_arr)
    #     E_CRS = np.zeros(N_w, dtype=np.complex128)
        
    #     self.lib.cpp_add_lines_direct(a,tau,N_w,ptr(w_arr), 
    #                     ptr(wi_arr), ptr(Gi_arr), ptr(Wi_arr), ptr(E_CRS))
        
    #     return E_CRS

    def calc_spectrum_direct(self, w_min, dw, N_w, p, T, tau, dt_FWHM, params, 
                          domain='w', FT_workers=8, **kwargs):
        x_arr = np.zeros(N_w, dtype=np.float64)
        I_arr = np.zeros(N_w, dtype=np.float64)
        
        t0 = perf_counter()
        self.lib.cpp_calc_spectrum_direct(w_min, dw, N_w, p, T, tau, dt_FWHM, 
            ptr(params), ptr(x_arr), ptr(I_arr), domain=='t',FT_workers)

        t1 = perf_counter()
        times = {'total': (t1-t0)*1e3}
        return x_arr, I_arr, times



class PythonFunctions:
    def __init__(self):
        pass
        
    def next_fast_aligned_len(self, n_in, elem_align=4):
        n = next_fast_len(n_in)
        while(n % elem_align):
            n = next_fast_len(n + 1)
        return n
    
    def generate_axes(self, w_min, dw, N_w):
        dt = 2*pi/(N_w*dw) #s
        t_max = 0.5*N_w*dt
        t_arr = fftfreq(N_w, d=1/(2*t_max)) #s
        w_arr = w_min + np.arange(N_w)*dw
        return w_arr, t_arr
    
    
    def calc_Gamma(self, p, T,
                   params,
                   T0=296.0):  
        
        global db
        
        a, alpha, beta, delta, n = params
        Gamma_Q = np.zeros((db.J_max + 1), dtype=np.float64)
        J_arr = np.arange(db.J_min, db.J_max + 1)
        
        for Ji in J_arr:
            U1 = ((1+((a*db.EvJ0[Ji]) / (k_B*T*delta))) /
                  (1+((a*db.EvJ0[Ji]) / (k_B*T))))**2;
        
            for Jj in J_arr:
                dE_ij = h*c*(db.EvJ1[Jj] - db.EvJ0[Ji]);
                    
                if Jj > Ji: 
                    U2 = np.exp((-beta*dE_ij)/(k_B*T))
                    D1 = (2*Ji+1)/(2*Jj+1)
                    D2 = np.exp(dE_ij/(k_B*T))
                    
                    gamma_ji = U1*U2;    
                    Gamma_Q[Ji] += gamma_ji
                    Gamma_Q[Jj] += gamma_ji*D1*D2
                    
        Gamma_Q *= p*alpha*((T0/T)**n) * pi * c
        Gamma_RPA = np.zeros((5,db.J_max+1), dtype=np.float64)
       
        #TODO: could be done over a loop..
        Gamma_RPA[0, 2:  ] =  0.5*(Gamma_Q[2:] + Gamma_Q[:-2])
        Gamma_RPA[1, 1:  ] =  0.5*(Gamma_Q[1:] + Gamma_Q[:-1])
        Gamma_RPA[2,  :  ] =  0.5*(Gamma_Q[ :] + Gamma_Q[:  ])
        Gamma_RPA[3,  :-1] =  0.5*(Gamma_Q[1:] + Gamma_Q[:-1])
        Gamma_RPA[4,  :-2] =  0.5*(Gamma_Q[2:] + Gamma_Q[:-2])

        G_min = np.min(Gamma_Q[db.J_min:db.J_max+1])
        G_max = np.max(Gamma_Q[db.J_min:db.J_max+1])

        return Gamma_RPA.flatten(), G_min, G_max

    
    def calc_matrix(self, p, T, tau, Gamma_RPA,                      
                        w_min, dw, N_w, G_min, dG, N_G,
                        chunksize=1024*128,
                        envelope_corr=True,
                        **kwargs,
                        ):
    
        l_arr = (Gamma_RPA - G_min) / dG
        l0_arr = l_arr.astype(np.int32)
        la_G_arr = l_arr - l0_arr    
        aG1_arr = (np.exp(-la_G_arr*tau*dG) - 1) / (np.exp(-tau*dG) - 1)
    
        Bprim = np.exp(-h*c* db.E0         /(k_B*T))
        Bbis  = np.exp(-h*c*(db.E0 + db.nu)/(k_B*T))
    
        Wi = db.sigma_gRmin * np.abs(Bprim - Bbis)
        
        wi = 2*pi*c*db.nu
        ki = (wi - w_min) / dw
        k0 = np.clip(ki.astype(np.int32), 0, N_w - 2) #TODO: quick&dirty way to prevent indexing issues
        k1 = k0 + 1
        la_w = ki - k0
    
        # # Polynomial approximation of aw:
        theta = 0.5*dw*tau
        theta2 = theta**2
        if theta < 1e-3:
            A0 = -theta2/3.0
            A1 = -theta2/5.0 #already better at theta < 1e-2
            B0 =  theta * (1.0 + theta2/12.0);
            B1 = -theta2/3.0
        else:
            sqrt3 = 3**0.5
            A0 = 3*(1 - sqrt3*np.sin(theta/sqrt3)/np.sin(theta))
            A1 = 6*((theta/np.tan(theta) - 1)/A0 - 1)    
            B0 = 2*np.tan(0.5*theta)
            B1 = 4*(theta / B0 - 1)
            
        A = [0.5, 1 + A0*(A1/24 - 0.5), 0, A0*(2 - 2*A1/3), 0, 2*A0*A1]
        B = [-B0/4, 0, B0*(1 - B1/4), 0, B0*B1]
        C = ([1 - theta2/16.0, 0.0, theta2/4.0] if envelope_corr else [1.0,0.0,0.0])
        
        x = la_w - 0.5 
        x2 = x**2      
        aw1r = A[0] + (A[1] + (A[3] + A[5]*x2)*x2)*x
        aw1i = B[0] + (B[2] + B[4]*x2)*x2
        Wi *= C[0] + C[2]*x2
        
        # r_tan = 0.5 / np.tan(theta)
        # r_sin = 0.5 / np.sin(theta)
        
        # phi_i = (2*la_w - 1)*theta
        # aw1r =  r_sin*np.sin(phi_i) + 0.5 # Works
        # aw1i = -r_sin*np.cos(phi_i) + r_tan # Works
    
        aw0r = 1 - aw1r
        aw0i = -aw1i;
        
        l0 = l0_arr[db.J_clip]
        l1 = l0 + 1
        aG1 = aG1_arr[db.J_clip]
        aG1Wi = aG1*Wi
        aG0Wi = Wi - aG1Wi
    
    
        W_kl = np.zeros((N_G, N_w, 2), dtype=np.float64) # float pairs make indexing easier (as opposed to complex)
        
        np.add.at(W_kl, (l0, k0, 0), aw0r * aG0Wi)
        np.add.at(W_kl, (l0, k0, 1), aw0i * aG0Wi)
        np.add.at(W_kl, (l0, k1, 0), aw1r * aG0Wi)
        np.add.at(W_kl, (l0, k1, 1), aw1i * aG0Wi)
        
        np.add.at(W_kl, (l1, k0, 0), aw0r * aG1Wi)
        np.add.at(W_kl, (l1, k0, 1), aw0i * aG1Wi)
        np.add.at(W_kl, (l1, k1, 0), aw1r * aG1Wi)
        np.add.at(W_kl, (l1, k1, 1), aw1i * aG1Wi)
    
        W_kl = W_kl.reshape((N_G, 2*N_w)).view(np.complex128)
    
        return W_kl
    
    
    def calc_transform(self, tau, w_min, dw, G_min, dG, 
                        W_kl, E_probe,        
                        domain='w',
                        FT_workers=0,
                        **kwargs,
                        ):
     
        N_G, N_t = W_kl.shape
        N_w = N_t
        if domain=='t': tau=0.0
        if FT_workers == 0:
            FT_workers = cpu_count()
        
        w_arr, t_arr = generate_axes(w_min, dw, N_w)
        
        W_kl *= np.exp(1j*tau*w_arr)
        ifft(W_kl, axis=1, overwrite_x=True, workers=FT_workers)
        chi_CRS = np.zeros(N_t, dtype=np.complex128)
        for l in range(N_G):
            G_l = G_min + l*dG
            chi_l = W_kl[l,:] * np.exp(-G_l * (t_arr + tau)) * N_t
            chi_CRS += chi_l
        
        if domain == 't':
            chi_CRS[N_t//2:] = 0.0
            chi2_FT =  rfft(np.abs(chi_CRS)**2, workers=FT_workers)
            Epr2_FT =  rfft(np.abs(E_probe)**2, workers=FT_workers) # this one could be done in advance to save a little time
            I_PDS   = irfft(Epr2_FT * chi2_FT,  workers=FT_workers)*N_t*dw
            return t_arr, I_PDS
        
        else: #if domain == 'w':
            chi_CRS[t_arr < -tau] = 0.0
            E_CRS = chi_CRS * E_probe
            fft(E_CRS, overwrite_x=True, workers=FT_workers)
            I_CRS = np.abs(E_CRS)**2
            return w_arr, I_CRS


    
    

#%% initialize objecs:
db = Database()
cpp = DLLWrapper()
py = PythonFunctions()

#%% Public functions:
next_fast_aligned_len = cpp.next_fast_aligned_len   
generate_axes = cpp.generate_axes
set_database = db.set_data
store_database = db.store_data
# set_database = cpp.copy_database
# set_database = cpp.set_database_refs
calc_spectrum = cpp.calc_spectrum
calc_spectrum_direct = cpp.calc_spectrum_direct
calc_spectrum_ref = partial(calc_spectrum, algo='ref')
calc_Gamma = cpp.calc_Gamma
calc_Gamma_py = py.calc_Gamma


#%% Pure python functions:

def calc_spectrum_py(w_min, dw, N_w, p, T, tau, E_probe, params, N_G=2, eps=1e-4, 
                  domain='w', chunksize=1024*2, FT_workers=0, envelope_corr=True, **kwargs):
        
    times = {}
    t0 = perf_counter()
    Gamma_RPA, G_min, G_max = py.calc_Gamma(p, T, params) #TODO: Make python version
    dG = (G_max * (1.0 + eps) - G_min) / (N_G - 1)
    times['axes'] = (perf_counter() - t0)*1e3
    
    t1 = perf_counter()
    W_kl = py.calc_matrix(p, T, tau, Gamma_RPA,
                                  w_min, dw, N_w, G_min, dG, N_G,
                                  chunksize=chunksize, 
                                  envelope_corr=(envelope_corr if domain=='t' else False),
                                  )
    times['distribute'] = (perf_counter() - t1)*1e3
    
    t2 = perf_counter()
    x_arr, I_arr = py.calc_transform(tau, w_min, dw, 
                                  G_min, dG, W_kl, E_probe, 
                                  domain=domain)
    times['transform'] = (perf_counter() - t2)*1e3
    
    times['total'] = np.sum([*times.values()])
    
    # x_arr = np.arange(N_w)*dw + w_min
    # I_arr = np.cos(1e-2*x_arr/(2*pi*c))**2
    # times = {'total':0.0}
    
    return x_arr, I_arr, times


#%% Direct calculation functions:
    
def py_calc_Wi(T):
    global db
    
    Bprim = np.exp(-h*c* db.E0         /(k_B*T))
    Bbis  = np.exp(-h*c*(db.E0 + db.nu)/(k_B*T))
    return db.sigma_gRmin * np.abs(Bprim - Bbis)
    

def calc_spectrum_ref_py(w_min, dw, N_w, p, T, tau, E_probe, params, domain='w', 
                         FT_workers=8, logger_kwargs={'t':0.0, 'jiggle_px':0}, **kwargs):
    global db
    db.start_logger(**logger_kwargs)
    
    times = {}
    t0 = perf_counter()
    N_t = N_w
    w_arr, t_arr = generate_axes(w_min, dw, N_w)
    Gamma_RPA, G_min, G_max = py.calc_Gamma(p, T, params)
    times['axes'] = (perf_counter() - t0)*1e3

    t1 = perf_counter()
    wi_arr = 2*pi*c*db.nu
    Gi_arr = Gamma_RPA[db.J_clip]
    Wi_arr = py_calc_Wi(T)
    times['calc lines'] = (perf_counter() - t1)*1e3
    
    t2 = perf_counter()
    chi_arr = np.zeros(N_w, dtype=np.complex128)
    t_offset = tau if domain=='w' else 0.0
    wi0_arr = wi_arr - w_min

    # chi_arr = np.zeros(N_t//2, dtype=np.complex128)
    # t_arr0 = t_arr[:N_t//2]
    for i, (wi, Gi, Wi) in enumerate(zip(wi0_arr, Gi_arr, Wi_arr)):
        db.current_line = i
        chi_i = Wi * np.exp((1j*wi - Gi)*(t_arr + t_offset))
        chi_arr += chi_i 
    times['add lines'] = (perf_counter() - t2)*1e3
    
    t3 = perf_counter()
    if domain == 'chi':
        I_arr = np.zeros(N_t, dtype=np.complex128)
        I_arr[:N_t//2] = chi_arr[:N_t//2]
        x_arr = t_arr
    
    elif domain == 't':
        chi_arr[N_w//2:] = 0.0
        chi2_FT = rfft(np.abs(chi_arr)**2, workers=FT_workers)
        Epr2_FT = rfft(E_probe**2, workers=FT_workers)
        I_arr   = irfft(Epr2_FT * chi2_FT, workers=FT_workers) * N_t * dw
        x_arr = t_arr
        
    else: #domain == 'w':
        chi_arr[t_arr<-tau] = 0.0
        E_CRS = chi_arr * E_probe
        I_arr = np.abs(fft(E_CRS))**2
        x_arr = w_arr
    times['square'] = (perf_counter() - t3)*1e3
    
    times['total'] = np.sum([*times.values()])
    db.stop_logger()
    return x_arr, I_arr, times


def calc_spectrum_direct_py(w_min, dw, N_w, p, T, tau, dt_FWHM, params, 
                      domain='w', 
                      implementation='py',
                      FT_workers=8, **kwargs,
                      ):
    global db
    rsqrt2 = 1/np.sqrt(2.)
    
    times = {}
    t0 = perf_counter()
    w_arr, t_arr = generate_axes(w_min, dw, N_w)
    Gamma_RPA, G_min, G_max = py.calc_Gamma(p, T, params) #TODO: provide pure python alternative
    dt = 2*pi/(N_w*dw)
    E_probe_FT_func = lambda w: Gaussian_FT(w, dt_FWHM)
    times['axes'] = (perf_counter() - t0)*1e3
    
    t1 = perf_counter()
    if implementation == 'py':    
        wi_arr = 2*pi*c*db.nu
        Gi_arr = Gamma_RPA[db.J_clip]
        Wi_arr = py_calc_Wi(T)
    else:
        wi_arr, Gi_arr, Wi_arr = cpp.calc_lines_direct(Gamma_RPA, T)
    times['calc lines'] = (perf_counter() - t1)*1e3
    
    t2 = perf_counter()
    
    if implementation == 'py':
        E_CRS = np.zeros(N_w, dtype=np.complex128)
        for wi, Gi, Wi in zip(wi_arr, Gi_arr, Wi_arr):

            if domain == 'w':
                chi = Wi * np.exp((1j*wi - Gi)*tau)
                E_CRS += chi * E_probe_FT_func(w_arr - wi - 1j*Gi)
            else:
                chi_t = Wi * np.exp((1j*wi - Gi)*t_arr)
                E_CRS += chi_t
    else:    
        E_CRS = cpp.add_lines_direct(dt_FWHM, tau, w_arr, wi_arr, Gi_arr, Wi_arr) #TODO:Add t domain
    times['add lines'] = (perf_counter() - t2)*1e3
    
    t3 = perf_counter()
    if domain == 't':
        
        E_CRS[N_w//2:] = 0.0
        chi2_FT =  rfft(np.abs(E_CRS)**2, workers=FT_workers)
        Epr2_FT = E_probe_FT_func((w_arr[:N_w//2 + 1] - w_arr[0]) * rsqrt2) * rsqrt2 * (2*pi)
        I_arr   = irfft(Epr2_FT * chi2_FT,  workers=FT_workers)
        
    else: #domain == 'w':
        I_arr = np.abs(E_CRS)**2
    times['square'] = (perf_counter() - t3)*1e3
    
    times['total'] = np.sum([*times.values()])
    return (w_arr if domain=='w' else t_arr), I_arr, times


