
// This file is part of Ultrafast CRS

// Copyright (C) 2024  Dirk van den Bekerom - dcmvdbekerom@gmail.com

// Ultrafast CRS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


#ifdef _WIN32
#define LIB_API extern "C" __declspec(dllexport)
#else
#define LIB_API extern "C"
#endif

#include <complex>
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include "pocketfft/pocketfft_hdronly.h"
#include "crs_cpp.h"
#include "crs_simd.h"

#define REQUIRED_BYTE_ALIGNMENT 32
//#define MINIMAL

using namespace std;
using namespace pocketfft;


LIB_API size_t cpp_next_fast_aligned_len(size_t n, int a);
LIB_API int cpp_generate_axes(double w_min, double dw, size_t N_w, double* w_arr, double* t_arr);

LIB_API int cpp_copy_database(
				int J_min_in, 
				int J_max_in, 
				size_t N_EvJ_in,
				double* EvJ1_in, 
				double* EvJ0_in,
				size_t N_lines_in,
                double* nu_in, 
				double* sigma_gRmin_in, 
				double* E0_in, 
				int* J_clip_in);

LIB_API int cpp_set_database_refs(
				int J_min_in, 
				int J_max_in, 
				size_t N_EvJ_in,
				double* EvJ1_in, 
				double* EvJ0_in,
				size_t N_lines_in,
                double* nu_in, 
				double* sigma_gRmin_in, 
				double* E0_in, 
				int* J_clip_in
                );
#ifndef MINIMAL
LIB_API int cpp_get_J_max();
LIB_API size_t cpp_get_N_lines();
LIB_API void cpp_calc_Gamma( double p, //bar
								 double T, //K
								 double* params,
								 double* Gamma_RPA,
								 double* G_min,
								 double* G_max,
								 double T0 //K
							   );

LIB_API void cpp_calc_matrix(double p,
                               double T, 
                               double tau,
                               double* Gamma_RPA,  
                               double w_min, double dw, size_t N_w,
                               double G_min, double dG, size_t N_G,
							   complex<double>* W_kl_arr,
                               int chunksize,
                               bool envelope_corr,
                               bool simd
                               );

LIB_API void cpp_calc_transform(double tau,
								double w_min, double dw, size_t N_w,
								double G_min, double dG, size_t N_G,
								complex<double>* W_kl_arr,
								complex<double>* chi_arr,
								int domain,
								int FT_workers,
								bool simd
								);
#endif								
							
LIB_API void cpp_calc_spectrum(double w_min, 
							double dw, 
							size_t N_w, 
							double p, 
							double T, 
							double tau, 
							double* E_probe, 
							double* params, 
							double* x_arr,
							double* I_arr,
							size_t N_G, 
							double eps, 
							int algo,
							int domain, 
							int chunksize,
							int FT_workers,
							bool envelope_corr, 
							bool simd);

LIB_API void cpp_calc_lines_direct(double* Gamma_RPA, 
						  double T,
						  double* wi_arr,
						  double* Gi_arr,
						  double* Wi_arr);
						  
LIB_API void cpp_add_lines_direct(double a, 
						double tau, 
						size_t N_w,
						double* w_arr,
						double* wi_arr, 
						double* Gi_arr, 
						double* Wi_arr,
						complex<double>* E_CRS,
						int domain);

LIB_API void cpp_calc_spectrum_direct(double w_min, double dw, size_t N_w, 
						double p, double T, double tau, 
						double dt_FWHM, double* params, 
						double* x_arr, double* I_arr,
                      int domain, 
                      int FT_workers
                      );


const double pi = 3.141592653589793;
const double c  = 29979245800.0; //cm.s-1
const double k_B = 1.380649e-23; //J.K-1
const double h   = 6.62607015e-34; //J.s


///////////////////////////////
// Global variables:


int J_min = -1;
int J_max = -1; 

vector<double> EvJ1_data; 
vector<double> EvJ0_data;

vector<double> nu_data; 
vector<double> sigma_gRmin_data; 
vector<double> E0_data; 
vector<int> J_clip_data; 

size_t N_EvJ = 0;	
double* EvJ1_ptr; 
double* EvJ0_ptr;

size_t N_lines = 0;			 
double* nu_ptr; 
double* sigma_gRmin_ptr; 
double* E0_ptr; 
int* J_clip_ptr; 

bool database_initialized = false;
bool database_aligned = false;

///////////////////////////////

inline double abs_sqr(complex<double> z){
	double re = z.real();
	double im = z.imag();
	return re*re + im*im;
}


template <typename T>
void _fft(complex<T>* arr_in, complex<T>* arr_out, 
		  shape_t shape, shape_t axes, int nthreads=0, bool forward=true){

	stride_t stride(shape.size());
	size_t tmp = sizeof(complex<T>);
	for (int i=shape.size()-1; i>=0; --i)
      {
      stride[i]=tmp;
      tmp*=shape[i];
      }
	T fct{1.0L};
	if (!forward) fct = T(1.0L)/T(shape[axes[0]]);
	c2c(shape, stride, stride, axes, forward, arr_in, arr_out, fct, nthreads);
}

template<typename T>
void _ifft(complex<T>* arr_in, complex<T>* arr_out, 
		   shape_t shape, shape_t axes, int nthreads=0, bool forward=false){
	_fft(arr_in, arr_out, shape, axes, nthreads, forward);
	
}

template<typename T>
void _rfft(T* arr_r, complex<T>* arr_c, 
		  shape_t shape, shape_t axes, int nthreads=0, bool forward=true){

	stride_t stride_r(shape.size());
	stride_t stride_c(shape.size());
	size_t tmp_r = sizeof(T);
	size_t tmp_c = sizeof(complex<T>);
	
	for (int i=shape.size()-1; i>=0; --i)
      {
      stride_r[i]=tmp_r;
	  stride_c[i]=tmp_c;
      tmp_r*=shape[i];
	  if (i==(shape.size()-1)) tmp_c *= shape[i]/2 + 1;
	  else tmp_c *= shape[i];
      }
	
	if (forward){
		T fct{1.0L};
		r2c(shape, stride_r, stride_c, axes, true, arr_r, arr_c, fct, nthreads);
	}
	else {
		T fct{T(1.0L)/T(shape[axes[0]])};
		c2r(shape, stride_c, stride_r, axes, false, arr_c, arr_r, fct, nthreads);
	}
}

template<typename T>
void _irfft(complex<T>* arr_c, T* arr_r, 
		  shape_t shape, shape_t axes, int nthreads=0, bool forward=false){

	_rfft(arr_r, arr_c, shape, axes, nthreads, forward);
}

int cpp_get_J_max(){
	return J_max;
}

size_t cpp_get_N_lines(){
	return N_lines;
}

size_t cpp_next_fast_aligned_len(size_t n, int a){
	//good_size_cmplx() was modified to include alignment
	return detail::util::good_size_cmplx(n, a);
}

inline double calc_t(double dt, size_t N_t, int k){
	return (k<(N_t + 1)/2) ? k*dt : (k - int(N_t))*dt;
}

inline double calc_dt(double dw, size_t N_w){
	return 2*pi/(N_w*dw);
}

int cpp_generate_axes(double w_min, double dw, size_t N_w, double* w_arr, double* t_arr){
	
	//Generate axes:
    // double dt = 2*pi/(N_w*dw);
	double dt = calc_dt(dw, N_w);
	size_t N_t = N_w;
	for (int k=0; k<N_w; k++){
		w_arr[k] = w_min + k*dw;
		// t_arr[k] = (k<(N_w + 1)/2) ? k*dt : (k - int(N_w))*dt;
		t_arr[k] = calc_t(dt, N_t, k);
	}

	// Check for alignment:
	int err = 0;
	int element_alignment = REQUIRED_BYTE_ALIGNMENT/sizeof(double);
	if (N_w                                % element_alignment) err |= 1;
	if (reinterpret_cast<uintptr_t>(w_arr) % element_alignment) err |= 2;
	if (reinterpret_cast<uintptr_t>(t_arr) % element_alignment) err |= 4;

	return err;
}


template <typename T>
void assign_vector_aligned(vector<T> &vec, int byte_alignment, T* arr_in, T** arr_out, size_t size) {
	
	int element_alignment = byte_alignment / sizeof(T);
	vec.clear();
	vec.resize(size + 2*element_alignment, 0);
	
	int offset = 0;
	int rem = reinterpret_cast<uintptr_t>(vec.data()) % byte_alignment;
	if (rem) offset = element_alignment - rem / sizeof(T);

	*arr_out = vec.data();
	*arr_out += offset;
	memcpy(*arr_out, arr_in, size*sizeof(T));
}

int cpp_copy_database(
				int J_min_in, 
				int J_max_in, 
				size_t N_EvJ_in,
				double* EvJ1_in, 
				double* EvJ0_in,
				size_t N_lines_in,
                double* nu_in, 
				double* sigma_gRmin_in, 
				double* E0_in, 
				int* J_clip_in){
					
    J_min = J_min_in;
    J_max = J_max_in;
	
	N_EvJ = N_EvJ_in;
    EvJ1_data.assign(EvJ1_in, EvJ1_in + N_EvJ);
    EvJ0_data.assign(EvJ0_in, EvJ0_in + N_EvJ);
	
	EvJ1_ptr = EvJ1_data.data();
	EvJ0_ptr = EvJ0_data.data();
    
	int align = REQUIRED_BYTE_ALIGNMENT;
	assign_vector_aligned(nu_data, align, nu_in, &nu_ptr, N_lines_in);
	assign_vector_aligned(sigma_gRmin_data, align, sigma_gRmin_in, &sigma_gRmin_ptr, N_lines_in);
	assign_vector_aligned(E0_data, align, E0_in, &E0_ptr, N_lines_in);
	assign_vector_aligned(J_clip_data, align/2, J_clip_in, &J_clip_ptr, N_lines_in);
	
	int N_lines_new = N_lines_in;
	int elem_align = align / sizeof(double); //TODO: ..not sizeof(T)?
	int rem = N_lines_in % elem_align;
	if (rem) N_lines_new += elem_align - rem;
	for (int i=N_lines_in; i< N_lines_new; i++){
		nu_ptr[i] = nu_ptr[i-1];
		sigma_gRmin_ptr[i] = sigma_gRmin_ptr[i-1];
		E0_ptr[i] = E0_ptr[i-1];
		J_clip_ptr[i] = J_clip_ptr[i-1];
	}
	N_lines = N_lines_new;
 
	database_initialized = true;
	
	int err = 0;
	//int element_alignment = REQUIRED_BYTE_ALIGNMENT/sizeof(double);
	if (N_lines                                      % REQUIRED_BYTE_ALIGNMENT) err |= 1;
	if (reinterpret_cast<uintptr_t>(nu_ptr)          % REQUIRED_BYTE_ALIGNMENT) err |= 2;
	if (reinterpret_cast<uintptr_t>(sigma_gRmin_ptr) % REQUIRED_BYTE_ALIGNMENT) err |= 4;
	if (reinterpret_cast<uintptr_t>(E0_ptr)          % REQUIRED_BYTE_ALIGNMENT) err |= 8;
	if (reinterpret_cast<uintptr_t>(J_clip_ptr)      % REQUIRED_BYTE_ALIGNMENT/2) err |= 16;
	
	database_aligned = (err) ? false : true;
	
	return N_lines_new;
					
}


int cpp_set_database_refs(
				int J_min_in, 
				int J_max_in, 
				size_t N_EvJ_in,
				double* EvJ1_in, 
				double* EvJ0_in,
				size_t N_lines_in,
                double* nu_in, 
				double* sigma_gRmin_in, 
				double* E0_in, 
				int* J_clip_in
                ){


    J_min = J_min_in;
    J_max = J_max_in;

	N_EvJ = N_EvJ_in;
    EvJ1_ptr = EvJ1_in;
	EvJ0_ptr = EvJ0_in;

	N_lines = N_lines_in;
	nu_ptr = nu_in;
	sigma_gRmin_ptr = sigma_gRmin_in;
	E0_ptr = E0_in;
	J_clip_ptr = J_clip_in;
	
	database_initialized = true;
	
	// The error codes report on data SIMD-alignment. 
	// Since SIMD alignment is not required,
	int err = 0;
	//int element_alignment = REQUIRED_BYTE_ALIGNMENT/sizeof(double);
	if (N_lines                                      % REQUIRED_BYTE_ALIGNMENT) err |= 1;
	if (reinterpret_cast<uintptr_t>(nu_ptr)          % REQUIRED_BYTE_ALIGNMENT) err |= 2;
	if (reinterpret_cast<uintptr_t>(sigma_gRmin_ptr) % REQUIRED_BYTE_ALIGNMENT) err |= 4;
	if (reinterpret_cast<uintptr_t>(E0_ptr)          % REQUIRED_BYTE_ALIGNMENT) err |= 8;
	if (reinterpret_cast<uintptr_t>(J_clip_ptr)      % (REQUIRED_BYTE_ALIGNMENT/2)) err |= 16;
	
	database_aligned = (err) ? false : true;
	
	return err;

}


void cpp_calc_Gamma( double p, 
                     double T, 
                     double* params,
                     double* Gamma_RPA,
					 double* G_min,
					 double* G_max,
					 double T0=296.0 //K
                   ){
	
    double a     = params[0];
	double alpha = params[1];
	double beta  = params[2]; 
	double delta = params[3]; 
	double n     = params[4];
	
	double* Gamma_Q = Gamma_RPA + 2*(J_max+1);
	
    for (int Ji=J_min; Ji<J_max + 1; Ji++){
        double U1 = ((1+((a*EvJ0_ptr[Ji]) / (k_B*T*delta))) /
              (1+((a*EvJ0_ptr[Ji]) / (k_B*T))));
		U1 *= U1;
    
        for (int Jj=Ji+1; Jj <J_max+1; Jj++){
			double dE_ij = h*c*(EvJ1_ptr[Jj] - EvJ0_ptr[Ji]);        
			double U2 = exp((-beta*dE_ij)/(k_B*T));
			double D1 = (2.*Ji+1.)/(2.*Jj+1.);
			double D2 = exp(dE_ij/(k_B*T));
			
			double gamma_ji = U1*U2*p*alpha*pow(double(T0/T),n)*pi*c;   
			Gamma_Q[Ji] += gamma_ji;
			Gamma_Q[Jj] += gamma_ji*D1*D2;
		
		}
    }

    for (int delta_J=-2; delta_J<=2; delta_J++){
        if (delta_J==0) continue;
		for (int J=0; J<J_max+1; J++){
			int J_clip = min(J_max, max(J_min, J));
			Gamma_RPA[(delta_J+2)*(J_max+1) + J_clip] = 0.5*(Gamma_Q[J_clip] + 
															 Gamma_Q[J_clip + delta_J]);
		}
	}
	*G_min = Gamma_Q[J_min];
	*G_max = Gamma_Q[J_min];
	for (int J=J_min+1; J<J_max+1; J++){
		*G_min = min(*G_min, Gamma_Q[J]);
		*G_max = max(*G_max, Gamma_Q[J]);
	}
}


void cpp_calc_matrix(  double p, //TODO: p not used?
					   double T, 
					   double tau,
					   double* Gamma_RPA,  
					   double w_min, double dw, size_t N_w,
					   double G_min, double dG, size_t N_G,
					   complex<double>* W_kl_arr,
					   int chunksize=1024,
					   bool envelope_corr=false,
					   bool simd=true
					   ){
    
	vector<double> exp_Gtau;
	for (int l=0; l< N_G; l++){
		double G_l = G_min + l*dG;
		exp_Gtau.push_back(exp(-G_l*tau));
	}
	
	size_t Gamma_RPA_size = 5*(J_max+1); //TODO: this should be passed as argument
    vector<int> l0_vec;
    vector<double> aG1_vec;
	for (int i=0; i<Gamma_RPA_size; i++){
        double Gi = Gamma_RPA[i];
		if (Gi == 0.0){
			// A masked line will default to l0_vec[0],
			// so make sure there is sensible data there even when Jmin>0.
			l0_vec.push_back(0);
			aG1_vec.push_back(0.0);
			continue;
		}
        double l = int((Gi - G_min) / dG);
        int l0 = int(l);
        l0_vec.push_back(l0);
		
		//different (but equivalent) expression from paper:
        double aG = (exp(-Gi*tau) - exp_Gtau[l0]) / (exp_Gtau[l0 + 1] - exp_Gtau[l0]); 		
		aG1_vec.push_back(aG);
    }

	auto calc_matrix_fp64 = (simd) ? simd_calc_matrix_fp64 : cpp_calc_matrix_fp64;
	//auto calc_matrix_fp64 = cpp_calc_matrix_fp64;
    calc_matrix_fp64(  p, T, tau,
					   nu_ptr, sigma_gRmin_ptr, E0_ptr, J_clip_ptr,
					   l0_vec.data(), aG1_vec.data(),                      
					   w_min, dw, N_w, N_G,
					   chunksize, N_lines,
					   NULL, //&Wi_view[0],       
					   reinterpret_cast<double*>(W_kl_arr),
					   envelope_corr
					   );
}


void cpp_calc_transform(double tau,
						double w_min, double dw, size_t N_w,
                        double G_min, double dG, size_t N_G,
                        complex<double>* W_kl_arr,
						complex<double>* chi_arr,
                        int domain=0,
                        int FT_workers=0,
						bool simd=true
                        ){
							
    size_t N_t = N_w;
    if (domain == 1) tau = 0.0;        
    auto shift_fp64 = (simd) ? simd_shift_fp64 : cpp_shift_fp64;
    //auto shift_fp64 = simd_shift_fp64;
	shift_fp64(tau, w_min, dw, N_w, N_G, reinterpret_cast<double*>(W_kl_arr)); //TODO: don't do this in t-domain

	shape_t shape{N_G, N_w};
	shape_t axes{1};
    _ifft(W_kl_arr, W_kl_arr, shape, axes, FT_workers);

    auto mult_fp64 = (simd) ? simd_mult_fp64 : cpp_mult_fp64;
	//auto mult_fp64 = simd_mult_fp64;
	
	double dt = calc_dt(dw, N_w);
    mult_fp64(tau, dt, N_t, G_min, dG, N_G, 
				reinterpret_cast<double*>(W_kl_arr),
                reinterpret_cast<double*>(chi_arr));
}


inline void cpp_calc_line(double T,
						  double* Gamma_RPA, 
						  int i,
						  double* wi,
						  double* Gi,
						  double* Wi){

	*wi = 2*pi*c*nu_ptr[i];
	*Gi = Gamma_RPA[J_clip_ptr[i]];
	double Bprim = exp(-h*c* E0_ptr[i]             /(k_B*T));
	double Bbis  = exp(-h*c*(E0_ptr[i] + nu_ptr[i])/(k_B*T));
	*Wi = sigma_gRmin_ptr[i] * (Bprim - Bbis);
}


void cpp_calc_lines(double T, double* Gamma_RPA, 
					double* wi_arr, double* Gi_arr, double* Wi_arr){
	for (int i=0; i< N_lines; i++) cpp_calc_line(T, Gamma_RPA, i, &wi_arr[i], &Gi_arr[i], &Wi_arr[i]);
}


void cpp_calc_chi(double T, 
				    double tau,
					double w_min,
					size_t N_w,
					double* t_arr,
					double* Gamma_RPA,
					complex<double>* chi_arr,
					int domain=0){
	
	double wi, Gi, Wi;
	double t_offset = (domain) ? 0.0 : tau;
	size_t N_t = N_w;

    for (int i=0; i<N_lines; i++){
		cpp_calc_line(T, Gamma_RPA, i, &wi, &Gi, &Wi);
		double wi0 = wi - w_min;
		for (int k=0; k<N_t; k++){
			double tk = t_arr[k] + t_offset;
			if (tk < -t_offset) continue;
			complex<double> z_ik = {-Gi*tk, wi0*tk};
			chi_arr[k] += Wi * exp(z_ik);
		}
    }
}


void cpp_chi2Iw(complex<double>* chi_arr, 
				   double* E_probe,
				   size_t N_w, 
				   double tau,
				   double dw, //Not used
				   double* I_arr, 
				   double* t_arr, //TODO remove t_arr dependence?
				   bool fftshift=false, //TODO: remove
				   int FT_workers=0){
					  
	size_t N_t = N_w;
	complex<double>* E_CRS_arr = chi_arr; 
	for (int k=0; k<N_t; k++){
		//TODO: edge case could be handled more gracefully
		if (t_arr[k] < -tau) E_CRS_arr[k] = 0.0;
		else E_CRS_arr[k] = chi_arr[k] * E_probe[k];
	}
	shape_t shape2{N_w};
	shape_t axes2{0};
	_fft(E_CRS_arr, E_CRS_arr, shape2, axes2, FT_workers);
		
	// int offset = (fftshift) ? N_w/2 : 0;
	// for (int k=0; k<N_w/2;   k++) I_arr[k + offset] = abs_sqr(E_CRS_arr[k]);
	// for (int k=N_w/2; k<N_w; k++) I_arr[k - offset] = abs_sqr(E_CRS_arr[k]);
	for (int k=0; k<N_w; k++) I_arr[k] = abs_sqr(E_CRS_arr[k]);

}


void cpp_chi2It(complex<double>* chi_arr, 
				  double* E_probe,
				  size_t N_w, 
				  double tau,
				  double dw,
				  double* I_arr, 
				  double* t_arr, //TODO: remove t_arr dependence?
				  bool fftshift=false, //Not used
				  int FT_workers=0){
					  
	size_t N_t = N_w;

	vector<double> chi2_vec(N_t, 0.0);
	vector<double> Epr2_vec(N_t, 0.0);
	
	for (int k = 0; k<N_t; k++){
		if (t_arr[k] >= 0.0) chi2_vec[k] = abs_sqr(chi_arr[k]);
		Epr2_vec[k] = E_probe[k]*E_probe[k];
	}
	
	vector<complex<double>> chi2_FT_vec(N_t/2 + 1, 0.0);
	vector<complex<double>> Epr2_FT_vec(N_t/2 + 1, 0.0);
	
	shape_t shape2{N_w};
	shape_t axes2{0};
	_rfft(chi2_vec.data(), chi2_FT_vec.data(), shape2, axes2, FT_workers);
	_rfft(Epr2_vec.data(), Epr2_FT_vec.data(), shape2, axes2, FT_workers);

	complex<double>* I_PDS_FT = chi2_FT_vec.data(); //reuse allocated memory
	for (int k=0; k< N_t/2 + 1; k++){ //TODO: why + 1?
		I_PDS_FT[k] = chi2_FT_vec[k] * Epr2_FT_vec[k];
		I_PDS_FT[k] *= N_t*dw;
	}

	_irfft(I_PDS_FT, I_arr, shape2, axes2, FT_workers);
}

//TODO: start with p, T, tau,
void cpp_calc_spectrum(double w_min, 
					double dw, 
					size_t N_w, 
					double p, 
					double T, 
					double tau, 
					double* E_probe, 
					double* params, 
					double* x_arr,
					double* I_arr,
					size_t N_G=2, 
					double eps=1e-4, 
					int algo=0,
					int domain=0,
					int chunksize=1024,
					int FT_workers=0,
					bool envelope_corr=true, 
					bool simd=true){
    
	// generate axes
	size_t N_t = N_w;
	vector<double> w_vec(N_w, 0.0);
	vector<double> t_vec(N_t, 0.0);
	cpp_generate_axes(w_min, dw, N_w, w_vec.data(), t_vec.data());
	if (domain) copy(t_vec.begin(), t_vec.end(), x_arr);
	else 		copy(w_vec.begin(), w_vec.end(), x_arr);
    
	// calc Gamma
    vector<double> Gamma_RPA(5*(J_max+1), 0.0);
	double G_min, G_max ;
	cpp_calc_Gamma(p, T, params, Gamma_RPA.data(), &G_min, &G_max);

	//calc chi
	vector<complex<double>> chi_vec(N_t, 0.0);
	if(algo==0){ //ultrafast
		vector<complex<double>> W_kl(N_G*N_w, 0.0);
		double dG = (G_max * (1.0 + eps) - G_min) / (N_G - 1);
		cpp_calc_matrix(p, T, tau, Gamma_RPA.data(), w_min, dw, N_w, G_min, dG, N_G, 
						  W_kl.data(), chunksize=chunksize, 
						  envelope_corr=((domain) ? envelope_corr : false), simd);

		cpp_calc_transform(tau, w_min, dw, N_w, G_min, dG, N_G, W_kl.data(), 
							chi_vec.data(), domain, FT_workers, simd);
	}
	else{ // reference
		cpp_calc_chi(T, tau, w_min, N_w, t_vec.data(), Gamma_RPA.data(), chi_vec.data(), domain);		
	}
	
	//calc I
	if (domain == 2){
		for (int k=0; k<N_t/2; k++){
			I_arr[2*k    ] = chi_vec[k].real();
			I_arr[2*k + 1] = chi_vec[k].imag();		
		}
	}
	else{
		auto cpp_chi2I = (domain) ? cpp_chi2It : cpp_chi2Iw;
		cpp_chi2I(chi_vec.data(), E_probe, N_w, tau, dw, I_arr, t_vec.data(), 
					 algo, FT_workers);
	}
}



template<typename T>
T Gaussian(T t, double dt_FWHM){ //a is FWHM
    return exp(-4*log(2.0)*(t/dt_FWHM)*(t/dt_FWHM));
}

template<typename T>
T Gaussian_FT(T w, double dt_FWHM){
    return 0.5*dt_FWHM*sqrt(pi/log(2.0)) * exp(-(dt_FWHM*w)*(dt_FWHM*w)/(16*log(2.0)));
}


void cpp_add_lines_direct(double dt_FWHM,
						double T,
						double tau, 
						size_t N_w,
						double* x_arr,
						double* Gamma_RPA,
						complex<double>* E_CRS,
						int domain=0){
	double t_offset = (domain==1) ? tau : 0.0;
	double wi, Gi, Wi;
	
    for (int i=0; i<N_lines; i++){
		cpp_calc_line(T, Gamma_RPA, i, &wi, &Gi, &Wi);
		complex<double> chi0 = {0.0, 0.0};
		if (!domain){
			complex<double> z0 = {-Gi*tau, wi*tau};
			chi0 = Wi * exp(z0);
		}
		for (int k=0; k<N_w; k++){
			if (domain){
				double tk = x_arr[k] + t_offset;
				complex<double> z_ik = {-Gi*tk, wi*tk};
				E_CRS[k] += Wi * exp(z_ik);
			}
			else{
				double wk = x_arr[k];
				complex<double> z_ik = {wk - wi, -Gi};
				E_CRS[k] += chi0 * Gaussian_FT(z_ik, dt_FWHM); //TODO: pass probe function as argument
			}
		}
    }
}

//TODO: match arguments with calc_spectrum()?
//TODO: implement as calc_sprectrum(algo=2)
void cpp_calc_spectrum_direct(double w_min, double dw, size_t N_w, 
						double p, double T, double tau, 
						double dt_FWHM, double* params, 
						double* x_arr, double* I_arr,
                      int domain=0, 
                      int FT_workers=8
                      ){
    size_t N_t = N_w;
	vector<double> w_vec(N_w, 0.0);
	vector<double> t_vec(N_t, 0.0);
	cpp_generate_axes(w_min, dw, N_w, w_vec.data(), t_vec.data());
	if (domain) copy(t_vec.begin(), t_vec.end(), x_arr);
	else 		copy(w_vec.begin(), w_vec.end(), x_arr);

    vector<double> Gamma_RPA(5*(J_max+1), 0.0);
	double G_min, G_max;
	cpp_calc_Gamma(p, T, params, Gamma_RPA.data(), &G_min, &G_max);

	vector<complex<double>> E_CRS_vec(N_w, 0.0);
    cpp_add_lines_direct(dt_FWHM, T, tau, N_w,
					x_arr, Gamma_RPA.data(), 
					E_CRS_vec.data(), 2*domain);
    
    if (domain == 1){ //t
		
		//TODO: **NOT IMPLEMENTED**
		
		// vector<double> chi2_vec(N_t, 0.0);
		
		// for (int k = 0; k<N_t/2; k++){
			// chi2_vec[k] = abs_sqr(E_CRS_vec[k]);
		// }
		
        // const double rsqrt2 = 1/sqrt(2.);
		
		// vector<complex<double>> chi2_FT_vec(N_t/2 + 1, 0.0);
		// vector<complex<double>> Epr2_FT_vec(N_t/2 + 1, 0.0);
		
		// shape_t shape2{N_w};
		// shape_t axes2{0};
		// _rfft(chi2_vec.data(), chi2_FT_vec.data(), shape2, axes2, FT_workers);

		// complex<double>* I_PDS_FT = chi2_FT_vec.data(); //reuse allocated memory
		// for (int k=0; k< N_t/2 + 1; k++){ //TODO: pass probe function as argument
			// complex<double> Epr2_FT_k = Gaussian_FT((w_vec[k] - w_vec[0]) * rsqrt2, dt_FWHM) * rsqrt2; //The rsqrt2's are to simulate the square before the FT (only works for Gaussian).
			// I_PDS_FT[k] = chi2_FT_vec[k] * Epr2_FT_k;
			// I_PDS_FT[k] *= 2*pi;
		// }
		// _irfft(I_PDS_FT, I_arr, shape2, axes2, FT_workers); 
		return;
	}
	
    else{ //domain == 'w':
		for (int k=0; k<N_w; k++) I_arr[k] = abs_sqr(E_CRS_vec[k]);
    }
}