

// Ultrafast CRS: ultrafast algorithm for synthetic CRS spectra
// Copyright (C) 2024  Dirk van den Bekerom - dcmvdbekerom@gmail.com

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.



#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#define vm_mm256_exp_pd(x) _mm256_exp_pd(x)
#define vm_mm256_sincos_pd(x, y) _mm256_sincos_pd(x, y)
// #include "vectormath/vectormath_exp.h"
// #include "vectormath/vectormath_trig.h"
// #define vm_mm256_exp_pd(x) static_cast<__m256d>(exp_d<Vec4d, 0, 0>(static_cast<Vec4d>(x)))
// #define vm_mm256_sincos_pd(x, y) static_cast<__m256d>(sincos_d<Vec4d, 3>(reinterpret_cast<Vec4d*>(x),static_cast<Vec4d>(y)))

#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
//#include <x86intrin.h>
#include "vectormath_exp.h"
#include "vectormath_trig.h"
#define vm_mm256_exp_pd(x) static_cast<__m256d>(exp_d<Vec4d, 0, 0>(static_cast<Vec4d>(x)))
#define vm_mm256_sincos_pd(x, y) static_cast<__m256d>(sincos_d<Vec4d, 3>(reinterpret_cast<Vec4d*>(x),static_cast<Vec4d>(y)))
#elif defined(__GNUC__) && defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#include <math.h>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

using namespace std;


const double c    = 29979245800.0; //cm.s-1
const double k_B  = 1.380649e-23; //J.K-1
const double h    = 6.62607015e-34; //J.s
const double pi   = 3.141592653589793;
const double sqrt3= 1.7320508075688772;

__m256d two_pi_c_vec = _mm256_set1_pd(2*pi*c); 
__m256d h_vec = _mm256_set1_pd(0.5);
__m256d factor_vec;     
__m256d w_min_vec;  	
__m256d dw_recp_vec;	
__m256d dwt_vec; 
__m256d h_dwt_vec;	
__m256d r_sin_vec;
__m256d r_tan_vec;
__m256i W_kl_addr_vec;
__m128i N_w_vec;
__m128i N_w_vec_m1;
// __m128i load_mask;
__m128i load_mask_last;

__m256d A1_vec;
__m256d A3_vec;
__m256d A5_vec;
__m256d B0_vec;
__m256d B2_vec;
__m256d B4_vec;
__m256d C0_vec;
__m256d C2_vec;

#define DBL_SIZE 8
#define EPI32_SIZE 4

inline void set_constants(double T, double tau, double w_min, double dw, int N_w, int N_lines, double* W_kl, bool envelope_corr){
	
	double dwt    = dw * tau;
	factor_vec    = _mm256_set1_pd(-h*c/(k_B*T));     
    w_min_vec     = _mm256_set1_pd(w_min);  	
	dw_recp_vec   = _mm256_set1_pd(1.0/dw);	
	W_kl_addr_vec = _mm256_set1_epi64x((unsigned long long)&W_kl[0]);
	N_w_vec       = _mm_set1_epi32(N_w);
	N_w_vec_m1    = _mm_set1_epi32(N_w-1);

	int align = N_lines % 4;
	load_mask_last =  _mm_set_epi32(0xFFFFFFFF*(!align),
									0xFFFFFFFF*(!((align+3)&2)),
									0xFFFFFFFF*(align!=1),
									0xFFFFFFFF);
					
	//dwt_vec       = _mm256_set1_pd(dwt); 
    //h_dwt_vec     = _mm256_set1_pd(0.5*dwt);	
	//r_sin_vec     = _mm256_set1_pd(0.5/sin(0.5*dwt));
    //r_tan_vec     = _mm256_set1_pd(0.5/tan(0.5*dwt));

	//W_kl_addr_vec0 = _mm256_set1_epi64x((unsigned long long)&W_kl[0]);
	//W_kl_addr_vec1 = _mm256_set1_epi64x((unsigned long long)&W_kl[0] + 2*N_w*DBL_SIZE);

	//position weights parametrization:
	double A0, A1, B0, B1;
	double theta = 0.5 * dw * tau;
	double theta2 = theta*theta;
	
	if (theta < 1e-3){	//at small theta use linear approximation to prevent divide by zero issues
        A0 = -theta2/3.0;
        A1 = -theta2/5.0; //already better at theta < 1e-2
        B0 =  theta * (1.0 + theta2/12.0);
        B1 = -theta2/3.0;
      	}
    else {
        A0 = 3*(1 - sqrt3*sin(theta/sqrt3)/sin(theta));
    	A1 = 6*((theta/tan(theta) - 1)/A0 - 1);
    	B0 = 2*tan(0.5*theta);
    	B1 = 4*(theta / B0 - 1); 
    }
    	
	A1_vec = _mm256_set1_pd(1 + A0*(A1/24.0 - 0.5));
	A3_vec = _mm256_set1_pd(A0*(2 - 2*A1/3.0));
	A5_vec = _mm256_set1_pd(2*A0*A1);
	
	B0_vec = _mm256_set1_pd(-B0/4.0);
	B2_vec = _mm256_set1_pd(B0*(1 - B1/4.0));
	B4_vec = _mm256_set1_pd(B0*B1);
	
	if (envelope_corr){ // if in time domain, correct for neighbor interference:
    	C0_vec = _mm256_set1_pd(1 - theta2/16.0);
    	C2_vec = _mm256_set1_pd(theta2/4.0);
    	}
	else{
		C0_vec = _mm256_set1_pd(1.0);
    	C2_vec = _mm256_set1_pd(0.0);
	}
	
}


inline void calc_Wi(__m256d nu_vec, __m256d E0_vec, __m256d sigma_gRmin_vec, __m256d env_corr, __m256d* Wi_vec, double* Wi_arr){
	
	__m256d temp_pd_0, temp_pd_1, Bprim_vec, Bbis_vec;

    //Bprim = exp( E0[i]*factor);
    temp_pd_0 = _mm256_mul_pd(factor_vec, E0_vec);  
	Bprim_vec = vm_mm256_exp_pd(temp_pd_0);                  

	//Bbis  = exp((E0[i] + nu[i])*factor);
	temp_pd_1 = _mm256_fmadd_pd(nu_vec, factor_vec, temp_pd_0); 
	Bbis_vec = vm_mm256_exp_pd(temp_pd_1);                       

	//Wi = sigma_gRmin[i]*(Bprim - Bbis);
	temp_pd_0 = _mm256_sub_pd(Bprim_vec, Bbis_vec);            
	*Wi_vec = _mm256_mul_pd(sigma_gRmin_vec, temp_pd_0);     
    *Wi_vec = _mm256_mul_pd(*Wi_vec, env_corr);
    // Return Wi to user (slow, disabled by default):
//    	_mm256_storeu_pd(Wi_arr, *Wi_vec); 
}



inline void calc_aw(__m256d nu_vec, __m128i* k0_vec, __m256d* aw1r_vec, __m256d* aw1i_vec, __m256d* env_corr){
 	
 	__m256d temp_pd_0, k_vec;	
 	temp_pd_0 = _mm256_mul_pd(nu_vec, two_pi_c_vec);  //wi = 2*pi*c*nu[i];
 	temp_pd_0 = _mm256_sub_pd(temp_pd_0, w_min_vec);
 	k_vec     = _mm256_mul_pd(temp_pd_0, dw_recp_vec); //k = (wi - w_min) / dw;
 	//k_vec = _mm256_fmsub_pd(temp_pd_0, dw_recp_vec, w_min_norm);        //2 ms slower
 	*k0_vec = _mm256_cvttpd_epi32(k_vec); //k0 = (int)k;
 	
 	// polynomial calculation of aw:
 	// x = la_i - 0.5 = k - k0 - 0.5;
 	__m256d temp_pd_1, temp_pd_2, x_vec, x2_vec;
 	temp_pd_0 = _mm256_cvtepi32_pd(*k0_vec);
 	temp_pd_1 = _mm256_sub_pd(k_vec, h_vec);
    x_vec = _mm256_sub_pd(temp_pd_1, temp_pd_0);
    x2_vec = _mm256_mul_pd(x_vec, x_vec);

    temp_pd_0 = _mm256_fmadd_pd(A5_vec, x2_vec, A3_vec);
    temp_pd_0 = _mm256_fmadd_pd(temp_pd_0, x2_vec, A1_vec);
    *aw1r_vec = _mm256_fmadd_pd(temp_pd_0, x_vec, h_vec);

    temp_pd_1 = _mm256_fmadd_pd(B4_vec, x2_vec, B2_vec);
    *aw1i_vec = _mm256_fmadd_pd(temp_pd_1, x2_vec, B0_vec);

    //correct for interference in time domain:
    *env_corr = _mm256_fmadd_pd(C2_vec, x2_vec, C0_vec);

//     // sincos calculation of aw
//  	//la_i = k - k0;
//  	__m256d sin_vec, cos_vec, tw_vec;
//  	temp_pd_0 = _mm256_cvtepi32_pd(*k0_vec);
//  	tw_vec = _mm256_sub_pd(k_vec, temp_pd_0);
// 
//  	temp_pd_0 = _mm256_fmsub_pd(tw_vec, dwt_vec, h_dwt_vec); //theta = 0.5*(2*la_i - 1)*dwt = tw*dwt - 0.5*dwt; 
//  	sin_vec = vm_mm256_sincos_pd(&cos_vec, temp_pd_0);
//  	*aw1r_vec = _mm256_fmadd_pd (r_sin_vec, sin_vec, h_vec);    //aw1r =  r_sin*sin(theta) + 0.5;
//  	*aw1i_vec = _mm256_fnmadd_pd(r_sin_vec, cos_vec, r_tan_vec);//aw1i = -r_sin*cos(theta) + r_tan; 

}



inline void get_aG(int* l0_arr, double* aG1_arr, __m128i v_index_vec, __m128i* l0_vec, __m256d* aG1_vec){
	
	//Gamma_k = (Gamma_JJ[J_clip + delta_J[i]] + Gamma_JJ[J_clip])
	*l0_vec = _mm_i32gather_epi32(&l0_arr[0], v_index_vec, EPI32_SIZE); //TODO: sizeof()?
	*aG1_vec = _mm256_i32gather_pd(&aG1_arr[0], v_index_vec, DBL_SIZE); //TODO: sizeof()?
}


inline void calc_addr(__m128i k0_vec, __m128i l0_vec, __m256i* addr0, __m256i* addr1){ 

	__m128i temp_i32_0, temp_i32_1;
	
	temp_i32_0 = _mm_mullo_epi32(N_w_vec, l0_vec);
	temp_i32_0 = _mm_add_epi32(temp_i32_0, k0_vec);
	temp_i32_1 = _mm_add_epi32(temp_i32_0, N_w_vec);
	
	//l0 index
	temp_i32_0 = _mm_slli_epi32(temp_i32_0, 4);
	*addr0 = _mm256_cvtepi32_epi64(temp_i32_0);
	*addr0 = _mm256_add_epi64(W_kl_addr_vec, *addr0);
	
	//l1 index (+ 2*N_w)
	temp_i32_1 = _mm_slli_epi32(temp_i32_1, 4);
	*addr1 = _mm256_cvtepi32_epi64(temp_i32_1);
	*addr1 = _mm256_add_epi64(W_kl_addr_vec, *addr1);
	
	/*
	__m128i temp_i32;
	__m256i temp_i64;
	
	//2*DBL_SIZE*(N_w*l0 + k0)
	temp_i32 = _mm_mullo_epi32(N_w_vec, l0_vec);
	temp_i32 = _mm_add_epi32(temp_i32, k0_vec);
	temp_i32 = _mm_slli_epi32(temp_i32, 4);
	temp_i64 = _mm256_cvtepi32_epi64(temp_i32);
	
	*addr0 = _mm256_add_epi64(W_kl_addr_vec0, temp_i64);
	*addr1 = _mm256_add_epi64(W_kl_addr_vec1, temp_i64);
	*/
}


template<unsigned char imm8>
inline void calc_intensity(__m256d aw1r_vec, __m256d aw1i_vec, __m256d aG1_vec, __m256d Wi_vec, 
						   __m256d* aw01ri, __m256d* aG0_Wi, __m256d* aG1_Wi){
	
	const __m256d IOIO_vec = {1.0, 0.0, 1.0, 0.0};
	__m256d temp_pd_0, temp_pd_1;
	constexpr unsigned char mask[] = {0x00, 0x55, 0xAA, 0xFF};

    temp_pd_0 = _mm256_permute4x64_pd(aw1r_vec, mask[imm8]);
    temp_pd_1 = _mm256_permute4x64_pd(aw1i_vec, mask[imm8]);
    temp_pd_1 = _mm256_blend_pd(temp_pd_0, temp_pd_1, 0x0A); //{aw1r, aw1i, aw1r, aw1i} //0b00001010
	temp_pd_0 = _mm256_sub_pd(IOIO_vec, temp_pd_1);          //{aw0r, aw0i, aw0r, aw0i}
	*aw01ri = _mm256_blend_pd(temp_pd_0, temp_pd_1, 0x0C);   //{aw0r, aw0i, aw1r, aw1i} //0b00001100
	
	const __m256d aG1_Wi_vec  = _mm256_mul_pd(Wi_vec, aG1_vec);
	const __m256d aG0_Wi_vec  = _mm256_sub_pd(Wi_vec, aG1_Wi_vec);
	*aG1_Wi = _mm256_permute4x64_pd(aG1_Wi_vec, mask[imm8]); //{aG1*Wi, aG1*Wi, aG1*Wi, aG1*Wi}
    *aG0_Wi = _mm256_permute4x64_pd(aG0_Wi_vec, mask[imm8]); //{aG0*Wi, aG0*Wi, aG0*Wi, aG0*Wi}
}


template<unsigned char imm8>
inline void read_fmadd_write(__m256i addr_vec, __m256d aw_vec, __m256d aG_vec){
	
	double* W_kl = reinterpret_cast<double*>(_mm256_extract_epi64(addr_vec, imm8));  	
    __m256d temp_pd = _mm256_loadu_pd(W_kl);
    temp_pd = _mm256_fmadd_pd(aw_vec, aG_vec, temp_pd);
    _mm256_storeu_pd(W_kl, temp_pd);
}


template<unsigned char imm8> 
inline void add_line( __m256d aw1r_vec, __m256d aw1i_vec, const __m256d aG1_vec, const __m256d Wi_vec, __m256i addr_0, __m256i addr_1){
	
	__m256d aw01ri, aG0_Wi, aG1_Wi;
	calc_intensity<imm8>(aw1r_vec, aw1i_vec, aG1_vec, Wi_vec, &aw01ri, &aG0_Wi, &aG1_Wi);
    read_fmadd_write<imm8>(addr_0, aw01ri, aG0_Wi);
	read_fmadd_write<imm8>(addr_1, aw01ri, aG1_Wi);
};


inline void calc_mask(__m128i k0_vec, __m128i load_mask, __m128i* mask){

	//calc mask
	//const __m128i min1_vec = {-1, -1, -1, -1};
	const __m128i min1_vec = _mm_set1_epi32(-1);
	__m128i temp_i32_0 = _mm_cmpgt_epi32(k0_vec, min1_vec);
	__m128i temp_i32_1 = _mm_cmplt_epi32(k0_vec, N_w_vec_m1); 
	*mask = _mm_and_si128(temp_i32_0, temp_i32_1);
	*mask = _mm_and_si128(*mask, load_mask);
}

inline void apply_mask(__m128i* k0_vec, __m128i* v_index_vec, __m256d* Wi_vec,  __m128i mask)	{
	//apply mask
	*k0_vec = _mm_and_si128(*k0_vec, mask);
	*v_index_vec = _mm_and_si128(*v_index_vec, mask);
	__m256i mask64 = _mm256_cvtepi32_epi64(mask);
	__m256i* Wi_ptr = reinterpret_cast<__m256i*>(Wi_vec);
	*Wi_ptr = _mm256_and_si256(mask64, *reinterpret_cast<__m256i*>(Wi_vec));
}


void simd_calc_matrix_fp64(double p, double T, double tau,
                           double* nu, double* sigma_gRmin, double* E0,
                           int* J_clip, int* l0_arr, double* aG1_arr,                      
                           double w_min, double dw, int N_w, int N_G,
                           int chunksize, int N_lines,
                           double* Wi_arr, double* W_kl, bool envelope_corr) {
	
	set_constants(T, tau, w_min, dw, N_w, N_lines, W_kl, envelope_corr);
	__m128i load_mask = _mm_set1_epi32(0xFFFFFFFF);
    int i_last = ((N_lines - 3)/4)*4;
	#pragma omp parallel for firstprivate(load_mask), schedule(guided, chunksize)
    for (int i=0; i < N_lines; i+=4){	
	
		__m128i k0_vec, l0_vec, mask;
		__m256i addr_0, addr_1;
		__m256d env_corr, Wi_vec, aw1r_vec, aw1i_vec, aG1_vec;
		
		const __m256d nu_vec = _mm256_loadu_pd(&nu[i]);
		calc_aw(nu_vec, &k0_vec, &aw1r_vec, &aw1i_vec, &env_corr);
		
		if (i == i_last) load_mask = load_mask_last;
		calc_mask(k0_vec, load_mask, &mask);
		if (_mm_test_all_zeros(mask, mask)) continue;
		
        const __m256d E0_vec = _mm256_loadu_pd(&E0[i]);                                          
        const __m256d sigma_gRmin_vec = _mm256_loadu_pd(&sigma_gRmin[i]);
		calc_Wi(nu_vec, E0_vec, sigma_gRmin_vec, env_corr, &Wi_vec, &Wi_arr[i]);

		__m128i v_index_vec = _mm_loadu_si128(reinterpret_cast<__m128i*>(&J_clip[i]));         
		apply_mask(&k0_vec, &v_index_vec, &Wi_vec, mask);
		get_aG(l0_arr, aG1_arr, v_index_vec, &l0_vec, &aG1_vec);
		
		calc_addr(k0_vec, l0_vec, &addr_0, &addr_1);

        add_line<0>(aw1r_vec, aw1i_vec, aG1_vec, Wi_vec, addr_0, addr_1);
        add_line<1>(aw1r_vec, aw1i_vec, aG1_vec, Wi_vec, addr_0, addr_1);
        add_line<2>(aw1r_vec, aw1i_vec, aG1_vec, Wi_vec, addr_0, addr_1);
        add_line<3>(aw1r_vec, aw1i_vec, aG1_vec, Wi_vec, addr_0, addr_1);
    }
};


inline void shift_line(double *base_addr, __m256d sin_vec, __m256d cos_vec){
    __m256d temp_pd_0, temp_pd_1;

 	temp_pd_0 = _mm256_loadu_pd(base_addr); //Sr0 Si0 Sr1 Si1
 	temp_pd_1 = _mm256_shuffle_pd(temp_pd_0, temp_pd_0, 0x05); //Si0 Sr0 Si1 Sr1 //0b00000101 

 	temp_pd_1 = _mm256_mul_pd(temp_pd_1, sin_vec);
 	temp_pd_0 = _mm256_fmaddsub_pd(temp_pd_0, cos_vec, temp_pd_1);
 	_mm256_storeu_pd(base_addr, temp_pd_0); 
}

void simd_shift_fp64(double tau,
					double w_min,
					double dw,
					size_t N_w,
					size_t N_G,
                    double* W_kl){

	size_t N_t = N_w;
	double* base_addr0 = W_kl;
	__m256d theta_vec = {w_min*tau, (w_min+2*dw)*tau, (w_min+1*dw)*tau, (w_min+3*dw)*tau}; //t0 t2 t1 t3
	__m256d dtheta_vec = _mm256_set1_pd(4*tau*dw);
	__m256d cos_vec;
	
    for (int k=0; k<N_t; k+=4){
		__m256d sin_vec = vm_mm256_sincos_pd(&cos_vec, theta_vec);
		
		__m256d sin_vec01 = _mm256_shuffle_pd(sin_vec, sin_vec, 0x00); //0b00000000
		__m256d cos_vec01 = _mm256_shuffle_pd(cos_vec, cos_vec, 0x00); //0b00000000
		__m256d sin_vec23 = _mm256_shuffle_pd(sin_vec, sin_vec, 0x0F); //0b00001111
		__m256d cos_vec23 = _mm256_shuffle_pd(cos_vec, cos_vec, 0x0F); //0b00001111
		
		double* base_addr1 = base_addr0;
		
        for (int l=0; l<N_G; l++){ 
            shift_line(base_addr1,     sin_vec01, cos_vec01);
            shift_line(base_addr1 + 4, sin_vec23, cos_vec23);
            base_addr1 += 2*N_t;	
        }
		base_addr0 += 8;
		theta_vec = _mm256_add_pd(theta_vec, dtheta_vec);
    }
}


template<unsigned char imm8> 
inline void multiply_lines(__m256d factor_vec, double* base_addr, double *E_CARS, int index){

    __m256d chi_vec, E_CARS_vec;
    __m256d temp_factor_vec = _mm256_permute4x64_pd(factor_vec, imm8); //0b01010000
//     __m256d temp_probe_vec = _mm256_permute4x64_pd(probe_vec, imm8); //0b01010000

    //chi_r = W_kl[2*N_t*l + 2*k] * factor;
    //chi_i = W_kl[2*N_t*l + 2*k+1] * factor;
    chi_vec = _mm256_loadu_pd(&base_addr[index]);
    chi_vec = _mm256_mul_pd(chi_vec, temp_factor_vec);
    
    //E_CARS[2*k  ] += chi_r * E_probe[k];
    //E_CARS[2*k+1] += chi_i * E_probe[k];
//     chi_vec = _mm256_mul_pd(chi_vec, temp_probe_vec);
    E_CARS_vec = _mm256_loadu_pd(&E_CARS[index]);
    E_CARS_vec = _mm256_add_pd(E_CARS_vec, chi_vec);
    _mm256_storeu_pd(&E_CARS[index], E_CARS_vec);
 
}

void simd_mult_fp64(double tau,
					double dt,
					size_t N_t,
					double G_min,
					double dG, 
					size_t N_G,
					double* W_kl,
					double* E_CARS){
    
    int l, k;
    double G_l;
    __m256d N_t_vec = _mm256_set1_pd(N_t);
    double* base_addr;   
    __m256d G_l_vec, factor_vec, temp_pd_0;
	__m256d tau_vec = _mm256_set1_pd(tau);
	
	__m256d dt_arr_vec = _mm256_set1_pd(4*dt);
	__m256d t_max_vec = _mm256_set1_pd(N_t*dt);
	
    for (l=0; l<N_G; l++){
        G_l = G_min + l*dG;
        G_l_vec = _mm256_set1_pd(-G_l);
        base_addr = &W_kl[2*N_t*l]; 
		
		//TODO: this should be refactored
		//TODO: make safe for 4-alignment (now 8 is required)
		__m256d t_arr_vec = _mm256_set_pd(3*dt, 2*dt, dt, 0);
		__m256d neg_t_arr_vec = _mm256_set1_pd(-double(N_t/2)*dt);
        
        for (k=0; k<N_t; k+=4){
			if (k==N_t/2) t_arr_vec = _mm256_sub_pd(t_arr_vec, t_max_vec);

            //factor = exp(-G_l * (t_arr[k] + tau)) * N_t; 
            //t_arr_vec = _mm256_loadu_pd(&t_arr[k]);
			

			
			//t_arr_vec = _mm256_add_pd(t_arr_vec, dt_arr_vec);
            temp_pd_0 = _mm256_add_pd(t_arr_vec, tau_vec);
            temp_pd_0 = _mm256_mul_pd(temp_pd_0, G_l_vec); // TODO: can be single fmadd
            temp_pd_0 = vm_mm256_exp_pd(temp_pd_0);
            factor_vec = _mm256_mul_pd(temp_pd_0, N_t_vec); //TODO: *N_t shouldn't be here
            
            multiply_lines<0x50>(factor_vec, base_addr, E_CARS, 2*k); //0b01010000
            multiply_lines<0xFA>(factor_vec, base_addr, E_CARS, 2*k+4); //0b11111010
			
			t_arr_vec = _mm256_add_pd(t_arr_vec, dt_arr_vec);

        };
    };
    
};
