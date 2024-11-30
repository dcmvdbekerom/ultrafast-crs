

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


void cpp_calc_matrix_fp64( double p,
                           double T, 
                           double tau,
                           double* nu,
                           double* sigma_gRmin,
                           double* E0,
                           int* J_clip,
                           int* l0_arr,
                           double* aG1_arr,                      
                           double w_min, 
                           double dw, 
                           int N_w,
                           int N_G,
                           int chunksize,
                           int Nlines,
                           double* Wi_arr, 
                           double* W_kl,
                           bool envelope_corr) {

    //double dwt = dw * tau;
    //double r_tan = 0.5/tan(0.5*dwt);
    //double r_sin = 0.5/sin(0.5*dwt);
    
    unsigned long long base_addr = (unsigned long long)&W_kl[0];
//     unsigned long long addr0, addr1;


    double theta = 0.5*dw*tau;
    double theta2 = theta*theta;
	double A0, A1, B0, B1;
    if (theta < 1e-3){
        A0 = -theta2/3.0;
        A1 = -theta2/5.0; //already better at theta < 1e-2
        B0 =  theta * (1.0 + theta2/12.0);
        B1 = -theta2/3.0;
        }
    else{
        A0 = 3*(1 - sqrt3*sin(theta/sqrt3)/sin(theta));
        A1 = 6*((theta/tan(theta) - 1)/A0 - 1);
        B0 = 2*tan(0.5*theta);
        B1 = 4*(theta / B0 - 1);
        }
        
    double A[6] = {0.5, 1 + A0*(A1/24 - 0.5), 0, A0*(2 - 2*A1/3), 0, 2*A0*A1};
    double B[5] = {-B0/4, 0, B0*(1 - B1/4), 0, B0*B1};
    double C[3] = {1.0, 0.0, 0.0};
    if (envelope_corr){
        C[0] = 1 - theta2/16.0;
        C[2] = theta2/4.0;
    }

	#pragma omp parallel for schedule(guided, chunksize)
    for (int i=0; i < Nlines; i++){

        
        double Bprim = exp(-h*c* E0[i]         /(k_B*T));
        double Bbis  = exp(-h*c*(E0[i] + nu[i])/(k_B*T));

        double Wi = sigma_gRmin[i] * abs(Bprim - Bbis);
        
        double wi = 2*pi*c*nu[i];
        double k = (wi - w_min) / dw; //wi/dw - w_min/dw; //
        int k0 = static_cast<int>(k);
        int k1 = k0 + 1;
        double tw = k - k0;       

        if ((k0 < 0) || (k1 >= N_w)) continue;
			
//         phi_i = (2*tw - 1)*0.5*dwt;
//         aw1r =  r_sin*sin(phi_i) + 0.5;
//         aw1i = -r_sin*cos(phi_i) + r_tan;

        double x = tw - 0.5; //TODO: should be lambda
        double x2 = x*x;      
        double aw1r = A[0] + (A[1] + (A[3] + A[5]*x2)*x2)*x;
        double aw1i = B[0] + (B[2] + B[4]*x2)*x2;
        Wi  *= C[0] + C[2]*x2;

        double aw0r = 1 - aw1r;
        double aw0i = -aw1i;

        int l0 = l0_arr[J_clip[i]];
        double aG1 = aG1_arr[J_clip[i]];
        double aG0 = 1 - aG1;
    
        int offset0 = 2*(l0 * N_w + k0);
        int offset1 = offset0 + 2*N_w;

//      //return Wi_arr (slow, disabled by default):
//         Wi_arr[i+0] = aw0r * aG0 * Wi;
//         Wi_arr[i+1] = aw0i * aG0 * Wi;
//         Wi_arr[i+2] = aw0r * aG1 * Wi;
//         Wi_arr[i+3] = aw0i * aG1 * Wi;
        
        W_kl[offset0 + 0] += aw0r * aG0 * Wi;
        W_kl[offset0 + 1] += aw0i * aG0 * Wi;        
        W_kl[offset0 + 2] += aw1r * aG0 * Wi;
        W_kl[offset0 + 3] += aw1i * aG0 * Wi;

        W_kl[offset1 + 0] += aw0r * aG1 * Wi;
        W_kl[offset1 + 1] += aw0i * aG1 * Wi;        
        W_kl[offset1 + 2] += aw1r * aG1 * Wi;
        W_kl[offset1 + 3] += aw1i * aG1 * Wi;

        // }
    }    
}

void cpp_shift_fp64(double tau,
                    double w_min,
                    double dw,
					size_t N_w,
					size_t N_G,
                    double* W_kl){
    
	double theta = tau*w_min;
	double dtheta = tau*dw;
    for (int k=0; k<N_w; k++){

        double sr = cos(theta);
        double si = sin(theta);

        for (int l=0; l<N_G; l++){ 
            double Sr = W_kl[2*N_w*l + 2*k  ];
            double Si = W_kl[2*N_w*l + 2*k+1];
            
            W_kl[2*N_w*l + 2*k  ] = Sr*sr - Si*si;
            W_kl[2*N_w*l + 2*k+1] = Sr*si + Si*sr;
        }
		
		theta += dtheta;
    }
}


void cpp_mult_fp64( double tau,
					double dt,
					size_t N_t,
					double G_min,
					double dG, 
					size_t N_G,
					double* W_kl,
					double* E_CARS){
    
    int l, k;
    double G_l, factor, chi_r, chi_i;
    double t_max = N_t*dt;
	
    for (l=0; l<N_G; l++){
        G_l = G_min + l*dG;
		
		double tk = 0.0;
        
        for (k=0; k<N_t; k++){
			if (k==N_t/2) tk -= t_max;
			
            factor = exp(-G_l * (tk + tau)) * N_t;
            
            chi_r = W_kl[2*N_t*l + 2*k ] * factor;
            chi_i = W_kl[2*N_t*l + 2*k+1] * factor;
            
            E_CARS[2*k  ] += chi_r;
            E_CARS[2*k+1] += chi_i;
			
			tk += dt;
        }
    }
    
}
