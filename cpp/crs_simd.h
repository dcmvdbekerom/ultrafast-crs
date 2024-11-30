using namespace std;

void simd_calc_matrix_fp64( double p,
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
                           bool envelope_corr);

void simd_shift_fp64(double tau,
                    double w_min,
                    double dw,
					size_t N_w,
					size_t N_G,
                    double* W_kl);


void simd_mult_fp64(double tau,
					double dt,
					size_t N_t,
					double G_min,
					double dG, 
					size_t N_G,
					double* W_kl,
					double* E_CARS);