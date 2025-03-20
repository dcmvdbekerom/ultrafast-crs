
//64bit fp
_mm256_set1_pd();
_mm256_set_pd();

_mm256_loadu_pd();
_mm256_storeu_pd();
_mm256_i32gather_pd();

_mm256_add_pd(); 
_mm256_sub_pd();            
_mm256_mul_pd();  
_mm256_fmadd_pd();
_mm256_fmaddsub_pd(); 
vm_mm256_exp_pd();
vm_mm256_sincos_pd();

_mm256_permute4x64_pd();
_mm256_blend_pd(); 
_mm256_shuffle_pd();


//32 bit int
_mm_set1_epi32();

_mm_loadu_si128();
_mm_i32gather_epi32();

_mm_cmpgt_epi32();
_mm_cmplt_epi32(); 
_mm_slli_epi32();
_mm_add_epi32();
_mm_mullo_epi32();
_mm_and_si128();


//64bit int
_mm256_set1_epi64x();
_mm256_extract_epi64(); 
_mm256_add_epi64();


//conversions
_mm256_cvttpd_epi32();
_mm256_cvtepi32_pd();
_mm256_cvtepi32_epi64();

//misc
_mm256_and_si256();
