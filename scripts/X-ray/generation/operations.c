#include <math.h>
#include <omp.h>

// OpenMP set number of threads
void set_threads(const unsigned int nthrd){
    omp_set_num_threads(nthrd);
}

void zero(float * const a, unsigned long n){
    #pragma omp parallel for
    for(unsigned long i=0; i<n; i++){
        a[i] = 0;
    }
}


void cplusab(const float * const a, const float b, float * const c, unsigned long n){
    #pragma omp parallel for
    for(unsigned long i=0; i<n; i++){
        c[i] += a[i]*b;
    }
}

void cplusexpab(const float * const a, const float b, float * const c, unsigned long n){
    #pragma omp parallel for
    for(unsigned long i=0; i<n; i++){
        c[i] += expf(-a[i])*b;
    }
}
