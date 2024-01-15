#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}

inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

/* --------------function signature for user defined helper functions---------------*/
int myGEMM1(double* A, double* B, double* C, double* alpha, double* beta, int M,
            int N, int K);

int myGEMM2(double* A, double* B, double* C, double* alpha, double* betta, int M,
            int N, int K);

int myGEMM3(double* A, double* B, double* C, double* alpha, double* betta, int M,
            int N, int K);

int myGEMM4(double* A, double* B, double* C, double* alpha, double* betta, int M,
            int N, int K);

__global__
void GEMM_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                 const double alpha, const double beta, unsigned int M, unsigned int N, unsigned int K);

__global__
void GEMM1_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                  const double alpha, const double beta, unsigned int M, unsigned int N, unsigned int K);

__global__
void GEMM2_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                  const double alpha, const double beta, unsigned int M, unsigned int N, unsigned int K);

__global__
void GEMM3_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                  const double alpha, const double beta, unsigned int M, unsigned int N, unsigned int K);

__global__
void GEMM4_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                  const double alpha, const double beta, unsigned int M, unsigned int N, unsigned int K);

void repColumn(double* b_rep, double* b, int M, int N);

__global__
void rep_column_kernel(double* b_rep, double* b, int M, int N);

void elemwiseSubtract(double* A, double* B, int M, int N);

__global__
void elemwise_subtract_kernel(double* A, double* B, int M, int N);

void elemwiseMultiply(double* A, double* B, int M, int N);

__global__
void elemwise_multiply_kernel(double* A, double* B, int M, int N);

void elemwiseMultiplyConstant(double* A, double c, int M, int N);

__global__
void elemwise_multiply_constant_kernel(double* A, double c, int M, int N);

void transpose(double* A_t, double* A, int M, int N); 

__global__
void transpose_kernel(double* A_t, double* A, int M, int N);

void fill(double* A, double val, int M, int N);

__global__
void fill_kernel(double* A, double val, int M, int N);

void sumRows(double* Asum, double* A, int M, int N);

__global__
void sum_rows_kernel(double* Asum, double* A, int M, int N);

void mySigmoid(double* z, double* a, int M, int N);

__global__
void sigmoid_kernel(double* z, double* a, int M, int N); 

void mySoftmax(double* z, double* a, int M, int N);

__global__
void softmax_kernel(double* z, double* a, int M, int N);

void myGradientDescent(double* weight, double* grad,
                       int M, int N, double learning_rate);

__global__
void gradient_descent_kernel(double* weight, double* grad,
                             int M, int N, double learning_rate);

#endif
