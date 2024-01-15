#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"


__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
 Just a dummy function that can be used to warm up GPU
 */
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    return result;
}

int myGEMM(double* A, double* B, double* C,
            double* alpha, double* beta, int M, int N, int K) {
    int run_val;

    if (N > 32) {
        run_val = myGEMM4(A, B, C, alpha, beta, M, N, K);
    } else {
        run_val = myGEMM2(A, B, C, alpha, beta, M, N, K);
    }
    //run_val = myGEMM4(A, B, C, alpha, beta, M, N, K);  
    return run_val;
}

/*
 Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
 */
int myGEMM1(double* A, double* B, double* C, 
            double* alpha, double* beta, int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    int n_thread_x = 32;
    int n_thread_y = 32;
    int n_block_x = (N + n_thread_x - 1) / n_thread_x;
    int n_block_y = (M + n_thread_y - 1) / n_thread_y;
    dim3 threads(n_thread_x, n_thread_y);
    dim3 blocks(n_block_x, n_block_y);
    GEMM1_kernel<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K);
    return 0;
}

/*
 Algorithm 1, no shared memory (for part 1)
 */
__global__
void GEMM1_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                  const double alpha, const double beta,
                  unsigned int M, unsigned int N, unsigned int K) {
    // A: M x K
    // B: K x N
    // C: M x N
    int m = threadIdx.y + blockIdx.y * blockDim.y;
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    int index = n * M + m; 

    double new_val = 0.0;
    if(m < M && n < N) {
        for(int k = 0; k < K; k++){
            new_val += A[k * M + m] * B[n * K + k];
        }
        C[index] = alpha * new_val + beta * C[index];
    }
}

/*
 Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
 */
int myGEMM2(double* A, double* B, double* C,
           double* alpha, double* beta, int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    int num_thread_x = 32; // rows
    int num_thread_y = 32; // cols
    int num_block_x = (M + num_thread_x - 1) / num_thread_x;
    int num_block_y = (N + num_thread_y - 1) / num_thread_y;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    GEMM2_kernel<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K);
    return 0;
}

/*
 GPU kernel called by myGEMM. Algorithm 2
 */
__global__
void GEMM2_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, 
                 const double alpha, const double beta, 
                 unsigned int M, unsigned int N, unsigned int K) {

    __shared__ double As[32][33];
    __shared__ double Bs[32][33];
    int b_row;
    int b_col = blockIdx.y * blockDim.y + threadIdx.y;
    int a_row = blockIdx.x * blockDim.x + threadIdx.x;
    int a_col;
    int c_index = b_col * M + a_row;
    double new_val = 0.0;
    for (int k = 0; k < (K + blockDim.x - 1) / blockDim.x; k++){
        b_row = k * blockDim.x + threadIdx.x;
        a_col = k * blockDim.y + threadIdx.y;
        if (b_row < K && b_col < N) { 
            Bs[threadIdx.x][threadIdx.y] = B[b_col * K + b_row];
        } else {
            Bs[threadIdx.x][threadIdx.y] = 0.0;
        }
        if (a_row < M && a_col < K) {
            As[threadIdx.x][threadIdx.y] = A[a_col * M + a_row];
        } else {
            As[threadIdx.x][threadIdx.y] = 0.0;
        }
        __syncthreads();

        if (a_row < M && b_col < N) {
            for (int e = 0; e < 32; ++e){
                new_val += As[threadIdx.x][e] * Bs[e][threadIdx.y];
            }
        }
        __syncthreads();
    }
    if (a_row < M && b_col < N) {
        C[c_index] = beta * C[c_index] + alpha * new_val;
    }
}

/*
 Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
 */
int myGEMM3(double* A, double* B, double* C,
            double* alpha, double* beta, int M, int N, int K) {
    int num_thread_x = 16;
    int num_thread_y = 4;
    int num_thread = num_thread_x * num_thread_y; 
    // the thread block is (16 * 4)
    // sub matrix is (64 * 16) and 64's everywhere is due to that
    int num_block_x = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y = (M + num_thread - 1) / num_thread;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    GEMM3_kernel<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K);
    return 0;
}

/*
 GPU kernel called by myGEMM. "an even better implementation (4.2)"
 new one   */
__global__
void GEMM3_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, 
                  const double alpha, const double beta,
                  unsigned int M, unsigned int N, unsigned int K) {
    
    // threaidx x and y: (0 ~ 15) and (0 ~ 3)
    // blockidx x and y: (0 ~ numblock.x-1), (0 ~ numblock.y-1)
    // blockIdx.y * 64 gives the beginning of A 
    // blockIdx.x * 16 gives the beginning of B
    __shared__ double Bs[5][17];
    double a[4];

    int a_row = 64 * blockIdx.y + blockDim.x * threadIdx.y + threadIdx.x;
    int b_row;
    int a_col;
    int b_col = blockIdx.x * blockDim.x + threadIdx.x;
    int c_col;
    int c_index;
    double new_val = 0.0;
    for (int k = 0; k < (K + blockDim.y - 1) / blockDim.y; k++){
        // load shared memory
        b_row = k * blockDim.y + threadIdx.y;
        if (b_row < K && b_col < N) { 
            Bs[threadIdx.y][threadIdx.x] = B[b_col * K + b_row];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        //load a[4]
        if (a_row < M) {
            for (int i = 0; i < 4; i++) {
                a_col = k * blockDim.y + i;
                if (a_col < K) {
                    a[i] = A[a_col * M + a_row];
                } else {
                    a[i] = 0.0;
                }
            }
        }

        // update 1 * 16 block of D
        for (int i = 0; i < 16; i++) {
            c_col = blockIdx.x * blockDim.x + i;
            c_index = c_col * M + a_row;
            if (a_row < M && c_col < N) {
                new_val = 0.0;
                for (int j = 0; j < 4; j++){
                    new_val += a[j] * Bs[j][i];
                }
                if (k == 0 ) {
                    C[c_index] = beta * C[c_index];
                }
                C[c_index] += alpha * new_val;
            }
        }
        __syncthreads();
    }
}


// ---------------------------new one in progress--------------------------

/*
 Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
 */
int myGEMM4(double* A, double* B, double* C,
            double* alpha, double* beta, int M, int N, int K) {
    int num_thread_x = 32;
    int num_thread_y = 16;
    int num_thread = num_thread_x * num_thread_y; 
    // the thread block is (32 * 8)
    // sub matrix is (256 * 32) and 64's everywhere is due to that
    int num_block_x = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y = (M + num_thread - 1) / num_thread;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    GEMM4_kernel<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K);
    return 0;
}

/*
 GPU kernel called by myGEMM. "an even better implementation (4.2)"
   with a different shared memory block size    
 */
__global__
void GEMM4_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, 
                  const double alpha, const double beta,
                  unsigned int M, unsigned int N, unsigned int K) {
    
    // threaidx x and y: (0 ~ 15) and (0 ~ 3)
    // blockidx x and y: (0 ~ numblock.x-1), (0 ~ numblock.y-1)
    // blockIdx.y * 64 gives the beginning of A 
    // blockIdx.x * 16 gives the beginning of B

    int dim_a = 16;
    int dim_b = 32;
    __shared__ double Bs[17][33];
    double a[16];

    int a_row = (dim_a * dim_b) * blockIdx.y + blockDim.x * threadIdx.y + threadIdx.x;
    int b_row;
    int a_col;
    int b_col = blockIdx.x * blockDim.x + threadIdx.x;
    int c_col;
    int c_index;
    double new_val = 0.0;
    for (int k = 0; k < (K + blockDim.y - 1) / blockDim.y; k++){
        // load shared memory
        b_row = k * blockDim.y + threadIdx.y;
        if (b_row < K && b_col < N) { 
            Bs[threadIdx.y][threadIdx.x] = B[b_col * K + b_row];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        //load a[4]
        if (a_row < M) {
            for (int i = 0; i < dim_a; i++) {
                a_col = k * blockDim.y + i;
                if (a_col < K) {
                    a[i] = A[a_col * M + a_row];
                } else {
                    a[i] = 0.0;
                }
            }
        }

        // update 1 * 16 block of D
        for (int i = 0; i < dim_b; i++) {
            c_col = blockIdx.x * blockDim.x + i;
            c_index = c_col * M + a_row;
            if (a_row < M && c_col < N) {
                new_val = 0.0;
                for (int j = 0; j < dim_a; j++){
                    new_val += a[j] * Bs[j][i];
                }
                if (k == 0 ) {
                    C[c_index] = beta * C[c_index];
                }
                C[c_index] += alpha * new_val;
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------

/*
 replicate a column vector (M x 1) and make it into M X N matrix
 */
void repColumn(double* b_rep, double* b, int M, int N) {
    int num_thread_x = 32;
    int num_thread_y = 32;
    int num_block_x = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y = (M + num_thread_y - 1) / num_thread_y;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    rep_column_kernel<<<blocks,threads>>>(b_rep, b, M, N);
}

/*
 GPU kernel for repColumn
 */
__global__
void rep_column_kernel(double* b_rep, double* b, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < M && col < N){
        b_rep[col * M + row] = b[row]; 
    }
}

/*
 A - B, elemwise
 */
void elemwiseSubtract(double* A, double* B, int M, int N) {
    int num_thread_x = 32;
    int num_thread_y = 32;
    int num_block_x  = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y  = (M + num_thread_y - 1) / num_thread_y;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    elemwise_subtract_kernel<<<blocks, threads>>>(A, B, M, N);
}

/*
 GPU kernel for elemwise Subtraction
 */
__global__
void elemwise_subtract_kernel(double* A, double* B, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int index = col * M + row;
    if (row < M && col < N){
        A[index] = A[index] - B[index]; 
    }
}

/*
 A * B, elemwise
 */
void elemwiseMultiply(double* A, double* B, int M, int N) {
    int num_thread_x = 32;
    int num_thread_y = 32;
    int num_block_x  = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y  = (M + num_thread_y - 1) / num_thread_y;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    elemwise_multiply_kernel<<<blocks, threads>>>(A, B, M, N);
}

/*
 GPU kernel for elemwise multiplication
 */
__global__
void elemwise_multiply_kernel(double* A, double* B, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int index = col * M + row;
    if (row < M && col < N){
        A[index] = A[index] * B[index]; 
    }
}

/*
 c * A, elemwise
 */
void elemwiseMultiplyConstant(double* A, double c, int M, int N) {
    int num_thread_x = 32;
    int num_thread_y = 32;
    int num_block_x  = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y  = (M + num_thread_y - 1) / num_thread_y;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    elemwise_multiply_constant_kernel<<<blocks, threads>>>(A, c, M, N);
}

/*
 GPU kernel for elemwise multiplication, by a constant
 */
__global__
void elemwise_multiply_constant_kernel(double* A, double c, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int index = col * M + row;
    if (row < M && col < N){
        A[index] = c * A[index];
    }
}

/*
 transpose a matrix 
 */
void transpose(double* A_t, double* A, int M, int N) {
    int num_thread_x = 32;
    int num_thread_y = 32;
    int num_block_x  = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y  = (M + num_thread_y - 1) / num_thread_y;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    transpose_kernel<<<blocks, threads>>>(A_t, A, M, N);
}

/*
 GPU kernel for transpose 
 */
__global__
void transpose_kernel(double* A_t, double* A, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int index   = col * M + row;
    int t_index = row * N + col;
    if (row < M && col < N){
        A_t[t_index] = A[index];
    }
}

/*
 fill with a single value
 */
void fill(double* A, double val, int M, int N) {
    int num_thread_x = 32;
    int num_thread_y = 32;
    int num_block_x  = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y  = (M + num_thread_y - 1) / num_thread_y;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    fill_kernel<<<blocks, threads>>>(A, val, M, N);
}

/*
 GPU kernel for fill 
 */
__global__
void fill_kernel(double* A, double val, int M, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int index = col * M + row;
    if (row < M && col < N){
        A[index] = val;
    }
}

/*
 sum rows
 */
void sumRows(double* Asum, double* A, int M, int N) {
    int num_thread_x = 1024;
    int num_thread_y = 1;
    int num_block_x = (M + num_thread_x - 1) / num_thread_x;
    int num_block_y = 1;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    sum_rows_kernel<<<blocks,threads>>>(Asum, A, M, N);
}

/*
 GPU kernel called by sumRows
 */
__global__
void sum_rows_kernel(double* Asum, double* A, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double sum = 0.0;
        for (int col = 0; col < N; col++){
            sum += A[col * M + row];
        }
        Asum[row] = sum;
    }
}


/*
 Element-wise application of sigmoid function on a matrix with GPU.
 */
void mySigmoid(double* z, double* a, int M, int N) {
    // z: result of affine transformation
    // a: activation
    int num_thread_x = 32;
    int num_thread_y = 32;
    int num_block_x  = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y  = (M + num_thread_y - 1) / num_thread_y;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    sigmoid_kernel<<<blocks,threads>>>(z, a, M, N);
}

/*
 GPU kernel called by mySigmoid 
 */
__global__
void sigmoid_kernel(double* z, double* a, int M, int N) {
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x; 
    int index = col * M + row;

    // sigmoid
    if(row < M && col < N){
        a[index] = 1 / (1 + exp(-z[index]));
    }
}

/*
 Element-wise application of sigmoid function on a matrix with GPU.
 */
void mySoftmax(double* z, double* a, int M, int N) {
    int num_thread_x = 1024;
    int num_thread_y = 1;
    int num_block_x = (N + num_thread_x - 1) / num_thread_x;
    int num_block_y = 1;
    dim3 threads(num_thread_x, num_thread_y);
    dim3 blocks(num_block_x, num_block_y);
    softmax_kernel<<<blocks,threads>>>(z, a, M, N);
}

/*
 GPU kernel called by mySoftmax
 */
__global__
void softmax_kernel(double* z, double* a, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // fancier 2 dimensional block implementation (log n) 
    // is probably not needed since matrix dimension
    // is never too huge. 
    if (col < N) {
        double sum_exp = 0.0;
        for (int row = 0; row < M; row++){
            sum_exp += exp(z[col * M + row]);
        }
        for (int row = 0; row < M; row++){
            a[col * M + row] = exp(z[col * M + row]) / sum_exp;
        }
    }
}

/*
 Gradient descent step with GPU 
 */
void myGradientDescent(double* weight, double* grad, 
                       int M, int N, double learning_rate) {
  int num_thread_x = 32; 
  int num_thread_y = 32; 
  int num_block_x = (N + num_thread_x - 1) / num_thread_x;
  int num_block_y = (M + num_thread_y - 1) / num_thread_y;
  dim3 threads(num_thread_x, num_thread_y);
  dim3 blocks(num_block_x, num_block_y);
  gradient_descent_kernel<<<blocks,threads>>>(weight, grad, M, N, learning_rate);
}

/*
 GPU kernel called by myGradientDescent
 */
__global__
void gradient_descent_kernel(double* weight, double* grad, 
                             int M, int N, double learning_rate) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int index = col * M + row;
  if (row < M && col < N){
    weight[index] -= learning_rate * grad[index];
  }
}
