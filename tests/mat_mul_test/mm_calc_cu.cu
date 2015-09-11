#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <chrono>

#include <cuda_runtime.h>

#include "../test.h"

using namespace std;

// Compute C = A * B

template <int BLOCK_SIZE>
__global__ void matrixMulCUDA(double *C, double *A, double *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    double Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
time_res_t matrixMultiply(const double * a, const double * b, double * c, int block_size, int size)
{
   // Allocate device memory
   double *d_A, *d_B, *d_C;

   cudaError_t error;
   
   time_res_t time_res;
   
   chrono::time_point<chrono::system_clock> time_start, time_finish;

   time_start = chrono::system_clock::now();

   error = cudaMalloc((void **) &d_A, sizeof(double) * size * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }

   error = cudaMalloc((void **) &d_B, sizeof(double) * size * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }

   error = cudaMalloc((void **) &d_C, sizeof(double) * size * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }

   // copy host memory to device
   error = cudaMemcpy(d_A, a, sizeof(double) * size * size, cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }

   error = cudaMemcpy(d_B, b, sizeof(double) * size * size, cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }

   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.mem_allocate_time_ = duration;

   // Setup execution parameters
   dim3 threads(block_size, block_size);
   dim3 grid(size / threads.x + 1, size / threads.y + 1);

   time_start = chrono::system_clock::now();

   // Performs warmup operation using matrixMul CUDA kernel
   if (block_size == 16)
   {
      matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, size, size);
   }
   else
   {
      matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, size, size);
   }

   cudaDeviceSynchronize();

   time_finish = chrono::system_clock::now();

   duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.computing_time_ = duration;

   time_start = chrono::system_clock::now();

   // Copy result from device to host
   error = cudaMemcpy(c, d_C, sizeof(double) * size * size, cudaMemcpyDeviceToHost);

   time_finish = chrono::system_clock::now();

   duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.mem_allocate_time_ = duration;

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   cudaDeviceReset();

   return time_res;
}

time_res_t mm_calc_cu( int size, const double * a, const double * b, double * c )
{
   int devID = 0;

   cudaError_t error;
   cudaDeviceProp deviceProp;
   error = cudaGetDevice(&devID);

   if (error != cudaSuccess)
   {
      printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
   }

   error = cudaGetDeviceProperties(&deviceProp, devID);

   if (deviceProp.computeMode == cudaComputeModeProhibited)
   {
      fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
      return time_res_t();
   }

   if (error != cudaSuccess)
   {
      printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
   }

   // Use a larger block size for Fermi and above
   int block_size = (deviceProp.major < 2) ? 16 : 32;

   time_res_t duration = matrixMultiply(a, b, c, block_size, size);

   return duration;
}
