#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <chrono>

#include <cuda_runtime.h>

#include "../test.h"

using namespace std;

__global__ void mm_warm_up_kernel( double * a )
{
   *a = 0;
}

void mm_warm_up()
{
   double * warm_tmp;
   
   cudaMalloc((void **) &warm_tmp, sizeof(double));
   
   mm_warm_up_kernel<<< dim3(1), dim3(1) >>>(warm_tmp);
   
   cudaFree(warm_tmp);
}

template<typename T, int BLOCK_SIZE>
__global__ void mat_mul_kernel( T * C, T * A, T * B, int size)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (i < size && j < size)
   {
      T res = 0;

      for (int k = 0; k < size; ++k)\
         res += A[k + i * size] * B[j + k * size];
      
      C[i * size + j] = res;
   }
}
template<typename T>
int mm_device_mem_alloc( T ** dev_a, T ** dev_b, T ** dev_c, const T *& host_a, const T *& host_b, size_t size )
{
   cudaError error = cudaMalloc((void **)dev_a, sizeof(T) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMalloc((void **)dev_b, sizeof(T) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMalloc((void **)dev_c, sizeof(T) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMemcpy(*dev_a, host_a, sizeof(T) * size, cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMemcpy(*dev_b, host_b, sizeof(T) * size, cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }
   
   return 1;
}

template<typename T>
time_res_t matrixMultiply(const T * a, const T * b, T * c, int block_size, int size)
{
   // Allocate device memory
   T *d_A, *d_B, *d_C;

   cudaError_t error;
   
   mm_warm_up();
   
   time_res_t time_res;
   
   time_res.measure_start();
   
   mm_device_mem_alloc<T>(&d_A, &d_B, &d_C, a, b, size * size);

   time_res.mem_allocate_time_ = time_res.measure_finish();

   // Setup execution parameters
   dim3 threads(block_size, block_size);
   dim3 grid   (size / threads.x + 1, size / threads.y + 1);

   time_res.measure_start();

   if (block_size == 32)
      mat_mul_kernel<T, 32><<< grid, threads >>>(d_C, d_A, d_B, size);
   else if (block_size == 16)
      mat_mul_kernel<T, 16><<< grid, threads >>>(d_C, d_A, d_B, size);

   cudaDeviceSynchronize();

   time_res.computing_time_ = time_res.measure_finish();

   time_res.measure_start();
   
   // Copy result from device to host
   error = cudaMemcpy(c, d_C, sizeof(T) * size * size, cudaMemcpyDeviceToHost);

   time_res.mem_allocate_time_ += time_res.measure_finish();

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

template<typename T>
time_res_t mm_calc( int size, const T * a, const T * b, T * c )
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

   time_res_t duration = matrixMultiply<T>(a, b, c, block_size, size);

   return duration;
}

time_res_t mm_calc_cu( int size, const double * a, const double * b, double * c )
{
   return mm_calc<double>(size, a, b, c);
}

time_res_t mm_calc_cu( int size, const float * a, const float * b, float * c )
{
   return mm_calc<float>(size, a, b, c);
}

time_res_t mm_calc_cu( int size, const int * a, const int * b, int * c )
{
   return mm_calc<int>(size, a, b, c);
}
