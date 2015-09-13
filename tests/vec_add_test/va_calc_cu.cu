#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <chrono>

#include <cuda_runtime.h>

#include "../test.h"

using namespace std;

__global__ void va_warm_up_kernel( double * a )
{
   *a = 0;
}

void va_warm_up()
{
   double * warm_tmp;
   
   cudaMalloc((void **) &warm_tmp, sizeof(double));
   
   va_warm_up_kernel<<< dim3(1), dim3(1) >>>(warm_tmp);
   
   cudaFree(warm_tmp);
}

__global__ void vec_add_kernel( double * a, double * b, double * c, int size )
{
   int bx = blockIdx.x;
   int tx = threadIdx.x;
   
   int global_idx = bx * blockDim.x + tx;
   
   c[global_idx] = a[global_idx] + b[global_idx];
}

__global__ void vec_add_with_check_kernel( double * a, double * b, double * c, int size )
{
   int bx = blockIdx.x;
   int tx = threadIdx.x;
   
   int global_idx = bx * blockDim.x + tx;
   
   if (global_idx < size)
      c[global_idx] = a[global_idx] + b[global_idx];
}

int va_device_mem_alloc( double ** dev_a, double ** dev_b, double ** dev_c, const double *& host_a, const double *& host_b, size_t size )
{
   cudaError error = cudaMalloc((void **)dev_a, sizeof(double) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMalloc((void **)dev_b, sizeof(double) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMalloc((void **)dev_c, sizeof(double) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMemcpy(*dev_a, host_a, sizeof(double) * size, cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMemcpy(*dev_b, host_b, sizeof(double) * size, cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }
   
   return 1;
}

time_res_t vec_add( const double * a, const double * b, double * c, int block_size, size_t size )
{
   double   * d_A = NULL
          , * d_B = NULL
          , * d_C = NULL;

   cudaError_t error;
   
   time_res_t time_res;
   
   va_warm_up();
   
   time_res.measure_start();
   
   va_device_mem_alloc(&d_A, &d_B, &d_C, a, b, size);

   time_res.mem_allocate_time_ = time_res.measure_finish();

   dim3 threads, grid;

   if (block_size < size)
   {
      threads.x = block_size;
      grid.x    = size / threads.x + 1;
   }
   else
   {
      threads.x = size;
      grid.x    = 1;
   }

   time_res.measure_start();

   vec_add_kernel<<< grid, threads >>>(d_A, d_B, d_C, size);

   cudaDeviceSynchronize();

   time_res.computing_time_ = time_res.measure_finish();

   time_res.measure_start();

   error = cudaMemcpy(c, d_C, sizeof(double) * size, cudaMemcpyDeviceToHost);

   time_res.mem_allocate_time_ += time_res.measure_finish();

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy(h_C, d_C) returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   cudaDeviceReset();

   return time_res;
}

time_res_t vec_add_with_check( const double * a, const double * b, double * c, int block_size, int size )
 {
   double   * d_A = NULL
          , * d_B = NULL
          , * d_C = NULL;

   cudaError_t error;
   
   time_res_t time_res;
   
   va_warm_up();
   
   time_res.measure_start();
   
   va_device_mem_alloc(&d_A, &d_B, &d_C, a, b, size);

   time_res.mem_allocate_time_ = time_res.measure_finish();

   dim3 threads, grid;

   if (block_size < size)
   {
      threads.x = block_size;
      grid.x    = size / threads.x + 1;
   }
   else
   {
      threads.x = size;
      grid.x    = 1;
   }

   time_res.measure_start();

   vec_add_with_check_kernel<<< grid, threads >>>(d_A, d_B, d_C, size);

   cudaDeviceSynchronize();

   time_res.computing_time_ = time_res.measure_finish();

   time_res.measure_start();

   // Copy result from device to host
   error = cudaMemcpy(c, d_C, sizeof(double) * size, cudaMemcpyDeviceToHost);

   time_res.mem_allocate_time_ += time_res.measure_finish();

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (h_C, d_C) returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   cudaDeviceReset();

   return time_res;
}

time_res_t va_calc_cu( int size, const double * a, const double * b, double * c )
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

   time_res_t duration = vec_add(a, b, c, block_size, size);

   return duration;
}

time_res_t va_calc_cu_with_check( int size, const double * a, const double * b, double * c )
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

   time_res_t duration = vec_add_with_check(a, b, c, block_size, size);

   return duration;
}
