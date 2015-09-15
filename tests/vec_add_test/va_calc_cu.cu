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

template<typename T>
__global__ void vec_add_kernel( T * a, T * b, T * c, int size )
{
   int bx = blockIdx.x;
   int tx = threadIdx.x;
   
   int global_idx = bx * blockDim.x + tx;
   
   c[global_idx] = a[global_idx] + b[global_idx];
}

template<typename T>
__global__ void vec_add_with_check_kernel( T * a, T * b, T * c, int size )
{
   int bx = blockIdx.x;
   int tx = threadIdx.x;
   
   int global_idx = bx * blockDim.x + tx;
   
   T   a_val = a[global_idx]
     , b_val = b[global_idx];
   
   if (global_idx < size)
   {
      if (a_val > b_val)
         c[global_idx] = a[global_idx];
      else
         c[global_idx] = b[global_idx];
   }
}

template<typename T>
int va_device_mem_alloc( T ** dev_a, T ** dev_b, T ** dev_c, const T *& host_a, const T *& host_b, size_t size )
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
time_res_t vec_add( const T * a, const T * b, T *  c, int block_size, size_t size )
{
   T   * d_A = NULL
     , * d_B = NULL
     , * d_C = NULL;

   cudaError_t error;
   
   time_res_t time_res;
   
   va_warm_up();
   
   time_res.measure_start();
   
   va_device_mem_alloc<T>(&d_A, &d_B, &d_C, a, b, size);

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

   vec_add_kernel<T><<< grid, threads >>>(d_A, d_B, d_C, size);

   cudaDeviceSynchronize();

   time_res.computing_time_ = time_res.measure_finish();

   time_res.measure_start();

   error = cudaMemcpy(c, d_C, sizeof(T) * size, cudaMemcpyDeviceToHost);

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

template<typename T>
time_res_t vec_add_with_check( const T * a, const T * b, T * c, int block_size, int size )
 {
   T   * d_A = NULL
     , * d_B = NULL
     , * d_C = NULL;

   cudaError_t error;
   
   time_res_t time_res;
   
   va_warm_up();
   
   time_res.measure_start();
   
   va_device_mem_alloc<T>(&d_A, &d_B, &d_C, a, b, size);

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

   vec_add_with_check_kernel<T><<< grid, threads >>>(d_A, d_B, d_C, size);

   cudaDeviceSynchronize();

   time_res.computing_time_ = time_res.measure_finish();

   time_res.measure_start();

   // Copy result from device to host
   error = cudaMemcpy(c, d_C, sizeof(T) * size, cudaMemcpyDeviceToHost);

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

template<typename T>
time_res_t va_calc( int size, const T * a, const T * b, T * c )
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

   time_res_t duration = vec_add<T>(a, b, c, block_size, size);

   return duration;
}

template<typename T>
time_res_t va_calc_with_check( int size, const T * a, const T * b, T * c )
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

time_res_t va_calc_cu( int size, const float * a, const float * b, float * c )
{
   return va_calc<float>(size, a, b, c);
}

time_res_t va_calc_cu( int size, const double * a, const double * b, double * c )
{
   return va_calc<double>(size, a, b, c);
}

time_res_t va_calc_cu( int size, const int * a, const int * b, int * c )
{
   return va_calc<int>(size, a, b, c);
}

time_res_t va_calc_cu_wc( int size, const float * a, const float * b, float * c )
{
   return va_calc_with_check<float>(size, a, b, c);
}

time_res_t va_calc_cu_wc( int size, const double * a, const double * b, double * c )
{
   return va_calc_with_check<double>(size, a, b, c);
}

time_res_t va_calc_cu_wc( int size, const int * a, const int * b, int * c )
{
   return va_calc_with_check<int>(size, a, b, c);
}
