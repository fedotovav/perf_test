#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <chrono>

#include <cuda_runtime.h>

#include <omp.h>

#include "tridiagonal_test.h"

using namespace std;

__global__ void warm_up_kernel1( double * a )
{
   *a = 0;
}

void warm_up1()
{
   double * warm_tmp;
   
   cudaMalloc((void **) &warm_tmp, sizeof(double));
   
   warm_up_kernel1<<< dim3(1), dim3(1) >>>(warm_tmp);
   
   cudaFree(warm_tmp);
}

template<typename T, int BLOCK_SIZE>
__global__ void solve_slae_kernel( int size, T * subdiag, T * diag, T * superdiag, T * right_prt, T * solution )
{
   int   begin_idx = (blockDim.x * blockIdx.x + threadIdx.x) * size
       , end_idx   = begin_idx + size;
   
   for (size_t i = begin_idx; i < end_idx; ++i)
   {
      right_prt[i] = right_prt[i];
      diag[i]      = diag[i]; 
   }
   
   T tmp;
    
   for (size_t i = begin_idx + 1; i < end_idx; ++i)
   {
      tmp = subdiag[i - 1] / diag[i - 1];
      
      diag[i]      -= superdiag[i - 1] * tmp;
      right_prt[i] -= right_prt[i - 1] * tmp;
   }
   
   solution[end_idx - 1] = right_prt[end_idx - 1] / diag[end_idx - 1];
   
   for (int i = end_idx - 2; i > begin_idx - 1; --i)
      solution[i] = (right_prt[i] - superdiag[i] * solution[i + 1]) / diag[i];
}

template<typename T>
extern int device_mem_alloc( T ** dev_subdiag, T ** dev_diag, T ** dev_superdiag, T ** dev_right_prt, T ** dev_solution
                     ,T *& host_subdiag, T *& host_diag, T *& host_superdiag, T *& host_right_prt
                     ,size_t size );

template<typename T>
void copy_slae( int size, test_data_t<T> const & data, T * new_subdiag, T * new_diag, T * new_superdiag, T * new_right_prt )
{
   size_t idx = 0;
   
   for (size_t n = 0; n < size; ++n)
   {
      new_subdiag[idx] = 0;
      
      for (size_t i = 0; i < 499; ++i, ++idx)
      {
         new_subdiag[idx + 1] = data.get()[n].sub_diag_[i];
         new_diag[idx]        = data.get()[n].diag_[i];
         new_superdiag[idx]   = data.get()[n].super_diag_[i];
         new_right_prt[idx]   = data.get()[n].right_prt_[i];
      }

      new_diag[idx]           = data.get()[n].diag_[499];
      new_right_prt[size - 1] = data.get()[n].right_prt_[499];
      new_superdiag[idx]      = 0;
   }
}

template<typename T>
time_res_t solve_slae_cpu_gpu( size_t size, test_data_t<T> & data, size_t block_size )
{
   warm_up1();
   
   cudaError_t error;
   
   time_res_t time_res;
   
   // Allocate device memory
   T * d_subdiag, * d_diag, * d_superdiag, * d_right_prt, * d_solution;

   // measure coeff
   
   time_res.measure_start();
   
   device_mem_alloc<T>(&d_subdiag, &d_diag, &d_superdiag, &d_right_prt, &d_solution
                      ,data.get()[0].sub_diag_, data.get()[0].diag_, data.get()[0].super_diag_, data.get()[0].right_prt_, 500);

   // Setup execution parameters
   dim3 threads(1);
   dim3 grid   (1);

   time_res.measure_start();

   if (block_size == 32)
      solve_slae_kernel<T, 32><<< grid, threads >>>( 500, d_subdiag, d_diag, d_superdiag, d_right_prt, d_solution );
   else if (block_size == 16)
      solve_slae_kernel<T, 16><<< grid, threads >>>( 500, d_subdiag, d_diag, d_superdiag, d_right_prt, d_solution );

   double gpu_time = time_res.measure_finish();
   
   time_res.measure_start();
   
   calc_cpp<T>(500, data.get()[1].sub_diag_, data.get()[1].diag_, data.get()[1].super_diag_, data.get()[1].right_prt_, data.get()[1].solution_);
   
   double cpu_time = time_res.measure_finish();
   
   double t1, t2;
   
   if (cpu_time > gpu_time)
      t1 = gpu_time, t2 = cpu_time;
   else
      t2 = gpu_time, t1 = cpu_time;
   
   double s1 = 500, s2 = t2 * s1 / t1;
   
   size_t   n2 = s1 * size / (s2 * (1. + s1 / s2)) - 1
          , n1 = size - n2 - 1;
      
   size_t size_cpu, size_gpu;
   
   if (cpu_time > gpu_time)
      size_cpu = n2, size_gpu = n1;
   else
      size_cpu = n1, size_gpu = n2;
   
   size_t threads_cnt = 4;

   time_res.measure_start();

   T   * new_subdiag   = new T[size_gpu * 500]
     , * new_diag      = new T[size_gpu * 500]
     , * new_superdiag = new T[size_gpu * 500]
     , * new_right_prt = new T[size_gpu * 500]
     , * new_solution  = new T[size_gpu * 500];
   
   copy_slae<T>(size_gpu, data, new_subdiag, new_diag, new_superdiag, new_right_prt);
   
   delete[] new_right_prt;
   delete[] new_superdiag;
   delete[] new_diag;
   delete[] new_subdiag;
   delete[] new_solution;

   device_mem_alloc<T>(&d_subdiag, &d_diag, &d_superdiag, &d_right_prt, &d_solution
                      ,new_subdiag, new_diag, new_superdiag, new_right_prt, size_gpu * 500);
   
   size_t   size_gpu_per_thread = size_gpu / threads_cnt
          , size_cpu_per_thread = size_cpu / threads_cnt;

   if (size_gpu_per_thread % 32 && size_gpu_per_thread > 32)
      threads = dim3(32), grid = dim3(size_gpu_per_thread / 32);
   else if (size_gpu_per_thread > 32)
      threads = dim3(32), grid = dim3(1 + size_gpu_per_thread / 32);
   else
      threads = dim3(size_gpu_per_thread), grid = dim3(1);
   
   size_t chunk = 10;

   #pragma omp parallel num_threads(threads_cnt) shared(new_diag, new_superdiag, new_subdiag, new_right_prt, new_solution, data, threads_cnt, size_cpu, size_gpu, chunk) private(threads, grid)
   {
      if (omp_get_thread_num() < threads_cnt - 1)
      {
         solve_slae_kernel<T, 32><<< grid, threads >>>( 500, d_subdiag, d_diag, d_superdiag, d_right_prt, d_solution );
         
         for (size_t i = 0; i < size_cpu_per_thread; ++i)
            calc_cpp<T>(500, data.get()[i].sub_diag_, data.get()[i].diag_, data.get()[i].super_diag_, data.get()[i].right_prt_, data.get()[i].solution_);
         
         cudaDeviceSynchronize();         
      }
      else if (omp_get_thread_num() == threads_cnt - 1)
      {
         solve_slae_kernel<T, 32><<< grid, threads >>>( 500, d_subdiag, d_diag, d_superdiag, d_right_prt, d_solution );
         
         for (size_t i = 0; i < size_cpu % (threads_cnt - 1); ++i)
            calc_cpp<T>(500, data.get()[i].sub_diag_, data.get()[i].diag_, data.get()[i].super_diag_, data.get()[i].right_prt_, data.get()[i].solution_);
         
         cudaDeviceSynchronize();         
      }
   }

   time_res.computing_time_ = time_res.measure_finish() + cpu_time + gpu_time;
   
   time_res.measure_start();
   
   // Copy result from device to host
   error = cudaMemcpy(new_solution, d_solution, sizeof(T) * size_gpu * 500, cudaMemcpyDeviceToHost);

   time_res.mem_allocate_time_ += time_res.measure_finish();

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }
   
   cudaFree(d_subdiag);
   cudaFree(d_diag);
   cudaFree(d_superdiag);
   cudaFree(d_right_prt);
   cudaFree(d_solution);

   cudaDeviceReset();

   return time_res;
}

template<typename T>
time_res_t calc_cpu_gpu( size_t size, test_data_t<T> & data )
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
   size_t block_size = (deviceProp.major < 2) ? 16 : 32;

   time_res_t duration = solve_slae_cpu_gpu<T>(size, data, block_size);

   return duration;
}

time_res_t t_calc_cpu_gpu( size_t size, test_data_t<int> & data )
{
   return calc_cpu_gpu<int>(size, data);
}

time_res_t t_calc_cpu_gpu( size_t size, test_data_t<float> & data )
{
   return calc_cpu_gpu<float>(size, data);
}

time_res_t t_calc_cpu_gpu( size_t size, test_data_t<double> & data )
{
   return calc_cpu_gpu<double>(size, data);
}
