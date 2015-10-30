#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <chrono>

#include <cuda_runtime.h>

#include "tridiagonal_test.h"

using namespace std;

__global__ void warm_up_kernel( double * a )
{
   *a = 0;
}

void warm_up()
{
   double * warm_tmp;
   
   cudaMalloc((void **) &warm_tmp, sizeof(double));
   
   warm_up_kernel<<< dim3(1), dim3(1) >>>(warm_tmp);
   
   cudaFree(warm_tmp);
}

template<typename T, int BLOCK_SIZE>
__global__ void subdiag_remove_fvr( T * subdiag, T * diag, T * superdiag, T * right_prt)
{
   size_t   start_idx = blockIdx.x * BLOCK_SIZE
          , end_idx   = (1 + blockIdx.x) * BLOCK_SIZE;
   
   T line_coeff;

   for (size_t i = start_idx + 1; i < end_idx; ++i)
   {
      line_coeff = subdiag[i - 1] / diag[i - 1];

      diag[i] -= superdiag[i - 1] * line_coeff;
      subdiag[i] = -subdiag[i - 2] * line_coeff;
      right_prt[i] -= right_prt[i - 1] * line_coeff;
   }
}

template<typename T, int BLOCK_SIZE>
__global__ void subdiag_remove_bvr( T * subdiag, T * diag, T * superdiag, T * right_prt)
{
   size_t   end_idx   = blockIdx.x * BLOCK_SIZE
          , start_idx = (1 + blockIdx.x) * BLOCK_SIZE;
   
   T line_coeff;

   for (size_t i = start_idx - 2; i > end_idx - 1; --i)
   {
      line_coeff = superdiag[i] / diag[i + 1];

      superdiag[i] = -superdiag[i + 1] * line_coeff;
      right_prt[i] -= right_prt[i + 1] * line_coeff;
   }
}

template<typename T, int BLOCK_SIZE>
__global__ void solve_submatr( T * subdiag, T * diag, T * superdiag, T * right_prt, T * solution )
{
   size_t   end_idx   = blockIdx.x * BLOCK_SIZE
          , start_idx = (1 + blockIdx.x) * BLOCK_SIZE;
   
   T   left_val  = solution[start_idx - 1]
     , right_val = solution[end_idx + 1];
   
   for (size_t i = start_idx + 1; i < end_idx; ++i)
      solution[i] = (right_prt[i] - (subdiag[i] * left_val + superdiag[i] * right_val)) / diag[i];
}

template<typename T>
int device_mem_alloc( T ** dev_subdiag, T ** dev_diag, T ** dev_superdiag, T ** dev_right_prt, T ** dev_solution
                     ,T *& host_subdiag, T *& host_diag, T *& host_superdiag, T *& host_right_prt
                     ,size_t size )
{
   cudaError error = cudaMalloc((void **)dev_subdiag, sizeof(T) * (size - 1));

   if (error != cudaSuccess)
   {
      printf("cudaMalloc device subdiag returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMalloc((void **)dev_diag, sizeof(T) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc devaice diag returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMalloc((void **)dev_superdiag, sizeof(T) * (size - 1));

   if (error != cudaSuccess)
   {
      printf("cudaMalloc device superdiag returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMalloc((void **)dev_right_prt, sizeof(T) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc device right_prt returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMalloc((void **)dev_solution, sizeof(T) * size);

   if (error != cudaSuccess)
   {
      printf("cudaMalloc device solution returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMemcpy(*dev_subdiag, host_subdiag, sizeof(T) * (size - 1), cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMemcpy(*dev_diag, host_diag, sizeof(T) * size, cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMemcpy(*dev_superdiag, host_superdiag, sizeof(T) * (size - 1), cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }

   error = cudaMemcpy(*dev_right_prt, host_right_prt, sizeof(T) * size, cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
      return 0;
   }
   
   return 1;
}

template<typename T>
time_res_t solve_slae( size_t size, T * subdiag, T * diag, T * superdiag, T * right_prt, T * solution, size_t block_size )
{
   // Allocate device memory
   T * d_subdiag, * d_diag, * d_superdiag, * d_right_prt, * d_solution;

   T   * new_subdiag   = new T[size - 1]
     , * new_diag      = new T[size]
     , * new_superdiag = new T[size - 1]
     , * new_right_prt = new T[size];
   
   copy_slae<T>(size, subdiag, diag, superdiag, right_prt, new_subdiag, new_diag, new_superdiag, new_right_prt);

   cudaError_t error;
   
   warm_up();
   
   time_res_t time_res;
   
   time_res.measure_start();
   
   device_mem_alloc<T>(&d_subdiag, &d_diag, &d_superdiag, &d_right_prt, &d_solution
                      ,subdiag, diag, superdiag, right_prt, size);

   time_res.mem_allocate_time_ = time_res.measure_finish();

   // Setup execution parameters
   dim3 threads(block_size);
   dim3 grid   (size / threads.x + 1);

   time_res.measure_start();

   if (block_size == 32)
      subdiag_remove_fvr<T, 32><<< grid, threads >>>( d_subdiag, d_diag, d_superdiag, d_right_prt);
   else if (block_size == 16)
      subdiag_remove_fvr<T, 16><<< grid, threads >>>( d_subdiag, d_diag, d_superdiag, d_right_prt);

   if (block_size == 32)
      subdiag_remove_bvr<T, 32><<< grid, threads >>>( d_subdiag, d_diag, d_superdiag, d_right_prt);
   else if (block_size == 16)
      subdiag_remove_bvr<T, 16><<< grid, threads >>>( d_subdiag, d_diag, d_superdiag, d_right_prt);

   cudaDeviceSynchronize();

   size_t sup_matr_size = 2 * grid.x;
   
//   T   * sup_subdiag   = new T[sup_matr_size - 1]
//     , * sup_diag      = new T[sup_matr_size]
//     , * sup_superdiag = new T[sup_matr_size - 1]
//     , * sup_right_prt = new T[sup_matr_size]
//     , * sup_solution  = new T[sup_matr_size];
//
   create_support_slae(size, grid.x, subdiag, diag, superdiag, right_prt, subdiag, diag, superdiag, right_prt);
   
   calc_cpp_parallel(sup_matr_size, subdiag, diag, superdiag, right_prt, solution );
   
   copy_sup_solution_to_solution<T>(size, grid.x, solution, solution);
   
   if (block_size == 32)
      solve_submatr<T, 32><<< grid, threads >>>( d_subdiag, d_diag, d_superdiag, d_right_prt, d_solution );
   else if (block_size == 16)
      solve_submatr<T, 16><<< grid, threads >>>( d_subdiag, d_diag, d_superdiag, d_right_prt, d_solution );
   
   cudaDeviceSynchronize();

   time_res.computing_time_ = time_res.measure_finish();

   time_res.measure_start();
   
   // Copy result from device to host
   error = cudaMemcpy(solution, d_solution, sizeof(T) * size, cudaMemcpyDeviceToHost);

   time_res.mem_allocate_time_ += time_res.measure_finish();

   if (error != cudaSuccess)
   {
      printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
      return time_res_t();
   }
   
//   delete[] sup_solution;
//   delete[] sup_right_prt;
//   delete[] sup_superdiag;
//   delete[] sup_diag;
//   delete[] sup_subdiag;
//   
   delete[] new_right_prt;
   delete[] new_superdiag;
   delete[] new_diag;
   delete[] new_subdiag;

   cudaFree(d_subdiag);
   cudaFree(d_diag);
   cudaFree(d_superdiag);
   cudaFree(d_right_prt);
   cudaFree(d_solution);

   cudaDeviceReset();

   return time_res;
}

template<typename T>
time_res_t t_calc_gpu( size_t size, T * subdiag, T * diag, T * superdiag, T * right_prt, T * solution )
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

   time_res_t duration = solve_slae<T>( size, subdiag, diag, superdiag, right_prt, solution, block_size );

   return duration;
}

time_res_t t_calc_gpu( size_t size, test_data_t<int> & data )
{
   return t_calc_gpu<int>(size, data.get()->sub_diag_, data.get()->diag_, data.get()->super_diag_, data.get()->right_prt_, data.get()->solution_);
}

time_res_t t_calc_gpu( size_t size, test_data_t<float> & data )
{
   return t_calc_gpu<float>(size, data.get()->sub_diag_, data.get()->diag_, data.get()->super_diag_, data.get()->right_prt_, data.get()->solution_);
}

time_res_t t_calc_gpu( size_t size, test_data_t<double> & data )
{
   return t_calc_gpu<double>(size, data.get()->sub_diag_, data.get()->diag_, data.get()->super_diag_, data.get()->right_prt_, data.get()->solution_);
}
