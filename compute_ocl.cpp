#include <CL/opencl.h>

#include <iostream>
#include <fstream>
#include <streambuf>

#include <chrono>

#include "test.h"

using namespace std;

// Thread block size
#define BLOCK_SIZE 32

void matrixMulGPU(cl_uint devices_num, cl_mem h_A, double* h_B_data, size_t mem_size_B, double* h_C );

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

size_t shrRoundUp(size_t localWorkSize, size_t numItems)
{
   size_t result = localWorkSize;
   while (result < numItems)
      result += localWorkSize;
    
   return result;
}

time_res_t matrixMulGPU( cl_context gpu_context, cl_command_queue command_queue, cl_kernel kernel, const double * a, const double * b, double * c, size_t size )
{
   time_res_t time_res;

   cl_int error;

   size_t   mem_size_a = sizeof(double) * size * size
          , mem_size_b = sizeof(double) * size * size
          , mem_size_c = sizeof(double) * size * size;

   chrono::time_point<chrono::system_clock> time_start, time_finish;

   time_start = chrono::system_clock::now();

   // create OpenCL buffer pointing to the host memory
   cl_mem dev_a = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, mem_size_a, NULL, &error);
   if (error != CL_SUCCESS)
   {
      cerr << "failed create buffer for a" << endl;
      return time_res_t();
   }

   cl_mem dev_b = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, mem_size_b, NULL, &error);
   if (error != CL_SUCCESS)
   {
      cerr << "failed create buffer for b" << endl;
      return time_res_t();
   }

   cl_mem dev_c = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, mem_size_c, NULL, &error);
   if (error != CL_SUCCESS)
   {
      cerr << "failed create buffer for c" << endl;
      return time_res_t();
   }

   cl_event GPUDone;
   cl_event GPUExecution;

   // Input buffer
   // Copy only assigned rows from host to device
   error = clEnqueueWriteBuffer(command_queue, dev_a, CL_FALSE, 0, sizeof(double) * size * size, a, 0, NULL, NULL);        
   if (error != CL_SUCCESS)
   {
      cerr << "failed load data (mat a) to device" << endl;
      return time_res_t();
   }
   error = clEnqueueWriteBuffer(command_queue, dev_b, CL_FALSE, 0, sizeof(double) * size * size, b, 0, NULL, NULL);        
   if (error != CL_SUCCESS)
   {
      cerr << "failed load data (mat b) to device" << endl;
      return time_res_t();
   }

   // set the args values
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &dev_c);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &dev_a);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &dev_b);

   clSetKernelArg(kernel, 3, sizeof(double) * BLOCK_SIZE * BLOCK_SIZE, 0);
   clSetKernelArg(kernel, 4, sizeof(double) * BLOCK_SIZE * BLOCK_SIZE, 0);

   clSetKernelArg(kernel, 5, sizeof(cl_int), (void *) &size);
   clSetKernelArg(kernel, 6, sizeof(cl_int), (void *) &size);
   clSetKernelArg(kernel, 7, sizeof(cl_int), (void *) &size);

   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.mem_allocate_time_ = duration;

   // Execute Multiplication on all GPUs in parallel
   size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
   size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, size), shrRoundUp(BLOCK_SIZE, size)};
    
   // Multiplication - non-blocking execution:  launch and push to device(s)
   globalWorkSize[1] = shrRoundUp(BLOCK_SIZE, size);

   time_start = chrono::system_clock::now();

   clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, &GPUExecution);

   clFlush(command_queue);

   // sync all queues to host
   clFinish(command_queue);

   time_finish = chrono::system_clock::now();

   duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.computing_time_ = duration;

   time_start = chrono::system_clock::now();

   // Non-blocking copy of result from device to host
   error = clEnqueueReadBuffer(command_queue, dev_c, CL_FALSE, 0, sizeof(double) * size * size, c, 0, NULL, &GPUDone);
   if (error != CL_SUCCESS)
   {
      cerr << "failed read buffer for c" << endl;
      return time_res_t();
   }
   
	// CPU sync with GPU
   clWaitForEvents(1, &GPUDone);

   time_finish = chrono::system_clock::now();

   duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.mem_allocate_time_ += duration;

   // Release mem and event objects    
   clReleaseEvent(GPUExecution);
   clReleaseEvent(GPUDone);

   error |= clReleaseMemObject(dev_c);
   error |= clReleaseMemObject(dev_b);
   error |= clReleaseMemObject(dev_a);

   if (error != CL_SUCCESS)
   {
      cerr << "failed release device mem buffers" << endl;
      return time_res_t();
   }
   
   return time_res;
}

time_res_t calc_ocl( int size, const double * a, const double * b, double * c )
{
   cl_uint   platform_to_get_num = 1
           , real_platform_num   = 0;
   
   cl_platform_id platform_id = NULL;
   
   if (clGetPlatformIDs(platform_to_get_num, &platform_id, &real_platform_num))
   {
      cerr << "Failed to init OpenCL platform!" << endl;
      return time_res_t();
   }
   
   cl_uint devices_num;
   
   clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &devices_num);
   
   cl_device_id * devices_id = new cl_device_id[devices_num];
   
   clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, devices_num, devices_id, NULL);
   
   cl_int error;
   
   cl_context gpu_context = clCreateContext(NULL, devices_num, devices_id, NULL, NULL, &error);
   
   if (error != CL_SUCCESS)
   {
      cerr << "failed to create OpenCL context (error: " << error << ")!" << endl;
      return time_res_t();
   }
   
   char device_name[128];
   
   cl_device_id device_id = devices_id[0];
   
   size_t max_buff_size = 0;
   
   clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &max_buff_size, NULL);
   
   cout << "max buff size: " << max_buff_size / 1048576 << " mb" << endl;
   
   // get and print the device for this queue
   clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, device_name, NULL);
   
   // create command queue
   cl_command_queue command_queue = clCreateCommandQueue(gpu_context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
   if (error != CL_SUCCESS)
   {
       cerr << "failed create command queue (error: " << error << ")!" << endl;
       return time_res_t();
   }

   // Program Setup
   const string prog_file = "/home/anton/projects/perf_test/ocl_kernel.cl";
    
   ifstream t(prog_file);

   string source((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());
   
   size_t program_length = source.length();
   
   char * source2 = new char[source.length() + 1];
   
   for (size_t i = 0; i < source.length(); ++i)
      source2[i] = source.c_str()[i];
      
   source2[source.length()] = 0;

   // create the program
   cl_program program = clCreateProgramWithSource(gpu_context, 1, (const char **)&source2, 0, &error);

   if (error != CL_SUCCESS)
   {
      cerr << "failed to create program" << endl;
   }

   // build the program
   error = clBuildProgram(program, 1, &device_id, "-cl-fast-relaxed-math", NULL, NULL);

   if (error != CL_SUCCESS)
   {
      // write out standard error, Build Log and PTX, then return error
      char build_log[2048];
      
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 2048 * sizeof(char), build_log, NULL);
      
      cout << build_log << endl;
      
      return time_res_t();
   }
   
   delete[] source2;

   // Create Kernel
   cl_kernel kernel;
   
   kernel = clCreateKernel(program, "matrixMul", &error);

   if (error != CL_SUCCESS)
   {
      cout << "failed to create kernel" << endl;   
      return time_res_t();
   }
        
   time_res_t duration = matrixMulGPU(gpu_context, command_queue, kernel, a, b, c, size);

   error |= clReleaseKernel(kernel);
   error |= clReleaseCommandQueue(command_queue);
   error |= clReleaseProgram(program);
   error |= clReleaseContext(gpu_context);

   if(error != CL_SUCCESS)
   {
      cerr << "failure releasing OpenCL resources (error: " << error << ")" << endl;
      return time_res_t();
   }

   delete[] devices_id;

   return duration;
}