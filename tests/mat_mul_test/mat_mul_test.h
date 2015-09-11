#pragma once

#include "../test.h"

extern "C"
{
   int calc_four_thread_f( int size, const double * a, const double * b, double * c );
   int calc_two_thread_f ( int size, const double * a, const double * b, double * c );
   int calc_one_thread_f ( int size, const double * a, const double * b, double * c );

   int fill_2_arrays( int size, const double * a, const double * b );
}

extern time_res_t mm_calc_cu    ( int size, const double * a, const double * b, double * c );
extern time_res_t calc_ocl      ( int size, const double * a, const double * b, double * c );
time_res_t calc_one_thread_fort ( int size, const double * a, const double * b, double * c );
time_res_t calc_four_thread_fort( int size, const double * a, const double * b, double * c );
time_res_t calc_two_thread_fort ( int size, const double * a, const double * b, double * c );
time_res_t mm_calc_cpp          ( int size, const double * a, const double * b, double * c );

test_units_t tests_init()
{
   test_units_t tests(new vector<test_unit_t>);
   
   test_unit_t unit_test("CPP test", mm_calc_cpp, "cpp.test", "cpp", 1);
   tests->push_back(unit_test);
   
   unit_test = test_unit_t("OpenMP four thread test", calc_four_thread_fort, "omp_4t.test", "openmp-4f");
   tests->push_back(unit_test);

   unit_test = test_unit_t("OpenMP two thread test", calc_two_thread_fort, "omp_2t.test", "openmp-2f");
   tests->push_back(unit_test);

   unit_test = test_unit_t("Fortran test", calc_one_thread_fort, "f.test", "fortran");
   tests->push_back(unit_test);

   unit_test = test_unit_t("OpenCL test", calc_ocl, "cl.test", "opencl");
   tests->push_back(unit_test);

   unit_test = test_unit_t("CUDA test", mm_calc_cu, "cuda.test", "cuda");
   tests->push_back(unit_test);

   return tests;
}

test_data_t prepare_date( size_t size )
{
   test_data_t matrices(new double*[3]);

   matrices.get()[0] = new double[size * size];
   matrices.get()[1] = new double[size * size];
   matrices.get()[2] = new double[size * size];
   
   fill_2_arrays(size, matrices.get()[0], matrices.get()[1]);
   
   return matrices;
}

void print_unit_test_info( size_t size )
{
   cout << "matrix size: " << size << "x" << size << " (" << size * size << " elements, " << sizeof(double) * size * size / 1048576. << " mb)" << endl;
}

size_t size_by_test_idx( size_t test_idx, size_t max_data_size, size_t measurement_cnt )
{
   max_data_size -= max_data_size % 32;
   
   static int   size_incr = max_data_size / measurement_cnt - max_data_size / measurement_cnt % 32
              , size = 0;
   
   size += size_incr;

   return size;
}

int run_matr_mul_test( int argc, char ** argv )
{
   try{
      test_t matr_mul_test(argc, argv, "matr_mul_test", tests_init(), size_by_test_idx, print_unit_test_info, prepare_date);

      matr_mul_test.start();
   }
   catch(const po::options_description & desc)
   {
      cout << desc;
      
      return 0;
   }
   catch(const string & err)
   {
      cerr << err << endl;
      
      return 1;
   }
   catch(...)
   {
      return 1;
   }
}

time_res_t calc_one_thread_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   chrono::time_point<chrono::system_clock> time_start, time_finish;

   time_start = chrono::system_clock::now();

   calc_one_thread_f(size, a, b, c);

   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();

   time_res.computing_time_    = duration;
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_four_thread_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   chrono::time_point<chrono::system_clock> time_start, time_finish;

   time_start = chrono::system_clock::now();

   calc_four_thread_f(size, a, b, c);

   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();

   time_res.computing_time_    = duration;
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_two_thread_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   chrono::time_point<chrono::system_clock> time_start, time_finish;

   time_start = chrono::system_clock::now();

   calc_two_thread_f(size, a, b, c);

   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();

   time_res.computing_time_    = duration;
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_cpp( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;

   int cur_idx;

   chrono::time_point<chrono::system_clock> time_start, time_finish;

   time_start = chrono::system_clock::now();

   for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j)
      {
         cur_idx = i * size + j;
         
         double d = a[0];
         //c[cur_idx] = 0;

         for (int k = 0; k < size; ++k)
            c[cur_idx] += a[i * size + k] * b[k * size + j];
      }
   
   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.computing_time_ = duration;
   time_res.mem_allocate_time_ = 0;
   
   return time_res;
}
