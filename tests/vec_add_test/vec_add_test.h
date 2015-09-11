#pragma once

#include "../test.h"

extern "C"
{
   int fill_2_arrays( int size, const double * a, const double * b );
}

extern time_res_t va_calc_cu           ( int size, const double * a, const double * b, double * c );
extern time_res_t va_calc_cu_with_check( int size, const double * a, const double * b, double * c );
time_res_t        va_calc_cpp          ( int size, const double * a, const double * b, double * c );

test_units_t tests_init()
{
   test_units_t tests(new vector<test_unit_t>);
   
   test_unit_t unit_test("CPP test", va_calc_cpp, "cpp.test", "cpp", 1);
   tests->push_back(unit_test);
   
   unit_test = test_unit_t("CUDA test", va_calc_cu, "cuda.test", "cuda");
   tests->push_back(unit_test);

   unit_test = test_unit_t("CUDA with check test", va_calc_cu_with_check, "cuda_wc.test", "cuda-wc");
   tests->push_back(unit_test);

   return tests;
}

test_data_t prepare_date( size_t size )
{
   test_data_t vectors(new double*[3]);

   vectors.get()[0] = new double[size];
   vectors.get()[1] = new double[size];
   vectors.get()[2] = new double[size];
   
   fill_2_arrays(size, vectors.get()[0], vectors.get()[1]);
   
   return vectors;
}

void print_unit_test_info( size_t size )
{
   cout << "vector size: " << size << " elements, (" << sizeof(double) * size / 1048576. << " mb)" << endl;
}

size_t size_by_test_idx( size_t test_idx, size_t max_data_size, size_t measurement_cnt )
{
   static int   size_incr = max_data_size / measurement_cnt
              , size = 0;
   
   size += size_incr;

   return size;
}

int run_vec_add_test( int argc, char ** argv )
{
   try{
      test_t vec_add_test(argc, argv, "vec_add_test", tests_init(), size_by_test_idx, print_unit_test_info, prepare_date);

      vec_add_test.start();
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

time_res_t va_calc_cpp( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;

   int cur_idx;

   chrono::time_point<chrono::system_clock> time_start, time_finish;

   time_start = chrono::system_clock::now();

   for (size_t i = 0; i < size; ++i)
   {
      c[i] = a[i] + b[i];
   }
   
   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.computing_time_ = duration;
   time_res.mem_allocate_time_ = 0;
   
   return time_res;
}
