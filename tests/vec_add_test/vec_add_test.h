#pragma once

#include "../test.h"

extern "C"
{
   int fill_2_arrays( int size, const double * a, const double * b );
}

extern time_res_t va_calc_cu           ( int size, const double * a, const double * b, double * c );
extern time_res_t va_calc_cu_with_check( int size, const double * a, const double * b, double * c );
time_res_t        va_calc_cpp          ( int size, const double * a, const double * b, double * c );

class vec_add_test_t : public test_t
{
public:
   virtual size_t      size_by_measure_idx( size_t meas_idx );
   virtual void        print_measere_info ( size_t size );

   vec_add_test_t( int argc, char ** argv, const string & test_name, const test_units_t tests );
};

vec_add_test_t::vec_add_test_t( int argc, char ** argv, const string & test_name, const test_units_t tests ) :
   test_t(argc, argv, test_name, tests)
{
}

void vec_add_test_t::print_measere_info( size_t size )
{
   cout << "vector size: " << size << " elements, (" << sizeof(double) * size / 1048576. << " mb)" << endl;
}

size_t vec_add_test_t::size_by_measure_idx( size_t meas_idx )
{
   static int   size_incr = max_data_size_ / measurement_cnt_
              , size = 0;
   
   size += size_incr;

   return size;
}

test_units_t tests_init()
{
   test_units_t tests(new vector<test_unit_t>);
   
   test_unit_t unit_test("CPP test", va_calc_cpp, "cpp.test", "cpp");
   tests->push_back(unit_test);
   
   unit_test = test_unit_t("CUDA fake", va_calc_cu, "cuda.test", "cuda-fake", 1);
   tests->push_back(unit_test);

   unit_test = test_unit_t("CUDA test", va_calc_cu, "cuda.test", "cuda");
   tests->push_back(unit_test);

   unit_test = test_unit_t("CUDA with check test", va_calc_cu_with_check, "cuda_wc.test", "cudawc");
   tests->push_back(unit_test);

   return tests;
}

int run_vec_add_test( int argc, char ** argv )
{
   try{
      vec_add_test_t vec_add_test(argc, argv, "vec_add_test", tests_init());

      vec_add_test.run();
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

   time_res.measure_start();
   
   for (size_t i = 0; i < size; ++i)
   {
      c[i] = a[i] + b[i];
   }
   
   time_res.computing_time_ = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;
   
   return time_res;
}
