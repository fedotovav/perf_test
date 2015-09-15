#pragma once

#include "../test.h"

template<typename T>
class vec_add_test_t : public test_t<T>
{
public:
   size_t size_by_measure_idx( size_t meas_idx );
   void   print_measere_info ( size_t size );

   vec_add_test_t( int argc, char ** argv, const string & test_name, const typename test_t<T>::test_units_t tests );
};

template<typename T>
vec_add_test_t<T>::vec_add_test_t( int argc, char ** argv, const string & test_name, const typename test_t<T>::test_units_t tests ) :
   test_t<T>(argc, argv, test_name, tests)
{
}

template<typename T>
void vec_add_test_t<T>::print_measere_info( size_t size )
{
   cout << "vector size: " << size << " elements, (" << sizeof(T) * size / 1048576. << " mb)" << endl;
}

template<typename T>
size_t vec_add_test_t<T>::size_by_measure_idx( size_t meas_idx )
{
   static int   size_incr = test_t<T>::max_data_size_  / test_t<T>::measurement_cnt_
              , size = 0;
   
   size += size_incr;

   return size;
}

time_res_t va_calc_cu( int size, const int * a, const int * b, int * c );
time_res_t va_calc_cu( int size, const double * a, const double * b, double * c );
time_res_t va_calc_cu( int size, const float * a, const float * b, float * c );

time_res_t va_calc_cu_wc( int size, const int * a, const int * b, int * c );
time_res_t va_calc_cu_wc( int size, const double * a, const double * b, double * c );
time_res_t va_calc_cu_wc( int size, const float * a, const float * b, float * c );

time_res_t va_calc_cpp( int size, const int * a, const int * b, int * c );
time_res_t va_calc_cpp( int size, const double * a, const double * b, double * c );
time_res_t va_calc_cpp( int size, const float * a, const float * b, float * c );

template<typename T>
typename test_t<T>::test_units_t tests_init()
{
   typename test_t<T>::test_units_t tests(new vector<test_unit_t<T>>);
   
   test_unit_t<T> unit_test("CPP test", va_calc_cpp, "cpp.test", "cpp");
   tests->push_back(unit_test);
   
   unit_test = test_unit_t<T>("CUDA fake", va_calc_cu, "cuda.test", "cuda-fake", 1);
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("CUDA test", va_calc_cu, "cuda.test", "cuda");
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("CUDA with check test", va_calc_cu_wc, "cuda_wc.test", "cudawc");
   tests->push_back(unit_test);

   return tests;
}

int run_vec_add_test( int argc, char ** argv )
{
   try{
      vec_add_test_t<int> vec_add_test(argc, argv, "vec_add_test", tests_init<int>());

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
   
   return 0;
}

template<typename T>
time_res_t calc_cpp( int size, const T * a, const T * b, T * c )
{
   time_res_t time_res;

   time_res.measure_start();
   
   for (size_t i = 0; i < size; ++i)
      c[i] = a[i] + b[i];
   
   time_res.computing_time_ = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;
   
   return time_res;
}

time_res_t va_calc_cpp( int size, const float * a, const float * b, float * c )
{
   return calc_cpp<float>(size, a, b, c);
}

time_res_t va_calc_cpp( int size, const double * a, const double * b, double * c )
{
   return calc_cpp<double>(size, a, b, c);
}

time_res_t va_calc_cpp( int size, const int * a, const int * b, int * c )
{
   return calc_cpp<int>(size, a, b, c);
}
