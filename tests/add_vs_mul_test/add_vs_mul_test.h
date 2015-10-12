#pragma once

#include "../test.h"

template<typename T>
class vec_mul_test_t : public test_t<T>
{
public:
   size_t size_by_measure_idx( size_t meas_idx );
   void   print_measere_info ( size_t size );

   vec_mul_test_t( int argc, char ** argv, const string & test_name, const typename test_t<T>::test_units_t tests );
};

template<typename T>
vec_mul_test_t<T>::vec_mul_test_t( int argc, char ** argv, const string & test_name, const typename test_t<T>::test_units_t tests ) :
   test_t<T>(argc, argv, test_name, tests)
{
}

template<typename T>
void vec_mul_test_t<T>::print_measere_info( size_t size )
{
   cout << "vector size: " << size << " elements, (" << sizeof(T) * size / 1048576. << " mb)" << endl;
}

template<typename T>
size_t vec_mul_test_t<T>::size_by_measure_idx( size_t meas_idx )
{
   static int   size_incr = test_t<T>::max_data_size_  / test_t<T>::measurement_cnt_
              , size = 0;
   
   size += size_incr;

   return size;
}

time_res_t vm_calc_cu( int size, const int * a, const int * b, int * c );
time_res_t vm_calc_cu( int size, const double * a, const double * b, double * c );
time_res_t vm_calc_cu( int size, const float * a, const float * b, float * c );

time_res_t va_calc_cu( int size, const int * a, const int * b, int * c );
time_res_t va_calc_cu( int size, const double * a, const double * b, double * c );
time_res_t va_calc_cu( int size, const float * a, const float * b, float * c );

template<typename T>
typename test_t<T>::test_units_t tests_init()
{
   typename test_t<T>::test_units_t tests(new vector<test_unit_t<T>>);
   
   test_unit_t<T> unit_test = test_unit_t<T>("CUDA fake", va_calc_cu, "cuda.test", "cuda-fake", 1);
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("CUDA add test", va_calc_cu, "cuda_add.test", "cudaadd");
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("CUDA mul test", va_calc_cu, "cuda_mul.test", "cudamul");
   tests->push_back(unit_test);

   return tests;
}

int run_add_vs_mul_test( int argc, char ** argv )
{
   try{
      vec_mul_test_t<double> add_vs_mul_test(argc, argv, "add_vs_mul_test", tests_init<double>());

      add_vs_mul_test.run();
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
