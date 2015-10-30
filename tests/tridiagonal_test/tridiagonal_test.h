#pragma once

#include "../test.h"

template<typename T>
class tridiagonal_test_t : public test_t<T>
{
public:
   size_t         size_by_measure_idx  ( size_t meas_idx );
   void           print_measere_info   ( size_t size );
   test_data_t<T> prepare_data         ( size_t size );
   void           clear_data           ( test_data_t<T> & data, size_t size );
   void           write_data_to_file   ( ofstream & output_file, test_data_t<T> const & data, size_t size );


   tridiagonal_test_t( int argc, char ** argv, const string & test_name, const typename test_t<T>::test_units_t tests );
   
//   void measurement( typename test_t<T>::test_data_t data, int size, ofstream & res_file, int need_check_file );
};

template<typename T>
tridiagonal_test_t<T>::tridiagonal_test_t( int argc, char ** argv, const string & test_name, const typename test_t<T>::test_units_t tests ) :
     test_t<T>  (argc, argv, test_name, tests)
{
}

template<typename T>
void tridiagonal_test_t<T>::print_measere_info( size_t size )
{
   cout << "matrix size: " << size * 1000 * 1000 << " elements, (" << sizeof(T) * size * 1000 * 1000 / 1048576. << " mb)" << endl;
}

template<typename T>
size_t tridiagonal_test_t<T>::size_by_measure_idx( size_t meas_idx )
{
   static int   size_incr = test_t<T>::max_data_size_  / test_t<T>::measurement_cnt_
              , size = 0;
   
   size += size_incr;

   return size;
}

time_res_t t_calc_gpu( size_t size, test_data_t<int> & data );
time_res_t t_calc_gpu( size_t size, test_data_t<float> & data );
time_res_t t_calc_gpu( size_t size, test_data_t<double> & data );

time_res_t t_calc_cpu( size_t size, test_data_t<int> & data );
time_res_t t_calc_cpu( size_t size, test_data_t<float> & data );
time_res_t t_calc_cpu( size_t size, test_data_t<double> & data );

time_res_t t_calc_cpu_gpu( size_t size, test_data_t<int> & data );
time_res_t t_calc_cpu_gpu( size_t size, test_data_t<float> & data );
time_res_t t_calc_cpu_gpu( size_t size, test_data_t<double> & data );

time_res_t t_calc_cpu_parallel( size_t size, test_data_t<int> & data );
time_res_t t_calc_cpu_parallel( size_t size, test_data_t<float> & data );
time_res_t t_calc_cpu_parallel( size_t size, test_data_t<double> & data );

template<typename T>
typename test_t<T>::test_units_t tests_init()
{
   typename test_t<T>::test_units_t tests(new vector<test_unit_t<T>>);
   
   test_unit_t<T> unit_test("standart", t_calc_cpu, "std.test", "std");
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("four thread", t_calc_cpu_parallel, "4t.test", "4t");
   tests->push_back(unit_test);
   
//   unit_test = test_unit_t<T>("CUDA fake", t_calc_gpu, "cuda.test", "cuda-fake", 1);
//   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("CUDA test", t_calc_gpu, "cuda.test", "cuda");
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("cpugpu test", t_calc_cpu_gpu, "cpu_gpu.test", "cpg");
   tests->push_back(unit_test);

   return tests;
}

template<typename T>
test_data_t<T> tridiagonal_test_t<T>::prepare_data( size_t size )
{
   test_data_t<T> data(new matrix_t<T>[size]);
   
   const size_t matrix_size = 1000;
   
   for (size_t i = 0; i < size; ++i)
   {
      data.get()[i].super_diag_ = new T[matrix_size - 1];
      data.get()[i].diag_       = new T[matrix_size];
      data.get()[i].sub_diag_   = new T[matrix_size - 1];
      data.get()[i].right_prt_  = new T[matrix_size];
      data.get()[i].solution_   = new T[matrix_size];

      fill_2_arrays(size - 1, data.get()[i].sub_diag_, data.get()[i].super_diag_);
      fill_2_arrays(size, data.get()[i].diag_, data.get()[i].right_prt_);
   }
   
   return data;
}

template<typename T>
void tridiagonal_test_t<T>::clear_data( test_data_t<T> & data, size_t size )
{
//   for (size_t i = 0; i < size; ++i)
//   {
//      delete[] data.get()[i].diag_;
//      delete[] data.get()[i].right_prt_;
//      delete[] data.get()[i].solution_;
//      delete[] data.get()[i].sub_diag_;
//      delete[] data.get()[i].super_diag_;
//   }
   
   data.reset();
}

template<typename T>
void tridiagonal_test_t<T>::write_data_to_file( ofstream & output_file, test_data_t<T> const & data, size_t size )
{
//   for (int i = 0; i < size - 1; ++i)
//      output_file << data.get()[0][i] << " ";
//
//   output_file << endl; 
//
//   for (int i = 0; i < size; ++i)
//      output_file << data.get()[1][i] << " ";
//
//   output_file << endl; 
//
//   for (int i = 0; i < size - 1; ++i)
//      output_file << data.get()[2][i] << " ";
//
//   output_file << endl; 
//
//   for (int i = 0; i < size; ++i)
//      output_file << data.get()[3][i] << " ";
//
//   output_file << endl; 
//
//   for (int i = 0; i < size; ++i)
//      output_file << data.get()[4][i] << " ";
}

inline int run_tridiagonal_test( int argc, char ** argv )
{
   try{
      tridiagonal_test_t<float> tridiagonal_test(argc, argv, "tridiagonal_test", tests_init<float>());

      tridiagonal_test.run();
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


// -- public functions

template< typename T >
void create_support_slae( size_t size, size_t threads_cnt, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt
                         ,T * new_subdiag, T * new_diag, T * new_superdiag, T * new_right_prt );

template<typename T>
time_res_t calc_cpp_parallel( size_t size, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt, T * solution );

template<typename T>
void copy_sup_solution_to_solution( size_t size, size_t threads_cnt, const T * sup_solution, T * solution );

template<typename T>
void copy_slae( int size, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt
               ,T * new_subdiag, T * new_diag, T * new_superdiag, T * new_right_prt );

template<typename T>
time_res_t calc_cpp( size_t size, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt, T * solution );
