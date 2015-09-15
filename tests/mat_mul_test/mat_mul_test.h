#pragma once

#include "../test.h"

template<typename T>
class mat_mul_test_t : public test_t<T>
{
public:
   size_t      size_by_measure_idx( size_t meas_idx );
   void        print_measere_info ( size_t size );
   typename test_t<T>::test_data_t prepare_data       ( size_t size );
   void        write_data_to_file ( ofstream & output_file, const T * data, size_t size );
   void        clear_data         ( typename test_t<T>::test_data_t data );
   void        compare_res        ( size_t size, size_t golden_test_idx );

   mat_mul_test_t( int argc, char ** argv, const string & test_name, const typename test_t<T>::test_units_t tests );
};

template<typename T>
mat_mul_test_t<T>::mat_mul_test_t( int argc, char ** argv, const string & test_name, const typename test_t<T>::test_units_t tests ) :
   test_t<T>(argc, argv, test_name, tests)
{
}

template<typename T>
void mat_mul_test_t<T>::write_data_to_file( ofstream & output_file, const T * data, size_t size )
{
   for (size_t i = 0; i < size * size; ++i)
      output_file << data[i] << " ";
}

template<typename T>
typename test_t<T>::test_data_t mat_mul_test_t<T>::prepare_data( size_t size )
{
   typename test_t<T>::test_data_t matrices(new T*[3]);

   matrices.get()[0] = new T[size * size];
   matrices.get()[1] = new T[size * size];
   matrices.get()[2] = new T[size * size];
   
   fill_2_arrays(size * size, matrices.get()[0], matrices.get()[1]);
   
   return matrices;
}

template<typename T>
void mat_mul_test_t<T>::clear_data( typename test_t<T>::test_data_t data )
{
   delete[] data.get()[0];
   delete[] data.get()[1];
   delete[] data.get()[2];
   
   data.reset();
}

template<typename T>
void mat_mul_test_t<T>::print_measere_info( size_t size )
{
   cout << "matrix size: " << size << "x" << size << " (" << size * size << " elements, " << sizeof(T) * size * size / 1048576. << " mb)" << endl;
}

template<typename T>
size_t mat_mul_test_t<T>::size_by_measure_idx( size_t meas_idx )
{
   static int   size_incr = test_t<T>::max_data_size_ / test_t<T>::measurement_cnt_
              , size = 0;
   
   size += size_incr;

   return size;
}

template<typename T>
void mat_mul_test_t<T>::compare_res( size_t size, size_t golden_test_idx )
{
   T   * ideal_res = new T[size * size]
     , * other_res;

   ifstream ideal_file(test_t<T>::tests_->at(golden_test_idx).check_file());

   for (size_t i = 0; i < size * size; ++i)
      ideal_file >> ideal_res[i];

   T max_diff;
   
   int max_diff_idx;
   
   for (size_t k = 0; k < test_t<T>::tests_->size(); ++k)
   {
      if (k == golden_test_idx || test_t<T>::tests_->at(k).is_fake())
         continue;
      
      ifstream input_file(test_t<T>::tests_->at(k).check_file());
      
      other_res = new T[size * size];

      for (size_t i = 0; i < size * size; ++i)
         input_file >> other_res[i];

      compare_2_arrays(size * size, ideal_res, other_res, &max_diff, &max_diff_idx);

      if (max_diff != 0)
      {
         cout << test_t<T>::tests_->at(k).name() << " result output is incorrect! (maximum difference: " << max_diff << ", index:" << max_diff_idx << ")"<< endl;
         max_diff     = 0;
         max_diff_idx = 0;
      }
      
      max_diff     = 0;
      max_diff_idx = 0;
      
      delete[] other_res;
      
      input_file.close();
   }
   
   delete[] ideal_res;
}

extern "C"
{
   int calc_four_thread_double( int size, const double * a, const double * b, double * c );
   int calc_four_thread_float ( int size, const float * a, const float * b, float * c );
   int calc_four_thread_int   ( int size, const int * a, const int * b, int * c );

   int calc_two_thread_double( int size, const double * a, const double * b, double * c );
   int calc_two_thread_float ( int size, const float * a, const float * b, float * c );
   int calc_two_thread_int   ( int size, const int * a, const int * b, int * c );

   int calc_one_thread_double( int size, const double * a, const double * b, double * c );
   int calc_one_thread_float ( int size, const float * a, const float * b, float * c );
   int calc_one_thread_int   ( int size, const int * a, const int * b, int * c );
}

time_res_t mm_calc_cu( int size, const double * a, const double * b, double * c );
time_res_t mm_calc_cu( int size, const float * a, const float * b, float * c );
time_res_t mm_calc_cu( int size, const int * a, const int * b, int * c );

//time_res_t calc_ocl( int size, const double * a, const double * b, double * c );
//time_res_t calc_ocl( int size, const float * a, const float * b, float * c );
//time_res_t calc_ocl( int size, const int * a, const int * b, int * c );

time_res_t calc_one_thread_fort ( int size, const double * a, const double * b, double * c );
time_res_t calc_one_thread_fort ( int size, const float * a, const float * b, float * c );
time_res_t calc_one_thread_fort ( int size, const int * a, const int * b, int * c );

time_res_t calc_four_thread_fort( int size, const double * a, const double * b, double * c );
time_res_t calc_four_thread_fort( int size, const float * a, const float * b, float * c );
time_res_t calc_four_thread_fort( int size, const int * a, const int * b, int * c );

time_res_t calc_two_thread_fort ( int size, const double * a, const double * b, double * c );
time_res_t calc_two_thread_fort ( int size, const float * a, const float * b, float * c );
time_res_t calc_two_thread_fort ( int size, const int * a, const int * b, int * c );

time_res_t mm_calc_cpp( int size, const double * a, const double * b, double * c );
time_res_t mm_calc_cpp( int size, const float * a, const float * b, float * c );
time_res_t mm_calc_cpp( int size, const int * a, const int * b, int * c );

template<typename T>
typename test_t<T>::test_units_t tests_init()
{
   typename test_t<T>::test_units_t tests(new vector<test_unit_t<T>>);
   
   test_unit_t<T> unit_test("CPP test", mm_calc_cpp, "cpp.test", "cpp");
   tests->push_back(unit_test);
   
   unit_test = test_unit_t<T>("CUDA fake", mm_calc_cu, "cudaf.test", "cuda-fake", 1);
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("CUDA test", mm_calc_cu, "cuda.test", "cuda");
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("OpenMP four thread test", calc_four_thread_fort, "omp_4t.test", "openmp4t");
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("OpenMP two thread test", calc_two_thread_fort, "omp_2t.test", "openmp2t");
   tests->push_back(unit_test);

   unit_test = test_unit_t<T>("Fortran test", calc_one_thread_fort, "f.test", "fortran");
   tests->push_back(unit_test);

//   unit_test = test_unit_t<T>("OpenCL test", calc_ocl, "cl.test", "opencl");
//   tests->push_back(unit_test);

   return tests;
}

int run_matr_mul_test( int argc, char ** argv )
{
   try{
      mat_mul_test_t<double> mat_mul_test(argc, argv, "matr_mul_test", tests_init<double>());

      mat_mul_test.run();
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
   
   time_res.measure_start();

   calc_one_thread_double(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_one_thread_fort( int size, const float * a, const float * b, float * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_one_thread_float(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_one_thread_fort( int size, const int * a, const int * b, int * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_one_thread_int(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_four_thread_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_four_thread_double(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_four_thread_fort( int size, const float * a, const float * b, float * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_four_thread_float(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_four_thread_fort( int size, const int * a, const int * b, int * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_four_thread_int(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_two_thread_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_two_thread_double(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_two_thread_fort( int size, const float * a, const float * b, float * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_two_thread_float(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_two_thread_fort( int size, const int * a, const int * b, int * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_two_thread_int(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

template<typename T>
time_res_t calc_cpp( int size, const T * a, const T * b, T * c )
{
   time_res_t time_res;

   size_t cur_idx;

   time_res.measure_start();
   
   for (size_t i = 0; i < size; ++i)
      for (size_t j = 0; j < size; ++j)
      {
         cur_idx = i * size + j;
         
         c[cur_idx] = 0;

         for (int k = 0; k < size; ++k)
            c[cur_idx] += a[i * size + k] * b[k * size + j];
      }
   
   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;
   
   return time_res;
}

time_res_t mm_calc_cpp( int size, const float * a, const float * b, float * c )
{
   return calc_cpp<float>(size, a, b, c);
}

time_res_t mm_calc_cpp( int size, const double * a, const double * b, double * c )
{
   return calc_cpp<double>(size, a, b, c);
}

time_res_t mm_calc_cpp( int size, const int * a, const int * b, int * c )
{
   return calc_cpp<int>(size, a, b, c);
}
