#pragma once

#include "../test.h"

extern "C"
{
   int calc_four_thread_f( int size, const double * a, const double * b, double * c );
   int calc_two_thread_f ( int size, const double * a, const double * b, double * c );
   int calc_one_thread_f ( int size, const double * a, const double * b, double * c );
}

extern time_res_t mm_calc_cu    ( int size, const double * a, const double * b, double * c );
extern time_res_t calc_ocl      ( int size, const double * a, const double * b, double * c );
time_res_t calc_one_thread_fort ( int size, const double * a, const double * b, double * c );
time_res_t calc_four_thread_fort( int size, const double * a, const double * b, double * c );
time_res_t calc_two_thread_fort ( int size, const double * a, const double * b, double * c );
time_res_t mm_calc_cpp          ( int size, const double * a, const double * b, double * c );

class mat_mul_test_t : public test_t
{
public:
   virtual size_t      size_by_measure_idx( size_t meas_idx );
   virtual void        print_measere_info ( size_t size );
   virtual test_data_t prepare_data       ( size_t size );
   virtual void        write_data_to_file ( ofstream & output_file, const double * data, size_t size );
   virtual void        clear_data         ( test_data_t data );

   mat_mul_test_t( int argc, char ** argv, const string & test_name, const test_units_t tests );
};

mat_mul_test_t::mat_mul_test_t( int argc, char ** argv, const string & test_name, const test_units_t tests ) :
   test_t(argc, argv, test_name, tests)
{
}

void mat_mul_test_t::write_data_to_file( ofstream & output_file, const double * data, size_t size )
{
   for (int i = 0; i < size * size; ++i)
      output_file << data[i] << " ";
}

test_data_t mat_mul_test_t::prepare_data( size_t size )
{
   test_data_t matrices(new double*[3]);

   matrices.get()[0] = new double[size * size];
   matrices.get()[1] = new double[size * size];
   matrices.get()[2] = new double[size * size];
   
   fill_2_arrays(size * size, matrices.get()[0], matrices.get()[1]);
   
   return matrices;
}

void mat_mul_test_t::clear_data( test_data_t data )
{
   delete[] data.get()[0];
   delete[] data.get()[1];
   delete[] data.get()[2];
   
   data.reset();
}

void mat_mul_test_t::print_measere_info( size_t size )
{
   cout << "matrix size: " << size << "x" << size << " (" << size * size << " elements, " << sizeof(double) * size * size / 1048576. << " mb)" << endl;
}

size_t mat_mul_test_t::size_by_measure_idx( size_t meas_idx )
{
   max_data_size_ -= max_data_size_ % 32;
   
   static int   size_incr = max_data_size_ / measurement_cnt_ - max_data_size_ / measurement_cnt_ % 32
              , size = 0;
   
   size += size_incr;

   return size;
}

test_units_t tests_init()
{
   test_units_t tests(new vector<test_unit_t>);
   
   test_unit_t unit_test("CPP test", mm_calc_cpp, "cpp.test", "cpp");
   tests->push_back(unit_test);
   
   unit_test = test_unit_t("OpenMP four thread test", calc_four_thread_fort, "omp_4t.test", "openmp4t");
   tests->push_back(unit_test);

   unit_test = test_unit_t("OpenMP two thread test", calc_two_thread_fort, "omp_2t.test", "openmp2t");
   tests->push_back(unit_test);

   unit_test = test_unit_t("Fortran test", calc_one_thread_fort, "f.test", "fortran");
   tests->push_back(unit_test);

   unit_test = test_unit_t("OpenCL test", calc_ocl, "cl.test", "opencl");
   tests->push_back(unit_test);

   unit_test = test_unit_t("CUDA test", mm_calc_cu, "cuda.test", "cuda");
   tests->push_back(unit_test);

   return tests;
}

int run_matr_mul_test( int argc, char ** argv )
{
   try{
      mat_mul_test_t mat_mul_test(argc, argv, "matr_mul_test", tests_init());

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

   calc_one_thread_f(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_four_thread_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_four_thread_f(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_two_thread_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   time_res.measure_start();

   calc_two_thread_f(size, a, b, c);

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t mm_calc_cpp( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;

   int cur_idx;

   time_res.measure_start();
   
   for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j)
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
