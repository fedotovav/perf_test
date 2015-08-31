#include "test.h"

extern "C"
{
   int calc_four_thread_f( int size, const double * a, const double * b, double * c );
   int calc_two_thread_f( int size, const double * a, const double * b, double * c );
   int calc_one_thread_f( int size, const double * a, const double * b, double * c );
}

extern time_res_t calc_cu ( int size, const double * a, const double * b, double * c );
extern time_res_t calc_ocl( int size, const double * a, const double * b, double * c );
time_res_t calc_one_thread_fort( int size, const double * a, const double * b, double * c );
time_res_t calc_four_thread_fort( int size, const double * a, const double * b, double * c );
time_res_t calc_two_thread_fort( int size, const double * a, const double * b, double * c );
time_res_t calc_cpp( int size, const double * a, const double * b, double * c );

vector<test_unit_t> tests_init()
{
   vector<test_unit_t> tests;
   
   test_unit_t unit_test("CPP test", calc_cpp, "cpp.test", "cpp");
   tests.push_back(unit_test);
   
   unit_test = test_unit_t("OpenMP four thread test", calc_four_thread_fort, "omp_4t.test", "openmp-4f");
   tests.push_back(unit_test);

   unit_test = test_unit_t("OpenMP two thread test", calc_two_thread_fort, "omp_2t.test", "openmp-4f");
   tests.push_back(unit_test);

   unit_test = test_unit_t("Fortran test", calc_one_thread_fort, "f.test", "fortran");
   tests.push_back(unit_test);

   unit_test = test_unit_t("OpenCL test", calc_ocl, "cl.test", "opencl");
   tests.push_back(unit_test);

   unit_test = test_unit_t("CUDA test", calc_cu, "cuda.test", "cuda");
   tests.push_back(unit_test);

   return tests;
}

int main( int argc, char ** argv )
{
   try{
      test_t test(argc, argv, tests_init());

      test.start();
   }
   catch(const po::options_description & desc)
   {
      cout << desc;
      
      return 0;
   }catch(...)
   {
      return 1;
   }
   
   return 0;
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
         
         c[cur_idx] = 0;

         for (int k = 0; k < size; ++k)
            c[cur_idx] += a[i * size + k] * b[k * size + j];
      }
   
   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();
   
   time_res.computing_time_ = duration;
   time_res.mem_allocate_time_ = 0;
   
   return time_res;
}
