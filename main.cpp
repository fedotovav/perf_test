#include <tclap/CmdLine.h>
#include <tclap/Arg.h>

#include "test.h"

extern void parse_cmd_args(int argc, char** argv);
extern void global_test(int max_size, int tests_cnt);

string output_file_name;

size_t   max_matr_size
       , tests_cnt;


int main( int argc, char ** argv )
{
   parse_cmd_args(argc, argv);
   
   global_test(max_matr_size, tests_cnt);
   
   return 0;
}

extern "C"
{
   int calc_parallel_f  ( int size, const double * a, const double * b, double * c );
   int calc_one_thread_f( int size, const double * a, const double * b, double * c );

   int compare_2_matr( int size, const double * a, const double * b, double * max_diff, int * max_diff_idx );
   int fill_2_matr   ( int size, const double * a, const double * b );
}

extern time_res_t calc_cu ( int size, const double * a, const double * b, double * c );
extern time_res_t calc_ocl( int size, const double * a, const double * b, double * c );

time_res_t calc_parallel_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   chrono::time_point<chrono::system_clock> time_start, time_finish;

   time_start = chrono::system_clock::now();

   calc_parallel_f(size, a, b, c);

   time_finish = chrono::system_clock::now();

   size_t duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish - time_start).count();

   time_res.computing_time_    = duration;
   time_res.mem_allocate_time_ = 0;

   return time_res;
}

time_res_t calc_one_thread_fort( int size, const double * a, const double * b, double * c )
{
   time_res_t time_res;
   
   time_res.computing_time_    = calc_one_thread_f(size, a, b, c);
   time_res.mem_allocate_time_ = 0;
   
   return time_res;
}

int calc_cpp( int size, const double * a, const double * b, double * c )
{
   int cur_idx;

   for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j)
      {
         cur_idx = i * size + j;
         
         c[cur_idx] = 0;

         for (int k = 0; k < size; ++k)
            c[cur_idx] += a[i * size + k] * b[k * size + j];
      }
}

void write_matr_to_file( ofstream & output_file, double * matr, int size )
{
   for (int i = 0; i < size; ++i)
   {
      for (int j = 0; j < size; ++j)
         output_file << matr[i] << " ";
      
      output_file << endl;
   }
}

// return max difference
double compare_res( const vector<test_unit_t> & tests, int size )
{
   double   * ideal_res = new double[size * size]
          , * other_res = new double[size * size];

   ifstream ideal_file (tests[0].output_file_.c_str());

   for (int i = 0; i < size * size; ++i)
      ideal_file >> ideal_res[i];

   double max_diff;
   
   int max_diff_idx;
   
   for (int k = 1; k < tests.size(); ++k)
   {
      ifstream input_file(tests[k].output_file_.c_str());

      for (int i = 0; i < size * size; ++i)
         input_file >> other_res[i];

      compare_2_matr(size, ideal_res, other_res, &max_diff, &max_diff_idx);

      if (max_diff != 0)
      {
         cout << tests[k].test_name_ << " result output is incorrect! (maximum difference: " << max_diff << ", index:" << max_diff_idx << ")"<< endl;
         max_diff = 0;
         max_diff_idx = 0;
      }
      
      max_diff = 0;
      max_diff_idx = 0;
      
      input_file.close();
   }
   
   delete[] ideal_res;
   delete[] other_res;
}

vector<test_unit_t> tests_init()
{
   vector<test_unit_t> tests(3);
   
   tests[2].calc_func_   = calc_parallel_fort;
   tests[2].output_file_ = "f_par.test";
   tests[2].test_name_   = "Fortran multithread test";
   
   tests[1].calc_func_   = calc_ocl;
   tests[1].output_file_ = "cl.test";
   tests[1].test_name_   = "OpenCL test";

   tests[0].calc_func_   = calc_cu;
   tests[0].output_file_ = "cu_par.test";
   tests[0].test_name_   = "CUDA test";

   return tests;
}

void test( const vector<test_unit_t> & tests, const double * a, const double * b, double * c, int size, ofstream & res_file )
{
   res_file << size << "\t";

   time_res_t duration;
   
   for (size_t i = 0; i < tests.size(); ++i)
   {
      cout << "call " << tests[i].test_name_ << endl;

      duration = tests[i].calc_func_(size, a, b, c);

      cout << "computation time: " << duration.computing_time_ << " ms" << endl;
      cout << "memory allocation time: " << duration.mem_allocate_time_ << " ms" << endl;
      cout << "total time: " << duration.mem_allocate_time_ + duration.computing_time_ << " ms" << endl << endl;
      
      res_file << duration.computing_time_ << "\t" << duration.mem_allocate_time_
               << "\t" << duration.mem_allocate_time_ + duration.computing_time_ << "\t";
      
      ofstream output_file(tests[i].output_file_.c_str());
      
      write_matr_to_file(output_file, c, size);
      
      output_file.close();
   }
   
   res_file << endl;
}

void global_test( int max_size, int tests_cnt )
{
   max_size -= max_size % 32;
   
   int   size_decr = max_size / tests_cnt
       , size;
   
   size_decr -= size_decr % 32;
   size       = size_decr;
   
   vector<test_unit_t> tests = tests_init();

   int test_idx = 0;
   
   ofstream result_file(output_file_name);
   
   result_file << "%% fields: \"size\" ";
   
   for (size_t i = 0; i < tests.size(); ++i)
      result_file << "\"" << tests[i].test_name_ << "\" ";
   
   result_file << endl << "%format of tests output (compute_time, mem_alloc_time, total_time)" << endl;
   
   result_file << endl;
      
   chrono::time_point<chrono::system_clock> start_test_time, finish_test_time;

   cout << "============= START GLOBAL TEST =============" << endl << endl;

   start_test_time = chrono::system_clock::now();
   
   while (size < max_size + 1)
   {
      cout << "---------test #" << test_idx << "---------" << endl;
      cout << "matrix size: " << size << "x" << size << " (" << size * size << " elements, " << sizeof(double) * size * size /  1048576 << " mb)" << endl;

      double  * a = new double[size * size]
            , * b = new double[size * size]
            , * c = new double[size * size];

      fill_2_matr(size, a, b);
      
      test(tests, a, b, c, size, result_file);

      delete[] a;
      delete[] b;
      delete[] c;

      compare_res(tests, size);
      
      size += size_decr;
      test_idx++;
      
      cout << endl;
   }

   finish_test_time = std::chrono::system_clock::now();
   
   cout << "============= FINISH GLOBAL TEST =============" << endl;
   
   size_t hours   = chrono::duration_cast<chrono::hours>  (finish_test_time - start_test_time).count();
   size_t minutes = chrono::duration_cast<chrono::minutes>(finish_test_time - start_test_time).count();
   size_t seconds = chrono::duration_cast<chrono::seconds>(finish_test_time - start_test_time).count();

   cout << "Test done for " << hours << "h, "
                            << minutes % 24 << "m, "
                            << seconds % 60 << "s"
                            << endl;
   
   result_file.close();
}

void parse_cmd_args( int argc, char** argv )
{
   try {
      TCLAP::CmdLine cmd("CUDA vs OpenMP performance test", ' ', "1.1", true);

      TCLAP::ValueArg<std::string> output_file_par  ("o", "output-file", "Test result file", true, "", "file name");
      TCLAP::ValueArg<size_t>      max_matr_size_par("s", "max-matr-size", "Maximum matix size in one dimmension", true, 0, "matr size");
      TCLAP::ValueArg<size_t>      tests_cnt_par    ("c", "tests-count", "Count of performance tests", true, 0, "tests count");

      //TCLAP::SwitchArg sys_info_par("i", "sys-info", "show system info");

      cmd.add(&tests_cnt_par);
      cmd.add(&max_matr_size_par);
      cmd.add(&output_file_par);

      cmd.parse(argc, argv);

      tests_cnt        = tests_cnt_par    .getValue();
      max_matr_size    = max_matr_size_par.getValue();
      output_file_name = output_file_par  .getValue();
   }
   catch (TCLAP::ArgException &err) {
      std::cerr << "error: " << err.error() << " for arg " << err.argId() << std::endl;
   }
}
