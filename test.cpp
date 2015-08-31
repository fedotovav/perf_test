#include "test.h"

extern "C"
{
   int compare_2_matr( int size, const double * a, const double * b, double * max_diff, int * max_diff_idx );
   int fill_2_matr   ( int size, const double * a, const double * b );
}

test_unit_t::test_unit_t( const string & test_name, const calc_func_t & func, const string & output_file_name
                         ,const string & cmd_line_par_name ) :
     test_name_        (test_name)
   , calc_func         (func)
   , output_file_      (output_file_name)
   , cmd_line_par_name_(cmd_line_par_name)
{
}

const string & test_unit_t::check_file() const
{
   return output_file_;
}

const string & test_unit_t::name() const
{
   return test_name_;
}

const string & test_unit_t::cmd_line_par() const
{
   return cmd_line_par_name_;
}

test_t::test_t( int argc, char ** argv, const vector<test_unit_t> & tests ) :
     max_matr_size_  (0)
   , matr_size_limit_(0)
   , measurement_cnt_(0)
   , output_file_name_("default")
{
   po::options_description head_description("Computing technologies performance test\n Author: Anton Fedotov");

   po::options_description general_options("General options");
   general_options.add_options()
   ("help,h", "show help")
   ("measurement-cnt,c", po::value<size_t>(&measurement_cnt_), "set measurement count")
   ("output-file,o", po::value<string>(&output_file_name_), "test result output file name")
   ("max-matr-size,s", po::value<size_t>(&max_matr_size_), "maximum matrix size");

   po::options_description tests_options("Available tests");

   tests_options.add_options()("all-tests-units,all", "do all test units");

   for (size_t i = 0; i < tests.size(); ++i)
      tests_options.add_options()(tests[i].cmd_line_par().c_str(), "test unit");
   
   head_description.add(general_options).add(tests_options);
   
   po::variables_map vm;
   
   try{
      po::parsed_options parsed_ops = po::command_line_parser(argc, argv).options(head_description).allow_unregistered().run();

      po::store(parsed_ops, vm);
      po::notify(vm);
   }catch(...)
   {
      cerr << "Bad program options usage! Example of using options:" << endl;
      cerr << "./<program name> -o test.res -c 10 -s 1000 --<some test name>" << endl << endl;

      throw head_description;
   }
   
   if (vm.count("help"))
      throw head_description;
   
   if (!vm.count("measurement-cnt") || !vm.count("output-file") || !vm.count("max-matr-size"))
   {
      cerr << "Bad program options usage! Example of using options:" << endl;
      cerr << "./<program name> -o test.res -c 10 -s 1000 --<some test name>" << endl << endl;
      
      throw head_description;
   }
   
   if (vm.count("all-tests-units"))
      tests_.insert(tests_.end(), tests.begin(), tests.end());
   else
      for (size_t i = 0; i < tests.size(); ++i)
         if (vm.count(tests[i].cmd_line_par()))
            tests_.push_back(tests[i]);
}

// return max difference
double compare_res( const vector<test_unit_t> & tests, int size )
{
   double   * ideal_res = new double[size * size]
          , * other_res = new double[size * size];

   ifstream ideal_file (tests[0].check_file());

   for (int i = 0; i < size * size; ++i)
      ideal_file >> ideal_res[i];

   double max_diff;
   
   int max_diff_idx;
   
   for (int k = 1; k < tests.size(); ++k)
   {
      ifstream input_file(tests[k].check_file());

      for (int i = 0; i < size * size; ++i)
         input_file >> other_res[i];

      compare_2_matr(size, ideal_res, other_res, &max_diff, &max_diff_idx);

      if (max_diff != 0)
      {
         cout << tests[k].name() << " result output is incorrect! (maximum difference: " << max_diff << ", index:" << max_diff_idx << ")"<< endl;
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

void write_matr_to_file( ofstream & output_file, double * matr, int size )
{
   for (int i = 0; i < size; ++i)
   {
      for (int j = 0; j < size; ++j)
         output_file << matr[i] << " ";
      
      output_file << endl;
   }
}

void test( const vector<test_unit_t> & tests, const double * a, const double * b, double * c, int size, ofstream & res_file )
{
   res_file << size << "\t";

   time_res_t duration;
   
   for (size_t i = 0; i < tests.size(); ++i)
   {
      cout << "call " << tests[i].name() << endl;

      duration = tests[i].calc_func(size, a, b, c);

      cout << "computation time: " << duration.computing_time_ << " ms" << endl;
      cout << "memory allocation time: " << duration.mem_allocate_time_ << " ms" << endl;
      cout << "total time: " << duration.mem_allocate_time_ + duration.computing_time_ << " ms" << endl << endl;
      
      res_file << duration.computing_time_ << "\t" << duration.mem_allocate_time_
               << "\t" << duration.mem_allocate_time_ + duration.computing_time_ << "\t";
      
      ofstream output_file(tests[i].check_file());
      
      write_matr_to_file(output_file, c, size);
      
      output_file.close();
   }
   
   res_file << endl;
}

void test_t::start()
{
   max_matr_size_ -= max_matr_size_ % 32;
   
   int   size_decr = max_matr_size_ / measurement_cnt_
       , size;
   
   size_decr -= size_decr % 32;
   size       = size_decr;
   
   int test_idx = 0;
   
   ofstream result_file(output_file_name_);
   
   result_file << "%% fields: \"size\" ";
   
   for (size_t i = 0; i < tests_.size(); ++i)
      result_file << "\"" << tests_[i].name() << "\" ";
   
   result_file << endl << "%format of tests output (compute_time, mem_alloc_time, total_time)" << endl;
   
   result_file << endl;
      
   chrono::time_point<chrono::system_clock> start_test_time, finish_test_time;

   cout << "============= START GLOBAL TEST =============" << endl << endl;

   start_test_time = chrono::system_clock::now();
   
   while (size < max_matr_size_ + 1)
   {
      cout << "---------test #" << test_idx << "---------" << endl;
      cout << "matrix size: " << size << "x" << size << " (" << size * size << " elements, " << sizeof(double) * size * size /  1048576 << " mb)" << endl;

      double  * a = new double[size * size]
            , * b = new double[size * size]
            , * c = new double[size * size];

      fill_2_matr(size, a, b);
      
      test(tests_, a, b, c, size, result_file);

      delete[] a;
      delete[] b;
      delete[] c;

      compare_res(tests_, size);
      
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
