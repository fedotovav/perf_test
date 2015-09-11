#include "test.h"

extern "C"
{
   int compare_2_matr( int size, const double * a, const double * b, double * max_diff, int * max_diff_idx );
   int fill_2_matr   ( int size, const double * a, const double * b );
}

test_unit_t::test_unit_t( const string & test_name, const calc_func_t & func, const string & output_file_name
                         ,const string & cmd_line_par_name, int is_golden_test ) :
     test_name_        (test_name)
   , calc_func         (func)
   , output_file_      (output_file_name)
   , cmd_line_par_name_(cmd_line_par_name)
   , is_golden_test_   (is_golden_test)
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

int test_unit_t::is_golden_test() const
{
   return is_golden_test_;
}

test_t::test_t( int argc, char ** argv, test_units_t tests ) :
     max_matr_size_   (0)
   , matr_size_limit_ (0)
   , measurement_cnt_ (0)
   , output_file_name_("default")
   , goloden_test_idx_(-1)
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

   for (size_t i = 0; i < tests.get()->size(); ++i)
      tests_options.add_options()(tests->at(i).cmd_line_par().c_str(), "test unit");
   
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
   
   if (tests)
      tests_ = tests;
   else
   {
      cerr << "No oune test found!" << endl;

      throw;
   }
   
   if (vm.count("all-tests-units"))
   {
      for (size_t i = 0; i < tests_.get()->size(); ++i)
      {
         if (tests_->at(i).is_golden_test())
         {
            if (goloden_test_idx_ < 0)
               goloden_test_idx_ = i;
            else
            {
               cerr << "Alredy have golden test: (" << tests->at(goloden_test_idx_).name() << ")" << endl;

               throw;
            }
         }
      }
   }
   else
      for (size_t i = 0; i < tests.get()->size(); ++i)
      {
         if (!vm.count(tests->at(i).cmd_line_par()))
            tests_->erase(tests_->begin() + i);

         if (tests->at(i).is_golden_test())
         {
            if (goloden_test_idx_ < 0)
               goloden_test_idx_ = i;
            else
            {
               cerr << "Alredy have golden test: (" << tests->at(goloden_test_idx_).name() << ")" << endl;

               throw "Golden test set error";
            }
         }
      }
   
   if (!tests_->size())
   {
      cerr << "Bad program options usage! Example of using options:" << endl;
      cerr << "./<program name> -o test.res -c 10 -s 1000 --<some test name>" << endl << endl;

      throw head_description;
   }

   
   if (goloden_test_idx_ < 0)
   {
      cerr << "Golden test doesn't set!" << endl;

      throw "Golden test set error";
   }
}

// return max difference
double compare_res( const test_units_t tests, int size, int golden_test_idx )
{
   double   * ideal_res = new double[size * size]
          , * other_res = new double[size * size];

   ifstream ideal_file (tests->at(golden_test_idx).check_file());

   for (int i = 0; i < size * size; ++i)
      ideal_file >> ideal_res[i];

   double max_diff;
   
   int max_diff_idx;
   
   for (int k = 0; k < tests->size(); ++k)
   {
      if (k == golden_test_idx)
         continue;
      
      ifstream input_file(tests->at(k).check_file());

      for (int i = 0; i < size * size; ++i)
         input_file >> other_res[i];

      compare_2_matr(size, ideal_res, other_res, &max_diff, &max_diff_idx);

      if (max_diff != 0)
      {
         cout << tests->at(k).name() << " result output is incorrect! (maximum difference: " << max_diff << ", index:" << max_diff_idx << ")"<< endl;
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

void write_matr_to_file( ofstream & output_file, const double * matr, int size )
{
   for (int i = 0; i < size; ++i)
   {
      for (int j = 0; j < size; ++j)
         output_file << matr[i] << " ";
      
      output_file << endl;
   }
}

void remove_tmp_files( const vector<test_unit_t> & tests )
{
   for (size_t i = 0; i < tests.size(); ++i)
      remove(tests[i].check_file().c_str());
}

void test( const test_units_t tests, const double * a, const double * b, double * c, int size, ofstream & res_file )
{
   res_file << size << "\t";

   time_res_t duration;
   
   for (size_t i = 0; i < tests->size(); ++i)
   {
      cout << "call " << tests->at(i).name() << endl;

      duration = tests->at(i).calc_func(size, a, b, c);

      cout << "computation time: " << duration.computing_time_ << " ms" << endl;
      cout << "memory allocation time: " << duration.mem_allocate_time_ << " ms" << endl;
      cout << "total time: " << duration.mem_allocate_time_ + duration.computing_time_ << " ms" << endl << endl;
      
      res_file << duration.computing_time_ << "\t" << duration.mem_allocate_time_
               << "\t" << duration.mem_allocate_time_ + duration.computing_time_ << "\t";
      
      ofstream output_file(tests->at(i).check_file());
      
      write_matr_to_file(output_file, c, size);
      
      output_file.close();
   }
   
   res_file << endl;
}

void test_t::start()
{
   max_matr_size_ -= max_matr_size_ % 32;
   
   int   size_incr = max_matr_size_ / measurement_cnt_
       , size;
   
   size_incr -= size_incr % 32;
   size       = size_incr;
   
   int test_idx = 0;
   
   ofstream result_file(output_file_name_);
   
   result_file << "%% fields: \"size\" ";
   
   for (size_t i = 0; i < tests_->size(); ++i)
      result_file << "\"" << tests_->at(i).name() << "\" ";
   
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

      compare_res(tests_, size, goloden_test_idx_);
      
      //remove_tmp_files(tests_);
      
      size += size_incr;
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
