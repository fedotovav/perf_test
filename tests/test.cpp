#include "test.h"

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

test_t::test_t( int argc, char ** argv, const string & test_name, test_units_t tests, size_func_t size_func
               ,print_test_info_func_t print_test_info_func, prepare_date_func_t prepare_date_func ) :
     max_data_size_       (0)
   , matr_size_limit_     (0)
   , measurement_cnt_     (0)
   , output_file_name_    ("default")
   , goloden_test_idx_    (-1)
   , test_name_           (test_name)
   , size_func_           (size_func)
   , print_test_info_func_(print_test_info_func)
   , prepare_date_func_   (prepare_date_func)

{
   po::options_description head_description("Computing technologies performance test\n Author: Anton Fedotov");

   po::options_description general_options("General options");
   general_options.add_options()
   ("help,h", "show help")
   ("measurement-cnt,c", po::value<size_t>(&measurement_cnt_), "set measurement count")
   ("output-file,o", po::value<string>(&output_file_name_), "test result output file name")
   ("max-matr-size,s", po::value<size_t>(&max_data_size_), "maximum matrix size");

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

   size_t tests_cnt = tests_->size();
   
   int test_exist = 1;
   
   if (vm.count("all-tests-units"))
   {
      for (size_t i = 0; i < tests_cnt; ++i)
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
      for (size_t i = 0, cur_test_idx = 0; i < tests_cnt; ++i)
      {
         if (!vm.count(tests->at(cur_test_idx).cmd_line_par()))
         {
            tests_->erase(tests_->begin() + cur_test_idx);

            test_exist = 0;
         }

         if (test_exist)
         {
            if (tests->at(cur_test_idx).is_golden_test())
            {
               if (goloden_test_idx_ < 0)
                  goloden_test_idx_ = i;
               else
               {
                  cerr << "Alredy have golden test: (" << tests->at(goloden_test_idx_).name() << ")" << endl;

                  throw "Golden test set error";
               }
            }
            
            cur_test_idx++;
         }
         
         test_exist = 1;
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

extern "C" int compare_2_arrays( int size, const double * a, const double * b, double * max_diff, int * max_diff_idx );

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

      compare_2_arrays(size * size, ideal_res, other_res, &max_diff, &max_diff_idx);

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

void write_data_to_file( ofstream & output_file, const double * data, int size )
{
   for (int i = 0; i < size; ++i)
   {
      for (int j = 0; j < size; ++j)
         output_file << data[i] << " ";
      
      output_file << endl;
   }
}

void remove_tmp_files( const test_units_t tests )
{
   for (size_t i = 0; i < tests->size(); ++i)
      if (!tests->at(i).check_file().empty())
         remove(tests->at(i).check_file().c_str());
}

void test( const test_units_t tests, test_data_t matrices, int size, ofstream & res_file )
{
   res_file << size << "\t";

   time_res_t duration;
   
   for (size_t i = 0; i < tests->size(); ++i)
   {
      cout << "call " << tests->at(i).name() << endl;

      duration = tests->at(i).calc_func(size, (const double *)matrices.get()[0], (const double *)matrices.get()[1], (double *)matrices.get()[2]);

      cout << "computation time: " << duration.computing_time_ << " ms" << endl;
      cout << "memory allocation time: " << duration.mem_allocate_time_ << " ms" << endl;
      cout << "total time: " << duration.mem_allocate_time_ + duration.computing_time_ << " ms" << endl << endl;
      
      res_file << duration.computing_time_ << "\t" << duration.mem_allocate_time_
               << "\t" << duration.mem_allocate_time_ + duration.computing_time_ << "\t";
      
      ofstream output_file(tests->at(i).check_file());
      
      write_data_to_file(output_file, (double *)matrices.get()[2], size);
      
      output_file.close();
   }
   
   res_file << endl;
}

void test_t::start()
{
   ofstream result_file(output_file_name_);
   
   result_file << "%% fields: \"size\" ";
   
   for (size_t i = 0; i < tests_->size(); ++i)
      result_file << "\"" << tests_->at(i).name() << "\" ";
   
   result_file << endl << "%format of tests output (compute_time, mem_alloc_time, total_time)" << endl;
   
   result_file << endl;
      
   chrono::time_point<chrono::system_clock> start_test_time, finish_test_time;

   cout << "============= START GLOBAL TEST (" << test_name_ << ") =============" << endl << endl;

   start_test_time = chrono::system_clock::now();
   
   size_t size = 0;
   
   for (size_t test_idx = 0; test_idx < measurement_cnt_; ++test_idx)
   {
      cout << "---------test #" << test_idx << "---------" << endl;

      size = size_func_(test_idx, max_data_size_, measurement_cnt_);
      
      print_test_info_func_(size);

      test_data_t matrices = prepare_date_func_(size);
      
      test(tests_, matrices, size, result_file);
      
      matrices.reset();

      compare_res(tests_, size, goloden_test_idx_);
      
      remove_tmp_files(tests_);
      
      test_idx++;
      
      cout << endl;
   }

   finish_test_time = std::chrono::system_clock::now();
   
   cout << "============= FINISH GLOBAL TEST (" << test_name_ << ") =============" << endl << endl;
   
   size_t hours   = chrono::duration_cast<chrono::hours>  (finish_test_time - start_test_time).count();
   size_t minutes = chrono::duration_cast<chrono::minutes>(finish_test_time - start_test_time).count();
   size_t seconds = chrono::duration_cast<chrono::seconds>(finish_test_time - start_test_time).count();

   cout << "Test done for " << hours << "h, "
                            << minutes % 24 << "m, "
                            << seconds % 60 << "s"
                            << endl;
   
   result_file.close();
}
