#include "test.h"

test_unit_t::test_unit_t( const string & test_name, const calc_func_t & func, const string & output_file_name
                         ,const string & cmd_line_par_name, int is_fake_test ) :
     test_name_        (test_name)
   , calc_func         (func)
   , output_file_      (output_file_name)
   , cmd_line_par_name_(cmd_line_par_name)
   , is_fake_test_     (is_fake_test)
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

int test_unit_t::is_fake() const
{
   return is_fake_test_;
}

test_t::test_t( int argc, char ** argv, const string & test_name, test_units_t tests ) :
     max_data_size_       (0)
   , matr_size_limit_     (0)
   , measurement_cnt_     (0)
   , output_file_name_    ("default")
   , goloden_test_idx_    (-1)
   , test_name_           (test_name)
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

   tests_options.add_options()("golden-test", po::value<string>(), "test with exact result that will use for check other tests results");

   for (size_t i = 0; i < tests.get()->size(); ++i)
      if (!tests->at(i).is_fake())
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
      cerr << "./<program name> -o test.plt -c 10 -s 1000 --<some test(s) name(s)> --golden-test=<some test name>" << endl << endl;
      
      throw head_description;
   }
   
   if (tests)
      tests_ = tests;
   else
   {
      cerr << "No one test found!" << endl;

      throw;
   }

   size_t tests_cnt = tests_->size();
   
   if (!tests_->size())
   {
      cerr << "Bad program options usage! Example of using options:" << endl;
      cerr << "./<program name> -o test.res -c 10 -s 1000 --<some test name>" << endl << endl;

      throw head_description;
   }
   
   if (!vm.count("all-tests-units"))
      for (size_t i = 0, cur_test_idx = 0; i < tests_cnt; ++i)
         if (!vm.count(tests->at(cur_test_idx).cmd_line_par()) && !tests->at(cur_test_idx).is_fake())
            tests_->erase(tests_->begin() + cur_test_idx);
         else
            cur_test_idx++;
      
   if (vm.count("golden-test"))
      for (size_t i = 0; i < tests_cnt; ++i)
         if (tests->at(i).cmd_line_par() == vm["golden-test"].as<string>())
            goloden_test_idx_ = i;
}

// return max difference
double compare_res( const test_units_t tests, int size, int golden_test_idx )
{
   double   * ideal_res = new double[size]
          , * other_res;

   ifstream ideal_file (tests->at(golden_test_idx).check_file());

   for (int i = 0; i < size; ++i)
      ideal_file >> ideal_res[i];

   double max_diff;
   
   int max_diff_idx;
   
   for (int k = 0; k < tests->size(); ++k)
   {
      if (k == golden_test_idx)
         continue;
      
      ifstream input_file(tests->at(k).check_file());
      
      other_res = new double[size];

      for (int i = 0; i < size; ++i)
         input_file >> other_res[i];

      compare_2_arrays(size, ideal_res, other_res, &max_diff, &max_diff_idx);

      if (max_diff != 0)
      {
         cout << tests->at(k).name() << " result output is incorrect! (maximum difference: " << max_diff << ", index:" << max_diff_idx << ")"<< endl;
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

void test_t::write_data_to_file( ofstream & output_file, const double * data, size_t size )
{
   for (int i = 0; i < size; ++i)
      output_file << data[i] << " ";
}

void test_t::test( test_data_t data, int size, ofstream & res_file, int need_check_file )
{
   res_file << size << "\t";

   time_res_t duration;
   
   for (size_t i = 0; i < tests_->size(); ++i)
   {
      if (!tests_->at(i).is_fake())
         cout << "call " << tests_->at(i).name() << endl;

      duration = tests_->at(i).calc_func(size, data.get()[0], data.get()[1], data.get()[2]);

      if (!tests_->at(i).is_fake())
      {
         cout << "computation time: " << duration.computing_time_ << " ms" << endl;
         cout << "memory allocation time: " << duration.mem_allocate_time_ << " ms" << endl;
         cout << "total time: " << duration.mem_allocate_time_ + duration.computing_time_ << " ms" << endl << endl;
      }
      
      if (!tests_->at(i).is_fake())
         res_file << duration.computing_time_ << "\t" << duration.mem_allocate_time_
                  << "\t" << duration.mem_allocate_time_ + duration.computing_time_ << "\t";
      
      if (!tests_->at(i).is_fake())
         if (need_check_file)
         {
            ofstream output_file(tests_->at(i).check_file());

            write_data_to_file(output_file, data.get()[2], size);

            output_file.close();
         }
   }
   
   res_file << endl;
}

size_t test_t::size_by_measure_idx( size_t meas_idx )
{
   static int   size_incr = max_data_size_ / measurement_cnt_
              , size = 0;
   
   size += size_incr;

   return size;
}

void test_t::print_measere_info( size_t size )
{
   cout << "measurement typical size: " << size << " elements, " << " (" << sizeof(double) * size / 1048576. << " mb)" << endl;
}

// fix it
test_data_t test_t::prepare_data( size_t size )
{
   test_data_t data(new double*[3]);

   data.get()[0] = new double[size];
   data.get()[1] = new double[size];
   data.get()[2] = new double[size];
   
   fill_2_arrays(size, data.get()[0], data.get()[1]);
   
   return data;
}

void test_t::clear_data( test_data_t data )
{
   delete[] data.get()[0];
   delete[] data.get()[1];
   delete[] data.get()[2];
   
   data.reset();
}

void remove_tmp_files( const test_units_t tests )
{
   for (size_t i = 0; i < tests->size(); ++i)
      if (!tests->at(i).check_file().empty())
         remove(tests->at(i).check_file().c_str());
}

void test_t::run()
{
   ofstream result_file(output_file_name_);
   
   result_file << endl << "Variables = \"size\", ";
   
   for (size_t i = 0; i < tests_->size(); ++i)
      if (!tests_->at(i).is_fake())
         result_file << "\"" << tests_->at(i).name() << " compute time\", "
                     << "\"" << tests_->at(i).name() << " mem alloc time\", "
                     << "\"" << tests_->at(i).name() << " total time\", ";
   
   result_file << endl << "Zone i=" << measurement_cnt_ << ", j=1, k=1";
   
   result_file << endl;
      
   chrono::time_point<chrono::system_clock> start_test_time, finish_test_time;

   cout << "============= START GLOBAL TEST (" << test_name_ << ") =============" << endl << endl;

   start_test_time = chrono::system_clock::now();
   
   size_t size = 0;
   
   int need_check_file = 1;
   
   for (size_t test_idx = 0; test_idx < measurement_cnt_; ++test_idx)
   {
      cout << "--------- measurement #" << test_idx << " ---------" << endl;

      size = size_by_measure_idx(test_idx);
      
      print_measere_info(size);

      test_data_t data = prepare_data(size);
      
      if (goloden_test_idx_ < 0)
         need_check_file = 0;
      
      test(data, size, result_file, need_check_file);
      
      clear_data(data);

      if (goloden_test_idx_ != -1)
         compare_res(tests_, size, goloden_test_idx_);
      
      if (need_check_file)
         remove_tmp_files(tests_);
      
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
