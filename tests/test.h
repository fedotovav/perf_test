#pragma once

#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <cstdio>

#include <boost/program_options.hpp>

using namespace std;

namespace po = boost::program_options;

class time_res_t
{
public:
   time_res_t() :
        computing_time_   (0)
      , mem_allocate_time_(0)
   {}
   
   void measure_start ()
   {
      time_start_ = chrono::system_clock::now();
   }
   
   double measure_finish()
   {
      time_finish_ = chrono::system_clock::now();

      return chrono::duration_cast<std::chrono::microseconds>(time_finish_ - time_start_).count() / 1000.;
   }
   
   double loop()
   {
      time_finish_ = chrono::system_clock::now();

      double duration = chrono::duration_cast<std::chrono::milliseconds>(time_finish_ - time_start_).count() / 1000.;
      
      time_start_ = chrono::system_clock::now();

      return duration;
   }
   
   time_res_t & operator+=( const time_res_t & time_res )
   {
      computing_time_    += time_res.computing_time_;
      mem_allocate_time_ += time_res.mem_allocate_time_;
      
      return *this;
   }

   time_res_t & operator/=( size_t val )
   {
      computing_time_    /= (double)val;
      mem_allocate_time_ /= (double)val;
      
      return *this;
   }
   
   double   computing_time_
          , mem_allocate_time_;
   
private:
   chrono::time_point<chrono::system_clock> time_start_, time_finish_;
};

template<typename T> class test_unit_t
{
public:
   typedef time_res_t(* calc_func_t)  ( int size, const T * a, const T * b, const T * c, const T * d, T * e );
   
   test_unit_t(const string & test_name, const calc_func_t & calc_func, const string & output_file_name
              ,const string & cmd_line_par_name, int is_fake_test = 0 );
   
   const string & check_file  () const;
   const string & name        () const;
   const string & cmd_line_par() const;
   
   int is_fake() const;

   calc_func_t calc_func;
   
   test_unit_t & operator=(test_unit_t const & test_unit)
   {
      calc_func          = test_unit.calc_func;
      test_name_         = test_unit.test_name_;
      output_file_       = test_unit.output_file_;
      cmd_line_par_name_ = test_unit.cmd_line_par_name_;
      is_fake_test_      = test_unit.is_fake_test_;
      
      return *this;
   }
   
private:
   string   test_name_
          , output_file_
          , cmd_line_par_name_;
   
   int is_fake_test_;
};

template<typename T> class test_t
{
public:
   typedef shared_ptr<T *>                    test_data_t;
   typedef shared_ptr<vector<test_unit_t<T>>> test_units_t;

   virtual size_t      size_by_measure_idx( size_t meas_idx );
   virtual void        print_measere_info ( size_t size );
   virtual test_data_t prepare_data       ( size_t size );
   virtual void        write_data_to_file ( ofstream & output_file, const test_data_t data, size_t size );
   virtual void        clear_data         ( test_data_t data );
   virtual void        compare_res        ( size_t size, size_t golden_test_idx );
   virtual void        remove_tmp_files   ( const test_units_t tests );

   test_t( int argc, char ** argv, const string & test_name, const test_units_t tests );
   
   virtual void run();
   
protected:
   
   virtual void measurement( test_data_t data, int size, ofstream & res_file, int need_check_file );

   test_units_t tests_;
   
   string   output_file_name_
          , test_name_;

   size_t   max_data_size_
          , matr_size_limit_ // maximum matrix size in bytes
          , measurement_cnt_
          , measure_precision_;
   
   int goloden_test_idx_;
};

template<typename T>
test_unit_t<T>::test_unit_t( const string & test_name, const calc_func_t & func, const string & output_file_name
                            ,const string & cmd_line_par_name, int is_fake_test ) :
     test_name_        (test_name)
   , calc_func         (func)
   , output_file_      (output_file_name)
   , cmd_line_par_name_(cmd_line_par_name)
   , is_fake_test_     (is_fake_test)
{
}

template<typename T>
const string & test_unit_t<T>::check_file() const
{
   return output_file_;
}

template<typename T>
const string & test_unit_t<T>::name() const
{
   return test_name_;
}

template<typename T>
const string & test_unit_t<T>::cmd_line_par() const
{
   return cmd_line_par_name_;
}

template<typename T>
int test_unit_t<T>::is_fake() const
{
   return is_fake_test_;
}

template<typename T>
test_t<T>::test_t( int argc, char ** argv, const string & test_name, test_units_t tests ) :
     max_data_size_    (0)
   , matr_size_limit_  (0)
   , measurement_cnt_  (0)
   , output_file_name_ ("default")
   , goloden_test_idx_ (-1)
   , test_name_        (test_name)
   , measure_precision_(1)
{
   po::options_description head_description("Computing technologies performance test\n Author: Anton Fedotov");

   po::options_description general_options("General options");
   general_options.add_options()
      ("help,h", "show help")
      ("measurement-cnt,c", po::value<size_t>(&measurement_cnt_), "set measurement count")
      ("output-file,o", po::value<string>(&output_file_name_), "test result output file name")
      ("max-matr-size,s", po::value<size_t>(&max_data_size_), "maximum matrix size")
      ("measure-precision,p", po::value<size_t>(&measure_precision_), "count of repeats unit test runin for get average");

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

   if (!tests_->size())
   {
      cerr << "Bad program options usage! Example of using options:" << endl;
      cerr << "./<program name> -o test.res -c 10 -s 1000 --<some test name>" << endl << endl;

      throw head_description;
   }
   
   size_t tests_cnt = tests_->size();
   
   if (!vm.count("all-tests-units"))
      for (size_t i = 0, cur_test_idx = 0; i < tests_cnt; ++i)
         if (!vm.count(tests->at(cur_test_idx).cmd_line_par()) && !tests->at(cur_test_idx).is_fake())
            tests_->erase(tests_->begin() + cur_test_idx);
         else
            cur_test_idx++;
      
   if (vm.count("golden-test"))
      for (size_t i = 0; i < tests_->size(); ++i)
         if (tests->at(i).cmd_line_par() == vm["golden-test"].as<string>())
            goloden_test_idx_ = i;
}

void compare_2_arrays(int size, const double * a, const double * b, double * max_diff, int * max_diff_idx);
void compare_2_arrays(int size, const int * a, const int * b, int * max_diff, int * max_diff_idx);
void compare_2_arrays(int size, const float * a, const float * b, float * max_diff, int * max_diff_idx);

void fill_2_arrays(int size, const double * a, const double * b);
void fill_2_arrays(int size, const int * a, const int * b);
void fill_2_arrays(int size, const float * a, const float * b);

template<typename T>
void test_t<T>::compare_res( size_t size, size_t golden_test_idx )
{
   double   * ideal_res = new double[size]
          , * other_res;

   ifstream ideal_file(tests_->at(golden_test_idx).check_file());

   for (int i = 0; i < size; ++i)
      ideal_file >> ideal_res[i];

   double max_diff;
   
   int max_diff_idx;
   
   for (int k = 0; k < tests_->size(); ++k)
   {
      if (k == golden_test_idx)
         continue;
      
      ifstream input_file(tests_->at(k).check_file());
      
      other_res = new double[size];

      for (int i = 0; i < size; ++i)
         input_file >> other_res[i];

      compare_2_arrays(size, ideal_res, other_res, &max_diff, &max_diff_idx);

      if (max_diff != 0)
      {
         cout << tests_->at(k).name() << " result output is incorrect! (maximum difference: " << max_diff << ", index:" << max_diff_idx << ")"<< endl;
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

template<typename T>
void test_t<T>::write_data_to_file( ofstream & output_file, const test_data_t data, size_t size )
{
   for (int i = 0; i < size; ++i)
      output_file << data.get()[0][i] << " ";
}

template< typename T >
T check( size_t size, const T * sub, const T * diag, const T * super, const T * right, const T * solution ) 
{
   T   max_err = right[0] - (diag[0] * solution[0] + super[0] * solution[1])
     , tmp_val;
   
   for (size_t i = 1; i < size - 1; ++i)
   {
      tmp_val = right[i] - (sub[i - 1] * solution[i - 1] + diag[i] * solution[i] + super[i] * solution[i + 1]);

      if (tmp_val > max_err)
         max_err = tmp_val;
   }
   
   return max_err;
}

template<typename T>
void test_t<T>::measurement( test_data_t data, int size, ofstream & res_file, int need_check_file )
{
   res_file << size << "\t";

   time_res_t duration;
   
   for (size_t i = 0; i < tests_->size(); ++i)
   {
      if (!tests_->at(i).is_fake())
         cout << "call " << tests_->at(i).name() << endl;

      duration = tests_->at(i).calc_func(size, data.get()[0], data.get()[1], data.get()[2], data.get()[3], data.get()[4]);

      if (tests_->at(i).is_fake())
         continue;
      
      for (size_t k = 0; k < measure_precision_ - 1; ++k)
         duration += tests_->at(i).calc_func(size, data.get()[0], data.get()[1], data.get()[2], data.get()[3], data.get()[4]);
         
      duration /= measure_precision_;

      cout << "computation time: " << duration.computing_time_ << " ms" << endl;
      cout << "memory allocation time: " << duration.mem_allocate_time_ << " ms" << endl;
      cout << "total time: " << duration.mem_allocate_time_ + duration.computing_time_ << " ms" << endl;
      
      res_file << duration.computing_time_ << "\t" << duration.mem_allocate_time_
               << "\t" << duration.mem_allocate_time_ + duration.computing_time_ << "\t";
      
      cout << "Maximum error: " << check<T>(size, data.get()[0], data.get()[1], data.get()[2], data.get()[3], data.get()[4]) << endl << endl;

//      if (need_check_file)
//      {
//         ofstream output_file(tests_->at(i).check_file());
//
//         write_data_to_file(output_file, data, size);
//
//         output_file.close();
//      }
   }
   
   res_file << endl;
}

template<typename T>
size_t test_t<T>::size_by_measure_idx( size_t meas_idx )
{
   static int   size_incr = max_data_size_ / measurement_cnt_
              , size = 0;
   
   size += size_incr;

   return size;
}

template<typename T>
void test_t<T>::print_measere_info( size_t size )
{
   cout << "measurement typical size: " << size << " elements, " << " (" << sizeof(T) * size / 1048576. << " mb)" << endl;
}

template<typename T>
typename test_t<T>::test_data_t test_t<T>::prepare_data( size_t size )
{
   test_data_t data(new T*[3]);

   data.get()[0] = new T[size];
   data.get()[1] = new T[size];
   data.get()[2] = new T[size];
   
   fill_2_arrays(size, data.get()[0], data.get()[1]);
   
   return data;
}

template<typename T>
void test_t<T>::clear_data( test_data_t data )
{
   delete[] data.get()[0];
   delete[] data.get()[1];
   delete[] data.get()[2];
   
   data.reset();
}

template<typename T>
void test_t<T>::remove_tmp_files( const test_units_t tests )
{
   for (size_t i = 0; i < tests->size(); ++i)
      if (!tests->at(i).check_file().empty())
         remove(tests->at(i).check_file().c_str());
}

template<typename T>
void test_t<T>::run()
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
      
      measurement(data, size, result_file, need_check_file);
      
      clear_data(data);

      if (goloden_test_idx_ != -1)
         compare_res(size, goloden_test_idx_);
      
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
