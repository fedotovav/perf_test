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
   
   double   computing_time_
          , mem_allocate_time_;
   
private:
   chrono::time_point<chrono::system_clock> time_start_, time_finish_;
};

extern "C" 
{
   int compare_2_arrays( int size, const double * a, const double * b, double * max_diff, int * max_diff_idx );
   int fill_2_arrays( int size, const double * a, const double * b );
}

class test_unit_t
{
public:
   typedef time_res_t(* calc_func_t)  ( int size, const double * a, const double * b, double * c );
   //typedef void      (* limits_func_t)( size_t & matrix_size_limit );
   
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

typedef shared_ptr<vector<test_unit_t>> test_units_t;
typedef shared_ptr<double *>            test_data_t ;

class test_t
{
public:
   virtual size_t      size_by_measure_idx( size_t meas_idx );
   virtual void        print_measere_info ( size_t size );
   virtual test_data_t prepare_data       ( size_t size );
   virtual void        write_data_to_file ( ofstream & output_file, const double * data, size_t size );
   virtual void        clear_data         ( test_data_t data );

   test_t( int argc, char ** argv, const string & test_name, const test_units_t tests );
   
   virtual void run();
   
protected:
   
   virtual void test( test_data_t data, int size, ofstream & res_file, int need_check_file );

   test_units_t tests_;
   
   string   output_file_name_
          , test_name_;

   size_t   max_data_size_
          , matr_size_limit_ // maximum matrix size in bytes
          , measurement_cnt_;
   
   size_t goloden_test_idx_;
};