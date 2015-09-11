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

struct time_res_t
{
   time_res_t() :
        computing_time_   (0)
      , mem_allocate_time_(0)
   {}
   
   size_t   computing_time_
          , mem_allocate_time_;
};

class test_unit_t
{
public:
   typedef time_res_t(* calc_func_t)  ( int size, const double * a, const double * b, double * c );
   //typedef void      (* limits_func_t)( size_t & matrix_size_limit );
   
   test_unit_t(const string & test_name, const calc_func_t & calc_func, const string & output_file_name
              ,const string & cmd_line_par_name, int is_golden_test = 0 );
   
   const string & check_file  () const;
   const string & name        () const;
   const string & cmd_line_par() const;

   int is_golden_test() const;

   calc_func_t calc_func;
   
   test_unit_t & operator=(test_unit_t const & test_unit)
   {
      calc_func          = test_unit.calc_func;
      test_name_         = test_unit.test_name_;
      output_file_       = test_unit.output_file_;
      cmd_line_par_name_ = test_unit.cmd_line_par_name_;
      is_golden_test_    = test_unit.is_golden_test_;
      
      return *this;
   }
   
private:
   string   test_name_
          , output_file_
          , cmd_line_par_name_;
   
   int is_golden_test_;
};

typedef shared_ptr<vector<test_unit_t>> test_units_t;
typedef shared_ptr<double *>            test_data_t ;

class test_t
{
public:
   typedef size_t      (* size_func_t)                 ( size_t test_idx, size_t max_data_size, size_t measurement_cnt );
   typedef void        (* print_test_info_func_t)      ( size_t size );
   typedef test_data_t (* prepare_date_func_t)         ( size_t size );
   typedef void        (* write_data_to_file_func_t)   ( ofstream & output_file, const double * data, int size );

   test_t( int argc, char ** argv, const string & test_name, const test_units_t tests, size_func_t size_func
          ,print_test_info_func_t print_test_info_func, prepare_date_func_t prepare_date_func );
   
   void start();
   
private:
   test_units_t tests_;
   
   size_func_t            size_func_;
   print_test_info_func_t print_test_info_func_;
   prepare_date_func_t    prepare_date_func_;
   
   string   output_file_name_
          , test_name_;

   size_t   max_data_size_
          , matr_size_limit_ // maximum matrix size in bytes
          , measurement_cnt_;
   
   int goloden_test_idx_;
};