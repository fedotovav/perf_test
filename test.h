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

using namespace std;

struct time_res_t
{
   time_res_t() :
        computing_time_(0)
      , mem_allocate_time_(0)
   {}
   
   size_t   computing_time_
          , mem_allocate_time_;
};

class test_unit_t
{
public:
   typedef time_res_t (* calc_func_t)( int size, const double * a, const double * b, double * c );
   
   calc_func_t calc_func_;
   string      test_name_;
   string      output_file_;
};