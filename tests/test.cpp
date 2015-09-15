#include "test.h"

extern "C"
{
   int compare_2_arrays_double( int size, const double * a, const double * b, double * max_diff, int * max_diff_idx );
   int compare_2_arrays_int   ( int size, const int * a, const int * b, int * max_diff, int * max_diff_idx );
   int compare_2_arrays_float ( int size, const float * a, const float * b, float * max_diff, int * max_diff_idx );

   int fill_2_arrays_double( int size, const double * a, const double * b );
   int fill_2_arrays_int   ( int size, const int * a, const int * b );
   int fill_2_arrays_float ( int size, const float * a, const float * b );
}

void compare_2_arrays(int size, const double * a, const double * b, double * max_diff, int * max_diff_idx)
{
   compare_2_arrays_double(size, a, b, max_diff, max_diff_idx);
}

void compare_2_arrays(int size, const int * a, const int * b, int * max_diff, int * max_diff_idx)
{
   compare_2_arrays_int(size, a, b, max_diff, max_diff_idx);
}

void compare_2_arrays(int size, const float * a, const float * b, float * max_diff, int * max_diff_idx)
{
   compare_2_arrays_float(size, a, b, max_diff, max_diff_idx);
}

void fill_2_arrays(int size, const double * a, const double * b)
{
   fill_2_arrays_double(size, a, b);
}

void fill_2_arrays(int size, const int * a, const int * b)
{
   fill_2_arrays_int(size, a, b);
}

void fill_2_arrays(int size, const float * a, const float * b)
{
   fill_2_arrays_float(size, a, b);
}