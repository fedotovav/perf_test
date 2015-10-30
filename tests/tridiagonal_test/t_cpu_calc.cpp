#include <omp.h>

#include "tridiagonal_test.h"

template<typename T>
time_res_t calc_cpp( size_t size, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt, T * solution )
{
   time_res_t time_res;

   time_res.measure_start();
   
   T   * new_diag      = new T[size]
     , * new_right_prt = new T[size];
   
   for (size_t i = 0; i < size; ++i)
   {
      new_right_prt[i] = right_prt[i];
      new_diag[i]      = diag[i]; 
   }
   
   T tmp;
    
   for (size_t i = 1; i < size; ++i)
   {
      tmp = subdiag[i - 1] / new_diag[i - 1];
      
      new_diag[i]      -= superdiag[i - 1] * tmp;
      new_right_prt[i] -= new_right_prt[i - 1] * tmp;
   }
   
   solution[size - 1] = new_right_prt[size - 1] / new_diag[size - 1];
   
   for (int i = size - 2; i > -1; --i)
      solution[i] = (new_right_prt[i] - superdiag[i] * solution[i + 1]) / new_diag[i];
   
   delete[] new_diag;
   delete[] new_right_prt;
    
   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;
   
   return time_res;
}

time_res_t t_calc_cpu( size_t size, test_data_t<int> & data )
{
   size_t   threads_cnt = 4
          , size_cpu_per_thread = size / threads_cnt;

   time_res_t time_res;
   
   time_res.measure_start();

   size_t chunk = 10;   

   #pragma omp parallel num_threads(threads_cnt) shared(data, threads_cnt, size, chunk)
   {
      if (omp_get_thread_num() < threads_cnt - 1)
      {
         for (size_t i = 0; i < size_cpu_per_thread; ++i)
            calc_cpp<int>(500, data.get()[i].sub_diag_, data.get()[i].diag_, data.get()[i].super_diag_, data.get()[i].right_prt_, data.get()[i].solution_);
      }
      else if (omp_get_thread_num() == threads_cnt - 1)
      {
         for (size_t i = 0; i < size % (threads_cnt - 1); ++i)
            calc_cpp<int>(500, data.get()[i].sub_diag_, data.get()[i].diag_, data.get()[i].super_diag_, data.get()[i].right_prt_, data.get()[i].solution_);
      }
   }
   
   time_res.computing_time_ = time_res.measure_finish();
   
   return time_res;
}

time_res_t t_calc_cpu( size_t size, test_data_t<float> & data )
{
   size_t   threads_cnt = 4
          , size_cpu_per_thread = size / threads_cnt;

   time_res_t time_res;
   
   time_res.measure_start();
   
   size_t chunk = 10;   

   #pragma omp parallel num_threads(threads_cnt) shared(data, threads_cnt, size, chunk)
   {
      if (omp_get_thread_num() < threads_cnt - 1)
      {
         for (size_t i = 0; i < size_cpu_per_thread; ++i)
            calc_cpp<float>(1000, data.get()[i].sub_diag_, data.get()[i].diag_, data.get()[i].super_diag_, data.get()[i].right_prt_, data.get()[i].solution_);
      }
      else if (omp_get_thread_num() == threads_cnt - 1)
      {
         for (size_t i = 0; i < size % (threads_cnt - 1); ++i)
            calc_cpp<float>(1000, data.get()[i].sub_diag_, data.get()[i].diag_, data.get()[i].super_diag_, data.get()[i].right_prt_, data.get()[i].solution_);
      }
   }
   
   time_res.computing_time_ = time_res.measure_finish();
   
   return time_res;
}

time_res_t t_calc_cpu( size_t size, test_data_t<double> & data )
{
   size_t   threads_cnt = 4
          , size_cpu_per_thread = size / threads_cnt;

   time_res_t time_res;
   
   time_res.measure_start();
   
   size_t chunk = 10;   

   #pragma omp parallel num_threads(threads_cnt) shared(data, threads_cnt, size, chunk)
   {
      if (omp_get_thread_num() < threads_cnt - 1)
      {
         for (size_t i = 0; i < size_cpu_per_thread; ++i)
            calc_cpp<double>(1000, data.get()[i].sub_diag_, data.get()[i].diag_, data.get()[i].super_diag_, data.get()[i].right_prt_, data.get()[i].solution_);
      }
      else if (omp_get_thread_num() == threads_cnt - 1)
      {
         for (size_t i = 0; i < size % (threads_cnt - 1); ++i)
            calc_cpp<double>(1000, data.get()[i].sub_diag_, data.get()[i].diag_, data.get()[i].super_diag_, data.get()[i].right_prt_, data.get()[i].solution_);
      }
   }
   
   time_res.computing_time_ = time_res.measure_finish();
   
   return time_res;
}

template< typename T >
void subdiagonal_elements_removing_fvr( T * subdiag, T * diag, T * superdiag, T * right_prt
                                       ,size_t start_idx, size_t end_idx, int need_subdiag_transform )
{
   T line_coeff;
      
   if (need_subdiag_transform)
   {
      for (size_t i = start_idx + 1; i < end_idx; ++i)
      {
         line_coeff = subdiag[i - 1] / diag[i - 1];
         
         diag[i] -= superdiag[i - 1] * line_coeff;
         subdiag[i] = -subdiag[i - 2] * line_coeff;
         right_prt[i] -= right_prt[i - 1] * line_coeff;
      }
   }
   else
      for (size_t i = start_idx + 1; i < end_idx; ++i)
      {
         line_coeff = subdiag[i - 1] / diag[i - 1];

         diag[i] -= superdiag[i - 1] * line_coeff;
         right_prt[i] -= right_prt[i - 1] * line_coeff;
      }
}

template< typename T >
void subdiagonal_elements_removing_bvr( T * subdiag, T * diag, T * superdiag, T * right_prt
                                       ,size_t start_idx, size_t end_idx, int need_superdiag_transform )
{
   if (need_superdiag_transform)
   {
      T line_coeff;

      for (size_t i = start_idx - 2; i > end_idx - 1; --i)
      {
         line_coeff = superdiag[i] / diag[i + 1];

         superdiag[i] = -superdiag[i + 1] * line_coeff;
         right_prt[i] -= right_prt[i + 1] * line_coeff;
      }
   }
   else
      for (size_t i = start_idx - 2; i > end_idx - 1; --i)
         right_prt[i] -= right_prt[i + 1] * superdiag[i] / diag[i + 1];
}

template<typename T>
void solve_submatr( const T * subdiag, const T * diag, const T * superdiag, const T * right_prt, T * solution
                   ,size_t start_idx, size_t end_idx )

{
   T   left_val  = solution[start_idx - 1]
     , right_val = solution[end_idx + 1];
   
   for (size_t i = start_idx + 1; i < end_idx; ++i)
      solution[i] = (right_prt[i] - (subdiag[i] * left_val + superdiag[i] * right_val)) / diag[i];
}

template<typename T>
void solve_first_submatr( const T * subdiag, const T * diag, const T * superdiag, const T * right_prt, T * solution
                         ,size_t start_idx, size_t end_idx )

{
   T right_val = solution[end_idx + 1];
   
   for (size_t i = start_idx + 1; i < end_idx; ++i)
      solution[i] = (right_prt[i] - superdiag[i] * right_val) / diag[i];
}

template<typename T>
void solve_last_submatr( const T * subdiag, const T * diag, const T * superdiag, const T * right_prt, T * solution
                        ,size_t start_idx, size_t end_idx )
{
   T   left_val  = solution[start_idx - 1];
   
   for (size_t i = start_idx + 1; i < end_idx; ++i)
      solution[i] = (right_prt[i] - subdiag[i] * left_val) / diag[i];
}

template<typename T>
void copy_slae( int size, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt
               ,T * new_subdiag, T * new_diag, T * new_superdiag, T * new_right_prt )
{
   for (size_t i = 0; i < size - 1; ++i)
   {
      new_subdiag[i]   = subdiag[i];
      new_diag[i]      = diag[i];
      new_superdiag[i] = superdiag[i];
      new_right_prt[i] = right_prt[i];
   }
   
   new_diag[size - 1]      = diag[size - 1];
   new_right_prt[size - 1] = right_prt[size - 1];
}

template<typename T>
void create_support_slae( size_t size, size_t threads_cnt, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt
                         ,T * new_subdiag, T * new_diag, T * new_superdiag, T * new_right_prt )
{
   size_t   submatr_size  = size / threads_cnt
          , new_matr_size = 2 * threads_cnt;
   
   new_subdiag[0]   = 0;
   new_diag[0]      = diag[0]     , new_diag[1]      = diag[submatr_size - 1];
   new_superdiag[0] = superdiag[0], new_superdiag[1] = superdiag[submatr_size - 1];
   new_right_prt[0] = right_prt[0], new_right_prt[1] = right_prt[submatr_size - 1];

   size_t top_idx, bottom_idx, new_idx;
   
   for (size_t i = 1; i < threads_cnt - 1; ++i)
   {
      top_idx    = i * submatr_size;
      bottom_idx = (i + 1) * submatr_size - 1;
      new_idx    = 2 * i;
      
      new_subdiag[new_idx]   = subdiag[top_idx - 1];
      new_diag[new_idx]      = diag[top_idx];
      new_superdiag[new_idx] = superdiag[top_idx];
      new_right_prt[new_idx] = right_prt[top_idx];

      new_subdiag[new_idx + 1]   = subdiag[bottom_idx - 1];
      new_diag[new_idx + 1]      = diag[bottom_idx];
      new_superdiag[new_idx + 1] = superdiag[bottom_idx];
      new_right_prt[new_idx + 1] = right_prt[bottom_idx];
   }

   new_subdiag[new_matr_size - 3]   = subdiag[size - submatr_size] , new_subdiag[new_matr_size - 2]    = subdiag[size - 1];
   new_diag[new_matr_size - 2]      = diag[size - submatr_size]    , new_diag[new_matr_size - 1]       = diag[size - 1];
   new_superdiag[new_matr_size - 2] = 0;
   new_right_prt[new_matr_size - 2] = right_prt[size - submatr_size], new_right_prt[new_matr_size - 1] = right_prt[size - 1];
}

template<typename T>
void copy_sup_solution_to_solution( size_t size, size_t threads_cnt, const T * sup_solution, T * solution )
{
   size_t submatr_size = size / threads_cnt;
   
   size_t top_idx, bottom_idx, sol_idx;
   
   for (size_t i = 0; i < threads_cnt; ++i)
   {
      top_idx    = i * submatr_size;
      bottom_idx = (i + 1) * submatr_size - 1;
      sol_idx    = 2 * i;
      
      solution[top_idx]    = sup_solution[sol_idx];
      solution[bottom_idx] = sup_solution[sol_idx + 1];
   }
}

template<typename T>
void print_matr( size_t size, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt )
{
   cout << "----" << endl;
   
   cout << diag[0] << "\t" << superdiag[0];
   
   for (size_t j = 0; j < size - 2; ++j)
      cout << "\t";
   
   cout << right_prt[0] << endl;

   for (size_t i = 1; i < size - 1; ++i)
   {
      for (size_t j = 0; j < i - 1; ++j)
         cout << "\t";
      
      cout << subdiag[i - 1] << "\t" << diag[i] << "\t" << superdiag[i];
      
      for (size_t j = i + 1; j < size; ++j)
         cout << "\t";

      cout << right_prt[i] << endl;
   }

   for (size_t j = 0; j < size - 2; ++j)
      cout << "\t";
      
   cout << diag[size - 1] << "\t" << superdiag[size - 2] << "\t" << right_prt[size - 1] << endl;
   
   cout << "----" << endl;
}

template<typename T>
time_res_t calc_cpp_parallel( size_t size, const T * subdiag, const T * diag, const T * superdiag, const T * right_prt, T * solution )
{
   T   * new_subdiag   = new T[size - 1]
     , * new_diag      = new T[size]
     , * new_superdiag = new T[size - 1]
     , * new_right_prt = new T[size];
   
   copy_slae<T>(size, subdiag, diag, superdiag, right_prt, new_subdiag, new_diag, new_superdiag, new_right_prt);

   time_res_t time_res;

   time_res.measure_start();
   
   size_t threads_cnt = 3;
   
   size_t submatr_size = size / threads_cnt;
   
   if (size % threads_cnt)
      submatr_size++;
   
   size_t chunk = 10;
   
   //print_matr<T>(size, new_subdiag, new_diag, new_superdiag, new_right_prt);
   
   //cout << endl;
   
   #pragma omp parallel num_threads(threads_cnt) shared(new_diag, new_superdiag, new_subdiag, new_right_prt, threads_cnt, chunk)
   {
      if (omp_get_thread_num() == 0)
         subdiagonal_elements_removing_fvr<T>(new_subdiag, new_diag, new_superdiag, new_right_prt, omp_get_thread_num() * submatr_size
                                             ,(omp_get_thread_num() + 1) * submatr_size, 0);
      else if (omp_get_thread_num() < threads_cnt - 1)
         subdiagonal_elements_removing_fvr<T>(new_subdiag, new_diag, new_superdiag, new_right_prt, omp_get_thread_num() * submatr_size
                                             ,(omp_get_thread_num() + 1) * submatr_size, 1);
      else if (omp_get_thread_num() == threads_cnt - 1)
         subdiagonal_elements_removing_fvr<T>(new_subdiag, new_diag, new_superdiag, new_right_prt, omp_get_thread_num() * submatr_size, size, 1);
   }
   
   // print_matr<T>(size, new_subdiag, new_diag, new_superdiag, new_right_prt);

   #pragma omp parallel num_threads(threads_cnt) shared(new_diag, new_superdiag, new_subdiag, new_right_prt, threads_cnt, chunk)
   {
      if (omp_get_thread_num() == 0)
         subdiagonal_elements_removing_bvr<T>(new_subdiag, new_diag, new_superdiag, new_right_prt, (omp_get_thread_num() + 1) * submatr_size
                                    ,omp_get_thread_num() * submatr_size, 0);
      else if (omp_get_thread_num() < threads_cnt - 1)
         subdiagonal_elements_removing_bvr<T>(new_subdiag, new_diag, new_superdiag, new_right_prt, (omp_get_thread_num() + 1) * submatr_size
                                             ,omp_get_thread_num() * submatr_size, 1);
      else if (omp_get_thread_num() == threads_cnt - 1)
         subdiagonal_elements_removing_bvr<T>(new_subdiag, new_diag, new_superdiag, new_right_prt, size, omp_get_thread_num() * submatr_size, 1);
   }
   
   // print_matr<T>(size, new_subdiag, new_diag, new_superdiag, new_right_prt);

   // creation suppport matrix, that contain connect variables

   size_t sup_matr_size = 2 * threads_cnt;
   
   T   * sup_subdiag   = new T[sup_matr_size - 1]
     , * sup_diag      = new T[sup_matr_size]
     , * sup_superdiag = new T[sup_matr_size - 1]
     , * sup_right_prt = new T[sup_matr_size];
   
   create_support_slae(size, threads_cnt, new_subdiag, new_diag, new_superdiag, new_right_prt, sup_subdiag, sup_diag, sup_superdiag, sup_right_prt);
   
   //print_matr<T>(size, new_subdiag, new_diag, new_superdiag, new_right_prt);

   T * sup_solution = new T[sup_matr_size];
   
   calc_cpp<T>(sup_matr_size, sup_subdiag, sup_diag, sup_superdiag, sup_right_prt, sup_solution);
   
   copy_sup_solution_to_solution<T>(size, threads_cnt, sup_solution, solution);
   
   #pragma omp parallel num_threads(threads_cnt) shared(new_diag, new_superdiag, new_subdiag, new_right_prt, solution, threads_cnt, chunk)
   {
      if (omp_get_thread_num() == 0)
         solve_first_submatr(new_subdiag, new_diag, new_superdiag, new_right_prt, solution, 0, submatr_size - 1);
      else if (omp_get_thread_num() < threads_cnt - 1)
         solve_submatr(new_subdiag, new_diag, new_superdiag, new_right_prt, solution, omp_get_thread_num() * submatr_size, omp_get_thread_num() * submatr_size - 1);
      else if (omp_get_thread_num() == threads_cnt - 1)
         solve_last_submatr(new_subdiag, new_diag, new_superdiag, new_right_prt, solution, omp_get_thread_num() * submatr_size, size - 1);
   }

   time_res.computing_time_    = time_res.measure_finish();
   time_res.mem_allocate_time_ = 0;

   delete[] sup_solution;
   delete[] sup_right_prt;
   delete[] sup_superdiag;
   delete[] sup_diag;
   delete[] sup_subdiag;
   
   delete[] new_right_prt;
   delete[] new_superdiag;
   delete[] new_diag;
   delete[] new_subdiag;

   return time_res;
}

time_res_t t_calc_cpu_parallel( size_t size, test_data_t<int> & data )
{
   return calc_cpp_parallel<int>(size, data.get()->sub_diag_, data.get()->diag_, data.get()->super_diag_, data.get()->right_prt_, data.get()->solution_);
}

time_res_t t_calc_cpu_parallel( size_t size, test_data_t<float> & data )
{
   return calc_cpp_parallel<float>(size, data.get()->sub_diag_, data.get()->diag_, data.get()->super_diag_, data.get()->right_prt_, data.get()->solution_);
}

time_res_t t_calc_cpu_parallel( size_t size, test_data_t<double> & data )
{
   return calc_cpp_parallel<double>(size, data.get()->sub_diag_, data.get()->diag_, data.get()->super_diag_, data.get()->right_prt_, data.get()->solution_);
}
