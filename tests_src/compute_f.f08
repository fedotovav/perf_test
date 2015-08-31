integer function calc_four_thread_f(n, a, b, c) bind(C)
use iso_c_binding
use omp_lib
implicit none
   integer(c_int), value, intent(in) :: n

   real(c_double), dimension(n * n), intent(in)  :: a, b
   real(c_double), dimension(n * n), intent(out) :: c

   integer i, j, k, cur_idx, chunk, threads_cnt, thread_id
   
   chunk = 10

   call omp_set_num_threads(4)
   
   !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k)
   
   thread_id = omp_get_thread_num()
   
   if (thread_id .eq. 0) then
      threads_cnt = omp_get_num_threads()
   end if

   !$omp do schedule(static, chunk)
   do i = 1, n
      do j = 1, n
         cur_idx = (i - 1) * n + j
         c(cur_idx) = 0
         
         do k = 1, n
            c(cur_idx) = c(cur_idx) + a((i - 1) * n + k) * b((k - 1) * n + j)
         end do
      end do
   end do
   !$omp end parallel

   calc_four_thread_f = 0

end function calc_four_thread_f

integer function calc_two_thread_f(n, a, b, c) bind(C)
use iso_c_binding
use omp_lib
implicit none
   integer(c_int), value, intent(in) :: n

   real(c_double), dimension(n * n), intent(in)  :: a, b
   real(c_double), dimension(n * n), intent(out) :: c

   integer i, j, k, cur_idx, chunk, threads_cnt, thread_id
   
   chunk = 10
   
   call omp_set_num_threads(2)

   !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k)
   
   thread_id = omp_get_thread_num()
   
   if (thread_id .eq. 0) then
      threads_cnt = omp_get_num_threads()
   end if

   !$omp do schedule(static, chunk)
   do i = 1, n
      do j = 1, n
         cur_idx = (i - 1) * n + j
         c(cur_idx) = 0
         
         do k = 1, n
            c(cur_idx) = c(cur_idx) + a((i - 1) * n + k) * b((k - 1) * n + j)
         end do
      end do
   end do
   !$omp end parallel

   calc_two_thread_f = 0

end function calc_two_thread_f

integer function calc_one_thread_f(n, a, b, c) bind(C)
use iso_c_binding
implicit none
   integer(c_int), value, intent(in) :: n

   real(c_double), dimension(n * n), intent(in)  :: a, b
   real(c_double), dimension(n * n), intent(out) :: c

   integer i, j, k, cur_idx

   do i = 1, n
      do j = 1, n
         cur_idx = (i - 1) * n + j
         c(cur_idx) = 0
         
         do k = 1, n
            c(cur_idx) = c(cur_idx) + a((i - 1) * n + k) * b((k - 1) * n + j)
         end do
      end do
   end do
   
   calc_one_thread_f = 0

end function calc_one_thread_f

integer function compare_2_matr(n, a, b, max_diff, max_diff_idx) bind(C)
use iso_c_binding
use omp_lib
implicit none
   integer(c_int), value, intent(in) :: n

   real(c_double), dimension(n * n), intent(in)  :: a, b

   real(c_double), intent(out) :: max_diff
   integer(c_int), intent(out) :: max_diff_idx

   integer i, cur_idx, chunk, threads_cnt, thread_id
   
   real curr_diff
   
   chunk = 10
   
   !$omp parallel shared(a, b, threads_cnt, chunk) private(thread_id, i)
   
   thread_id = omp_get_thread_num()
   
   if (thread_id .eq. 0) then
      threads_cnt = omp_get_num_threads()
   end if

   curr_diff = 0
   max_diff = 0
   
   !$omp do schedule(static, chunk)
   do i = 1, n * n
      curr_diff = a(i) - b(i);

      if (max_diff < curr_diff) then
         max_diff = curr_diff
         max_diff_idx = i;
      else if (max_diff < -curr_diff) then
         max_diff = -curr_diff
         max_diff_idx = i;
      end if
   end do
   !$omp end parallel

   compare_2_matr = 0

end function compare_2_matr

integer function fill_2_matr(n, a, b) bind(C)
use iso_c_binding
use omp_lib
implicit none
   integer(c_int), value, intent(in) :: n

   real(c_double), dimension(n * n), intent(inout)  :: a, b

   integer i, cur_idx, chunk, threads_cnt, thread_id
   
   chunk = 10
   
   !$omp parallel shared(a, b, threads_cnt, chunk) private(thread_id, i)
   
   thread_id = omp_get_thread_num()
   
   if (thread_id .eq. 0) then
      threads_cnt = omp_get_num_threads()
   end if

   !$omp do schedule(static, chunk)
   do i = 1, n * n
      call RANDOM_NUMBER(a(i))
      call RANDOM_NUMBER(b(i))
   end do
   !$omp end parallel

   fill_2_matr = 0

end function fill_2_matr
