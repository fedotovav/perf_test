integer function compare_2_arrays_double(n, a, b, max_diff, max_diff_idx) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_double), dimension(n), intent(in) :: a, b

    real(c_double), intent(out) :: max_diff
    integer(c_int), intent(out) :: max_diff_idx

    integer i, cur_idx, chunk, threads_cnt, thread_id

    real curr_diff

    curr_diff = 0
    max_diff = 0

    chunk = 10

    call omp_set_num_threads(4)

    !$omp parallel shared(a, b, curr_diff, max_diff, max_diff_idx, n, threads_cnt, chunk) private(thread_id, i)

    thread_id = omp_get_thread_num()

    if (thread_id .eq. 0) then
        threads_cnt = omp_get_num_threads()
    end if

    !$omp do schedule(static, chunk)
    do i = 1, n
        curr_diff = a(i) - b(i);

        if (max_diff < curr_diff) then
            max_diff = curr_diff
            max_diff_idx = i;
        else if (max_diff < -curr_diff) then
            max_diff = -curr_diff
            max_diff_idx = i;
        end if
    end do
    !omp end do
    
    !$omp end parallel

    compare_2_arrays_double = 0

end function compare_2_arrays_double

integer function compare_2_arrays_int(n, a, b, max_diff, max_diff_idx) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    integer(c_int), dimension(n), intent(in) :: a, b

    integer(c_int), intent(out) :: max_diff
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
    do i = 1, n
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

    compare_2_arrays_int = 0

end function compare_2_arrays_int

integer function compare_2_arrays_float(n, a, b, max_diff, max_diff_idx) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_float), dimension(n), intent(in) :: a, b

    real(c_float), intent(out) :: max_diff
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
    do i = 1, n
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

    compare_2_arrays_float = 0

end function compare_2_arrays_float

integer function fill_2_arrays_double(n, a, b) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_double), dimension(n), intent(inout) :: a, b

    integer i, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    !$omp parallel shared(a, b, threads_cnt, chunk) private(thread_id, i)

    thread_id = omp_get_thread_num()

    if (thread_id .eq. 0) then
        threads_cnt = omp_get_num_threads()
    end if

    !$omp do schedule(static, chunk)
    do i = 1, n
        call RANDOM_NUMBER(a(i))
        call RANDOM_NUMBER(b(i))
    end do
    !$omp end parallel

    fill_2_arrays_double = 0

end function fill_2_arrays_double

integer function fill_2_arrays_int(n, a, b) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    integer(c_int), dimension(n), intent(inout) :: a, b

    integer i, cur_idx, chunk, threads_cnt, thread_id
    
    real random_val

    chunk = 10

    !$omp parallel shared(a, b, threads_cnt, chunk) private(thread_id, i)

    thread_id = omp_get_thread_num()

    if (thread_id .eq. 0) then
        threads_cnt = omp_get_num_threads()
    end if

    !$omp do schedule(static, chunk)
    do i = 1, n
        call RANDOM_NUMBER(random_val)
        a(i) = random_val
        call RANDOM_NUMBER(random_val)
        b(i) = random_val
    end do
    !$omp end parallel

    fill_2_arrays_int = 0

end function fill_2_arrays_int

integer function fill_2_arrays_float(n, a, b) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_float), dimension(n), intent(inout) :: a, b

    integer i, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    !$omp parallel shared(a, b, threads_cnt, chunk) private(thread_id, i)

    thread_id = omp_get_thread_num()

    if (thread_id .eq. 0) then
        threads_cnt = omp_get_num_threads()
    end if

    !$omp do schedule(static, chunk)
    do i = 1, n
        call RANDOM_NUMBER(a(i))
        call RANDOM_NUMBER(b(i))
    end do
    !$omp end parallel

    fill_2_arrays_float = 0

end function fill_2_arrays_float
