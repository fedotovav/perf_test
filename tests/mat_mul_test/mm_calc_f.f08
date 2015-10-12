integer function calc_four_thread_double(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_double), dimension(n * n), intent(in) :: a, b
    real(c_double), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(4)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_four_thread_double = 0

end function calc_four_thread_double

integer function calc_four_thread_float(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_float), dimension(n * n), intent(in) :: a, b
    real(c_float), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(4)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_four_thread_float = 0

end function calc_four_thread_float

integer function calc_four_thread_int(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    integer(c_int), dimension(n * n), intent(in) :: a, b
    integer(c_int), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(4)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_four_thread_int = 0

end function calc_four_thread_int

integer function calc_two_thread_double(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_double), dimension(n * n), intent(in) :: a, b
    real(c_double), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(2)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_two_thread_double = 0

end function calc_two_thread_double

integer function calc_two_thread_float(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_float), dimension(n * n), intent(in) :: a, b
    real(c_float), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(2)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_two_thread_float = 0

end function calc_two_thread_float

integer function calc_two_thread_int(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    integer(c_int), dimension(n * n), intent(in) :: a, b
    integer(c_int), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(2)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_two_thread_int = 0

end function calc_two_thread_int

integer function calc_one_thread_double(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_double), dimension(n * n), intent(in) :: a, b
    real(c_double), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(1)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_one_thread_double = 0

end function calc_one_thread_double

integer function calc_one_thread_float(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    real(c_float), dimension(n * n), intent(in) :: a, b
    real(c_float), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(1)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_one_thread_float = 0

end function calc_one_thread_float

integer function calc_one_thread_int(n, a, b, c) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int), value, intent(in) :: n

    integer(c_int), dimension(n * n), intent(in) :: a, b
    integer(c_int), dimension(n * n), intent(out) :: c

    integer i, j, k, cur_idx, chunk, threads_cnt, thread_id

    chunk = 10

    call omp_set_num_threads(1)

    !$omp parallel shared(a, b, c, threads_cnt, chunk) private(thread_id, i, j, k, cur_idx)

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

    calc_one_thread_int = 0

end function calc_one_thread_int
