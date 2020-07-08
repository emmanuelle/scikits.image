#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport malloc, free

cnp.import_array()

cdef inline dtype_t _max(dtype_t a, dtype_t b) nogil:
    return a if a >= b else b


cdef inline dtype_t _min(dtype_t a, dtype_t b) nogil:
    return a if a <= b else b


cdef inline void histogram_increment(Py_ssize_t[::1] histo, double* pop,
                                     dtype_t value) nogil:
    histo[value] += 1
    pop[0] += 1


cdef inline void histogram_decrement(Py_ssize_t[::1] histo, double* pop,
                                     dtype_t value) nogil:
    histo[value] -= 1
    pop[0] -= 1


cdef inline void _count_attack_border_elements(char[:, :, ::1] selem,
                                               Py_ssize_t [:, :, ::1] se,
                                               Py_ssize_t [::1] num_se,
                                               Py_ssize_t splanes,
                                               Py_ssize_t srows,
                                               Py_ssize_t scols, 
                                               Py_ssize_t centre_p,
                                               Py_ssize_t centre_r,
                                               Py_ssize_t centre_c):

    # build attack and release borders by using difference along axis
    t = np.dstack((selem, np.zeros((selem.shape[0], selem.shape[1], 1))))
    cdef unsigned char[:, :, :] t_e = (np.diff(t, axis=2) < 0).view(np.uint8)

    t = np.dstack((np.zeros((selem.shape[0], selem.shape[1], 1)), selem))
    cdef unsigned char[:, :, :] t_w = (np.diff(t, axis=2) > 0).view(np.uint8)

    t = np.hstack((selem, np.zeros((selem.shape[0], 1, selem.shape[2]))))
    cdef unsigned char[:, :, :] t_s = (np.diff(t, axis=1) < 0).view(np.uint8)

    t = np.hstack((np.zeros((selem.shape[0], 1, selem.shape[2])), selem))
    cdef unsigned char[:, :, :] t_n = (np.diff(t, axis=1) > 0).view(np.uint8)

    for r in range(srows):
        for c in range(scols):
            for p in range(splanes):
                if t_e[p, r, c]:
                    se[0, 0, num_se[0]] = p - centre_p
                    se[0, 1, num_se[0]] = r - centre_r
                    se[0, 2, num_se[0]] = c - centre_c
                    num_se[0] += 1
                if t_n[p, r, c]:
                    se[1, 0, num_se[1]] = p - centre_p
                    se[1, 1, num_se[1]] = r - centre_r
                    se[1, 2, num_se[1]] = c - centre_c
                    num_se[1] += 1
                if t_w[p, r, c]:
                    se[2, 0, num_se[2]] = p - centre_p
                    se[2, 1, num_se[2]] = r - centre_r
                    se[2, 2, num_se[2]] = c - centre_c
                    num_se[2] += 1
                if t_s[p, r, c]:
                    se[3, 0, num_se[3]] = p - centre_p
                    se[3, 1, num_se[3]] = r - centre_r
                    se[3, 2, num_se[3]] = c - centre_c
                    num_se[3] += 1


cdef inline void _build_initial_histogram_from_neighborhood(dtype_t[:, :, ::1] image,
                                                            char[:, :, ::1] selem,
                                                            Py_ssize_t [::1] histo,
                                                            double* pop,
                                                            char* mask_data,
                                                            Py_ssize_t p,
                                                            Py_ssize_t planes,
                                                            Py_ssize_t rows,
                                                            Py_ssize_t cols, 
                                                            Py_ssize_t splanes,
                                                            Py_ssize_t srows,
                                                            Py_ssize_t scols, 
                                                            Py_ssize_t centre_p,
                                                            Py_ssize_t centre_r,
                                                            Py_ssize_t centre_c):
    for r in range(srows):
        for c in range(scols):
            for j in range(splanes):
                pp = j - centre_p + p
                rr = r - centre_r
                cc = c - centre_c

                if selem[j, r, c]:
                    if is_in_mask_3D(rows, cols, planes, rr, cc, pp,
                                     mask_data):
                        # histogram_increment(histo, pop, image[pp, rr, cc])
                        histo[image[pp, rr, cc]] += 1
                        pop[0] += 1


cdef inline void _update_histogram(dtype_t[:, :, ::1] image,
                                   Py_ssize_t [:, :, ::1] se,
                                   Py_ssize_t [::1] num_se,
                                   Py_ssize_t [::1] histo,
                                   double* pop, char* mask_data,
                                   Py_ssize_t p, Py_ssize_t r, Py_ssize_t c,
                                   Py_ssize_t planes, Py_ssize_t rows,
                                   Py_ssize_t cols,
                                   Py_ssize_t axis_inc):
    # Increment histogram
    for j in range(num_se[axis_inc]):
        pp = p + se[axis_inc, 0, j]
        rr = r + se[axis_inc, 1, j]
        cc = c + se[axis_inc, 2, j]
        if is_in_mask_3D(rows, cols, planes, rr, cc, pp,
                         mask_data):
            histo[image[pp, rr, cc]] += 1
            pop[0] += 1

    # Decrement histogram
    axis_dec = (axis_inc + 2) % 4
    for j in range(num_se[axis_dec]):
        pp = p + se[axis_dec, 0, j]
        rr = r + se[axis_dec, 1, j]
        cc = c + se[axis_dec, 2, j]
        if axis_dec == 2:
            cc -= 1
        elif axis_dec == 1:
            rr -= 1
        elif axis_dec == 0:
            cc += 1
        if is_in_mask_3D(rows, cols, planes, rr, cc, pp,
                         mask_data):
            histo[image[pp, rr, cc]] -= 1
            pop[0] -= 1


cdef inline char is_in_mask(Py_ssize_t rows, Py_ssize_t cols,
                            Py_ssize_t r, Py_ssize_t c,
                            char* mask) nogil:
    """Check whether given coordinate is within image and mask is true."""
    if r < 0 or r > rows - 1 or c < 0 or c > cols - 1:
        return 0
    else:
        if not mask:
            return 1
        elif mask[r * cols + c]:
            return 1
        else:
            return 0


cdef inline char is_in_mask_3D(Py_ssize_t planes, Py_ssize_t rows,
                               Py_ssize_t cols, Py_ssize_t r,
                               Py_ssize_t c, Py_ssize_t p,
                               char* mask) nogil:
    """Check whether given coordinate is within image and mask is true."""
    if (r < 0 or r > rows - 1 or c < 0 or c > cols - 1 or
            p < 0 or p > planes - 1):
        return 0
    else:
        if not mask:
            return 1
        return mask[p * rows * cols + r * cols + c]


cdef void _core(void kernel(dtype_t_out*, Py_ssize_t, Py_ssize_t[::1], double,
                            dtype_t, Py_ssize_t, Py_ssize_t, double,
                            double, Py_ssize_t, Py_ssize_t) nogil,
                dtype_t[:, ::1] image,
                char[:, ::1] selem,
                char[:, ::1] mask,
                dtype_t_out[:, :, ::1] out,
                signed char shift_x, signed char shift_y,
                double p0, double p1,
                Py_ssize_t s0, Py_ssize_t s1,
                Py_ssize_t n_bins) except *:
    """Compute histogram for each pixel neighborhood, apply kernel function and
    use kernel function return value for output image.
    """

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]
    cdef Py_ssize_t srows = selem.shape[0]
    cdef Py_ssize_t scols = selem.shape[1]
    cdef Py_ssize_t odepth = out.shape[2]

    cdef Py_ssize_t centre_r = <Py_ssize_t>(selem.shape[0] / 2) + shift_y
    cdef Py_ssize_t centre_c = <Py_ssize_t>(selem.shape[1] / 2) + shift_x

    # check that structuring element center is inside the element bounding box
    assert centre_r >= 0
    assert centre_c >= 0
    assert centre_r < srows
    assert centre_c < scols

    cdef Py_ssize_t mid_bin = n_bins / 2

    # define pointers to the data
    cdef char * mask_data = NULL
    if mask is not None:
        mask_data = &mask[0, 0]

    # define local variable types
    cdef Py_ssize_t r, c, rr, cc, s, j, value, local_max, i, even_row

    # number of pixels actually inside the neighborhood (double)
    cdef double pop = 0

    # build attack and release borders by using difference along axis
    t = np.hstack((selem, np.zeros((selem.shape[0], 1))))
    cdef unsigned char[:, :] t_e = (np.diff(t, axis=1) < 0).view(np.uint8)

    t = np.hstack((np.zeros((selem.shape[0], 1)), selem))
    cdef unsigned char[:, :] t_w = (np.diff(t, axis=1) > 0).view(np.uint8)

    t = np.vstack((selem, np.zeros((1, selem.shape[1]))))
    cdef unsigned char[:, :] t_s = (np.diff(t, axis=0) < 0).view(np.uint8)

    t = np.vstack((np.zeros((1, selem.shape[1])), selem))
    cdef unsigned char[:, :] t_n = (np.diff(t, axis=0) > 0).view(np.uint8)

    # the current local histogram distribution
    # cdef Py_ssize_t* histo
    cdef Py_ssize_t [::1] histo = np.zeros(n_bins, dtype=np.intp)

    # these lists contain the relative pixel row and column for each of the 4
    # attack borders east, west, north and south e.g. se_e_r lists the rows of
    # the east structuring element border
    cdef Py_ssize_t se_size = srows * scols
    cdef Py_ssize_t [::1] se_e_r = np.empty(se_size, dtype=np.intp)
    cdef Py_ssize_t [::1] se_e_c = np.empty(se_size, dtype=np.intp)

    cdef Py_ssize_t [::1] se_w_r = np.empty(se_size, dtype=np.intp)
    cdef Py_ssize_t [::1] se_w_c = np.empty(se_size, dtype=np.intp)

    cdef Py_ssize_t [::1] se_n_r = np.empty(se_size, dtype=np.intp)
    cdef Py_ssize_t [::1] se_n_c = np.empty(se_size, dtype=np.intp)

    cdef Py_ssize_t [::1] se_s_r = np.empty(se_size, dtype=np.intp)
    cdef Py_ssize_t [::1] se_s_c = np.empty(se_size, dtype=np.intp)

    # number of element in each attack border
    cdef Py_ssize_t num_se_n, num_se_s, num_se_e, num_se_w

    num_se_n = num_se_s = num_se_e = num_se_w = 0

    for i in range(n_bins):
        histo[i] = 0

    for r in range(srows):
        for c in range(scols):
            if t_e[r, c]:
                se_e_r[num_se_e] = r - centre_r
                se_e_c[num_se_e] = c - centre_c
                num_se_e += 1
            if t_w[r, c]:
                se_w_r[num_se_w] = r - centre_r
                se_w_c[num_se_w] = c - centre_c
                num_se_w += 1
            if t_n[r, c]:
                se_n_r[num_se_n] = r - centre_r
                se_n_c[num_se_n] = c - centre_c
                num_se_n += 1
            if t_s[r, c]:
                se_s_r[num_se_s] = r - centre_r
                se_s_c[num_se_s] = c - centre_c
                num_se_s += 1

    for r in range(srows):
        for c in range(scols):
            rr = r - centre_r
            cc = c - centre_c
            if selem[r, c]:
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image[rr, cc])

    r = 0
    c = 0
    kernel(&out[r, c, 0], odepth, histo, pop, image[r, c], n_bins, mid_bin,
           p0, p1, s0, s1)

    # main loop
    r = 0
    for even_row in range(0, rows, 2):

        # ---> west to east
        for c in range(1, cols):
            for s in range(num_se_e):
                rr = r + se_e_r[s]
                cc = c + se_e_c[s]
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image[rr, cc])

            for s in range(num_se_w):
                rr = r + se_w_r[s]
                cc = c + se_w_c[s] - 1
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_decrement(histo, &pop, image[rr, cc])

            kernel(&out[r, c, 0], odepth, histo, pop, image[r, c], n_bins,
                   mid_bin, p0, p1, s0, s1)

        r += 1  # pass to the next row
        if r >= rows:
            break

        # ---> north to south
        for s in range(num_se_s):
            rr = r + se_s_r[s]
            cc = c + se_s_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_increment(histo, &pop, image[rr, cc])

        for s in range(num_se_n):
            rr = r + se_n_r[s] - 1
            cc = c + se_n_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_decrement(histo, &pop, image[rr, cc])

        kernel(&out[r, c, 0], odepth, histo, pop, image[r, c], n_bins,
               mid_bin, p0, p1, s0, s1)

        # ---> east to west
        for c in range(cols - 2, -1, -1):
            for s in range(num_se_w):
                rr = r + se_w_r[s]
                cc = c + se_w_c[s]
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_increment(histo, &pop, image[rr, cc])

            for s in range(num_se_e):
                rr = r + se_e_r[s]
                cc = c + se_e_c[s] + 1
                if is_in_mask(rows, cols, rr, cc, mask_data):
                    histogram_decrement(histo, &pop, image[rr, cc])

            kernel(&out[r, c, 0], odepth, histo, pop, image[r, c], n_bins,
                   mid_bin, p0, p1, s0, s1)

        r += 1  # pass to the next row
        if r >= rows:
            break

        # ---> north to south
        for s in range(num_se_s):
            rr = r + se_s_r[s]
            cc = c + se_s_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_increment(histo, &pop, image[rr, cc])

        for s in range(num_se_n):
            rr = r + se_n_r[s] - 1
            cc = c + se_n_c[s]
            if is_in_mask(rows, cols, rr, cc, mask_data):
                histogram_decrement(histo, &pop, image[rr, cc])

        kernel(&out[r, c, 0], odepth, histo, pop, image[r, c],
               n_bins, mid_bin, p0, p1, s0, s1)


cdef void _core_3D(void kernel(dtype_t_out*, Py_ssize_t, Py_ssize_t[::1], double,
                               dtype_t, Py_ssize_t, Py_ssize_t, double,
                               double, Py_ssize_t, Py_ssize_t) nogil,
                   dtype_t[:, :, ::1] image,
                   char[:, :, ::1] selem,
                   char[:, :, ::1] mask,
                   dtype_t_out[:, :, :, ::1] out,
                   signed char shift_x, signed char shift_y,
                   signed char shift_z, double p0, double p1,
                   Py_ssize_t s0, Py_ssize_t s1,
                   Py_ssize_t n_bins) except *:
    """Compute histogram for each pixel neighborhood, apply kernel function and
    use kernel function return value for output image.
    """

    cdef Py_ssize_t planes = image.shape[0]
    cdef Py_ssize_t rows = image.shape[1]
    cdef Py_ssize_t cols = image.shape[2]
    cdef Py_ssize_t splanes = selem.shape[0]
    cdef Py_ssize_t srows = selem.shape[1]
    cdef Py_ssize_t scols = selem.shape[2]
    cdef Py_ssize_t odepth = out.shape[3]

    cdef Py_ssize_t centre_p = (selem.shape[0] // 2) + shift_x
    cdef Py_ssize_t centre_r = (selem.shape[1] // 2) + shift_y
    cdef Py_ssize_t centre_c = (selem.shape[2] // 2) + shift_z

    # check that structuring element center is inside the element bounding box
    if not 0 <= centre_p < splanes:
        raise ValueError("half selem + shift_x must be between 0 and selem")
    if not 0 <= centre_r < srows:
        raise ValueError("half selem + shift_y must be between 0 and selem")
    if not 0 <= centre_c < scols:
        raise ValueError("half selem + shift_z must be between 0 and selem")

    cdef Py_ssize_t mid_bin = n_bins // 2

    # define pointers to the data
    cdef char* mask_data = &mask[0, 0, 0]

    # define local variable types
    cdef Py_ssize_t p, r, c, j, rr, cc, pp, value, local_max, i, even_row

    # number of pixels actually inside the neighborhood (double)
    cdef double pop = 0

    # the current local histogram distribution
    cdef Py_ssize_t [::1] histo = np.zeros(n_bins, dtype=np.intp)

    # these lists contain the relative pixel plane, row and column for each of
    # the 4 attack borders east, north, west and south
    # e.g. se[0, 0, :] lists the planes of the east structuring element border
    cdef Py_ssize_t se_size = splanes * srows * scols
    cdef Py_ssize_t [:, :, ::1] se = np.zeros([4, 3, se_size], dtype=np.intp)

    # number of element in each attack border in 4 directions
    cdef Py_ssize_t [::1] num_se = np.zeros(4, dtype=np.intp)
    
    _count_attack_border_elements(selem, se, num_se, splanes, srows, scols, 
                                  centre_p, centre_r, centre_c)

    for p in range(planes):
        histo[:] = 0
        pop = 0
        _build_initial_histogram_from_neighborhood(image, selem, histo, &pop,
                                                   mask_data, p, planes, rows, cols,
                                                   splanes, srows, scols,
                                                   centre_p, centre_r,
                                                   centre_c)
        r = 0
        c = 0
        kernel(&out[p, r, c, 0], odepth, histo, pop, image[p, r, c],
               n_bins, mid_bin, p0, p1, s0, s1)

    # main loop

        for even_row in range(0, rows, 2):

            # ---> west to east
            for c in range(1, cols):
                _update_histogram(image, se, num_se, histo, &pop, mask_data, p,
                                  r, c, planes, rows, cols, axis_inc=0)

                kernel(&out[p, r, c, 0], odepth, histo, pop,
                       image[p, r, c], n_bins, mid_bin, p0, p1, s0, s1)

            r += 1  # pass to the next row
            if r >= rows:
                break

            # ---> north to south
            _update_histogram(image, se, num_se, histo, &pop, mask_data, p,
                              r, c, planes, rows, cols, axis_inc=3)

            kernel(&out[p, r, c, 0], odepth, histo, pop,
                   image[p, r, c], n_bins, mid_bin, p0, p1, s0, s1)

            # ---> east to west
            for c in range(cols - 2, -1, -1):
                _update_histogram(image, se, num_se, histo, &pop, mask_data, p,
                                  r, c, planes, rows, cols, axis_inc=2)

                kernel(&out[p, r, c, 0], odepth, histo, pop,
                       image[p, r, c], n_bins, mid_bin, p0, p1, s0, s1)

            r += 1  # pass to the next row
            if r >= rows:
                break

            # ---> north to south
            _update_histogram(image, se, num_se, histo, &pop, mask_data, p,
                              r, c, planes, rows, cols, axis_inc=3)

            kernel(&out[p, r, c, 0], odepth, histo, pop, image[p, r, c],
                   n_bins, mid_bin, p0, p1, s0, s1)
