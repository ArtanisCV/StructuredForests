import numpy as N
cimport numpy as N

ctypedef N.float32_t FLOAT32
ctypedef N.float64_t FLOAT64


def histogram_core(double[:, :] magnitude, double[:, :] orientation,
                   int downscale, int n_orient, int interp):
    cdef int n_row = magnitude.shape[0], n_col = magnitude.shape[1]
    cdef int n_rbin = (n_row + downscale - 1) / downscale
    cdef int n_cbin = (n_col + downscale - 1) / downscale
    cdef int i, j, r, c, o1, o2
    cdef double o_range = N.pi / n_orient, o
    cdef N.ndarray[FLOAT64, ndim=3] hist_arr

    hist_arr = N.zeros((n_rbin, n_cbin, n_orient), dtype=N.float64)
    cdef double[:, :, :] hist = hist_arr

    with nogil:
        for i from 0 <= i < n_row:
            for j from 0 <= j < n_col:
                r, c = i / downscale, j / downscale

                if interp:
                    o = orientation[i, j] / o_range
                    o1 = <int>o % n_orient
                    o2 = (o1 + 1) % n_orient
                    hist[r, c, o1] += magnitude[i, j] * (1 + <int>o - o)
                    hist[r, c, o2] += magnitude[i, j] * (o - <int>o)
                else:
                    o1 = <int>(orientation[i, j] / o_range + 0.5) % n_orient
                    hist[r, c, o1] += magnitude[i, j]

    return hist_arr / downscale ** 2


def pdist_core(double[:, :, :] src):
    cdef int n_in = src.shape[0], n_pt = src.shape[1], n_dim = src.shape[2]
    cdef int i, j, k, m, n
    cdef N.ndarray[FLOAT64, ndim=3] dst_arr

    dst_arr = N.zeros((src.shape[0], n_pt * (n_pt - 1) / 2, n_dim),
                      dtype=N.float64)
    cdef double[:, :, :] dst = dst_arr

    with nogil:
        for i from 0 <= i < n_in:
            n = 0

            for j from 0 <= j < n_pt:
                for k from j + 1 <= k < n_pt:
                    for m from 0 <= m < n_dim:
                        dst[i, n, m] = src[i, j, m] - src[i, k, m]

                    n += 1

    return dst_arr