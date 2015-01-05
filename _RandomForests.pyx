__author__ = 'artanis'

import numpy as np
cimport numpy as np

ctypedef np.int32_t INT32
ctypedef np.float32_t FLOAT32


cdef extern from "_random_forests.h":
    void forestFindThr(int H, int N, int F, const FLOAT32 *data, const INT32 *hs,
                       const float* ws, const INT32 *order, const int split,
                       int &fid, float &thr, double &gain) except +


def find_threshold(H, split, ftrs, lbls, dwts):
    cdef int N = ftrs.shape[0], F = ftrs.shape[1]
    cdef np.ndarray[FLOAT32, ndim=2] data
    cdef np.ndarray[INT32, ndim=1] hs
    cdef np.ndarray[FLOAT32, ndim=1] ws
    cdef np.ndarray[INT32, ndim=2] order

    data = np.asfortranarray(ftrs, dtype=np.float32)
    hs = np.asfortranarray(lbls, dtype=np.int32)
    ws = np.asfortranarray(dwts, dtype=np.float32)
    order = np.asfortranarray(np.argsort(ftrs, axis=0, kind='mergesort'),
                              dtype=np.int32)

    cdef int fid = -1
    cdef float thr = 0
    cdef double gain = 0

    forestFindThr(H, N, F, &data[0, 0], &hs[0], &ws[0], &order[0, 0],
                  split, fid, thr, gain)

    return fid, thr, gain