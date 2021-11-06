# distutils: language = c++

from typing import List

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from libcpp.vector cimport vector

from cla cimport c_geodesic_distance

ctypedef cnp.uint8_t DTYPE_BOOL
ctypedef cnp.float64_t DTYPE_FLOAT
ctypedef cnp.int_t DTYPE_INT


def geodesic_distance(idx: List[int], b: int, n: int,
             pts: np.ndarray[:,:],
             pap: List[List[int]],
             cond: np.dnarray[:,:]) -> np.ndarray[:]:
    """
    Main entry point
    :param idx: indices to compute
    :param b:
    :param n:
    :param pts:
    :param pap:
    :param cond:
    :return:
    """
    return geodesic_distance_(idx, b, n, pts, pap, cond)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef geodesic_distance_(vector[int] idx, int b, int n,
                      cnp.ndarray[DTYPE_FLOAT, ndim=2] pts: np.ndarray,
                      vector[vector[int]] pap,
                      cnp.ndarray[DTYPE_FLOAT, ndim=1] cond):
    cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] dist = np.zeros(n, dtype=float)
    cdef DTYPE_INT i, m = len(idx)

    cdef vector[float] c_cond = cond
    cdef vector[vector[float]] c_pts = pts

    for i in prange(m, nogil=True):
        dist[idx[i]] = c_geodesic_distance(b, idx[i], n, c_pts, pap, c_cond)

    return dist
