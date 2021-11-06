# distutils: language=c++

import numpy as np
cimport numpy as cnp
cimport cython
from libcpp.list cimport list as cpplist
from libcpp.algorithm cimport min_element
from libc.math cimport fmin, fmax
from cython.parallel import prange

ctypedef cnp.uint8_t DTYPE_BOOL
ctypedef cnp.int_t DTYPE_INT
ctypedef cnp.float64_t DTYPE_FLOAT


def compute_distance(DTYPE_INT a,
                     DTYPE_INT b,
                     cnp.ndarray[DTYPE_FLOAT, ndim=2] pts,
                     cnp.ndarray[DTYPE_INT, ndim=2] pap,
                     cnp.ndarray[DTYPE_FLOAT, ndim=1] cond,
                     DTYPE_INT n) -> float:
    """
    Main function to compute the geodesic distance between two vertices

    :param a: first vertex
    :param b: second vertex
    :param pts: coordinates of all points in mesh
    :param pap: list of points ids around points padded with -1
    :param cond: normalized conductivity (defaults to np.ones)
    :param n: number of mesh points
    :return: geodesic distance from a to b

    .. Exemple:
                n = self.nbPoints
                pap = self.pointIdsAroundPoint
                shape = (n, len(max(pap, key=lambda x: len(x))))
                pts_a_pts = np.ones(shape, dtype=int) * -1
                for i, x in enumerate(pap):
                    pts_a_pts[i][: len(x)] = x

                geo = partial(compute_distance, b=0, pts=self.pts, pap=pts_a_pts, n=n)
                with Pool() as pool:
                    _dist = np.array(list(tqdm(pool.imap(geo, range(n)), total=n)))
                # _dist: geodesic distance for each point from point b=0
    """
    return geodesic(a, b, pts, pap, cond, n)


def compute_distances(DTYPE_INT b,
                      cnp.ndarray[DTYPE_FLOAT, ndim=2] pts,
                      cnp.ndarray[DTYPE_INT, ndim=2] pap,
                      cnp.ndarray[DTYPE_FLOAT, ndim=1] cond,
                      DTYPE_INT n) -> np.ndarray:

    cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] pts_ = pts
    cdef cnp.ndarray[DTYPE_INT, ndim=2] pap_ = pap

    cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] dist = np.zeros(n)
    cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] K = cond  # normalized conductivity
    cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] d = np.zeros(n)  # distances
    cdef cnp.ndarray[DTYPE_BOOL, ndim=1] visited = np.zeros(n, dtype=int)
    cdef DTYPE_FLOAT dmin, dcur, nc
    cdef DTYPE_INT x, y, n_iter = 0, i

    cdef cpplist[float] endnodes
    cdef cpplist[int] around, origins, new_origins

    visited[0] = 1
    visited[i] = 1

    for i in prange(n, nogil=True):
        if i == b:
            dist[i] = 0
        else:
            endnodes.clear()
            around.clear()
            new_origins.clear()
            origins.clear()
            origins.push_back(i)

            while origins.size() > 0 and n_iter < n:
                n_iter += 1
                dmin = min_element(endnodes) if endnodes.size() > 0 else 1e10
                new_origins.clear()
                for y in origins:
                    around.clear()
                    for x in pap_[y]:
                        if x != -1:
                            around.push_back(x)
                    for x in around:
                        nc = fmax(0.5 * (K[y] + K[x]), 1e-6)  # threshold
                        dcur = d[y] + n_ng(pts_[x], pts_[y]) / nc
                        if x == b:
                            endnodes.push_back(dcur)
                        elif not visited[x]:
                            if dcur < dmin:
                                new_origins.push_back(x)
                            d[x] = dcur
                            visited[x] = 1
                        else:
                            d[x] = fmin(dcur, d[x])

                origins = new_origins

            if endnodes.size() == 0:
                dist[i] = -1.

            # return np.min(endnodes)

            dist[i] = min_ng(endnodes)

        # dist[i] = geodesic_nogil(i, b, pts, pap, cond, n)

    return dist



@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_FLOAT n_ng(double[:] x, double[:] y) nogil:  # norm
    return ((y[0]-x[0])**2 + (y[1]-x[1])**2 + (y[2]-x[2])**2) ** 0.5


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_FLOAT min_ng(double[:] x) nogil:  # min
    cdef DTYPE_FLOAT y, m = x[0]
    for y in x:
        if y < m:
            m = y
    return m



# # @cython.boundscheck(False) # turn off bounds-checking for entire function
# # @cython.wraparound(False)  # turn off negative index wrapping for entire function
# cdef DTYPE_FLOAT geodesic_nogil(DTYPE_INT a,
#                           DTYPE_INT b,
#                           cnp.ndarray[DTYPE_FLOAT, ndim=2] pts,
#                           cnp.ndarray[DTYPE_INT, ndim=2] pap,
#                           cnp.ndarray[DTYPE_FLOAT, ndim=1] conductivity,
#                           DTYPE_INT n) nogil:
#     if a == b:
#         return 0
#
#     cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] K = conductivity  # normalized conductivity
#     cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] d  = np.zeros(n)  # distances
#     cdef cnp.ndarray[DTYPE_BOOL, ndim=1] visited  = np.zeros(n, dtype=int)
#     cdef DTYPE_FLOAT dmin, dcur, nc
#     cdef DTYPE_INT x, y, n_iter = 0
#
#
#     visited[0] = 1
#     visited[a] = 1
#
#
#     return 0
#
#     #
#     # cdef cpplist[float] endnodes
#     # cdef cpplist[int] around, origins, new_origins
#     # origins.push_back(a)
#     #
#     # while origins.size() > 0 and n_iter < n:
#     #     n_iter += 1
#     #     dmin = np.min(endnodes) if len(endnodes) > 0 else 1e10
#     #     new_origins.clear()
#     #     for y in origins:
#     #         around.clear()
#     #         for x in pap[y]:
#     #             if x != -1:
#     #                 around.push_back(x)
#     #         # around = [x for x in pap[y] if x!= -1]
#     #         for x in around:
#     #             nc = np.maximum(0.5 * (K[y] + K[x]), 1e-6)  # threshold
#     #             dcur = d[y] + n_(pts[x], pts[y]) / nc
#     #             if x == b:
#     #                 endnodes.push_back(dcur)
#     #             elif not visited[x]:
#     #                 if dcur < dmin:
#     #                     new_origins.push_back(x)
#     #                 d[x] = dcur
#     #                 visited[x] = 1
#     #             else:
#     #                 d[x] = min(dcur, d[x])
#     #
#     #     origins = new_origins
#     #
#     # if endnodes.size() == 0:
#     #     return -1.
#     #
#     # return np.min(endnodes)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef DTYPE_FLOAT n_(double[:] x, double[:] y):  # norm
#     return ((y[0]-x[0])**2 + (y[1]-x[1])**2 + (y[2]-x[2])**2) ** 0.5




@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_FLOAT geodesic(DTYPE_INT a,
                          DTYPE_INT b,
                          double[:, :] pts,
                          long[:, :] pap,
                          double[:] conductivity,
                          DTYPE_INT n):
    if a == b:
        return 0

    cdef double[:] d = np.zeros(n)  # distances
    cdef double[:] K = conductivity  # normalized conductivity
    cdef long[:] visited = np.zeros(n, dtype=int)
    visited[0] = True
    visited[a] = True
    cdef DTYPE_FLOAT dmin, dcur, nc
    cdef DTYPE_INT x, y, n_iter = 0


    cdef cpplist[float] endnodes
    cdef cpplist[int] origins, new_origins
    origins.push_back(a)

    while origins.size() > 0 and n_iter < n:
        n_iter += 1
        dmin = np.min(endnodes) if len(endnodes) > 0 else 1e10
        new_origins.clear()
        for y in origins:
            around = [x for x in pap[y] if x!= -1]
            for x in around:
                nc = np.maximum(0.5 * (K[y] + K[x]), 1e-6)  # threshold
                dcur = d[y] + n_(pts[x], pts[y]) / nc
                if x == b:
                    endnodes.push_back(dcur)
                elif not visited[x]:
                    if dcur < dmin:
                        new_origins.push_back(x)
                    d[x] = dcur
                    visited[x] = True
                else:
                    _d = dcur
                    d[x] = min(_d, d[x])

        origins = new_origins

    if endnodes.size() == 0:
        return -1.

    return np.min(endnodes)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_FLOAT n_(double[:] x, double[:] y):  # norm
    return ((y[0]-x[0])**2 + (y[1]-x[1])**2 + (y[2]-x[2])**2) ** 0.5
