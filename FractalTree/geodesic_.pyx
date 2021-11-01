# distutils: language=c++

import numpy as np
cimport numpy as np
cimport cython
from libcpp.list cimport list as cpplist

dtype_int = np.int
dtype_float = np.float64

ctypedef np.float64_t DTYPE_FLOAT
ctypedef np.int_t DTYPE_INT


def compute_distance(DTYPE_INT a,
                     DTYPE_INT b,
                     np.ndarray[DTYPE_FLOAT, ndim=2] pts,
                     np.ndarray[DTYPE_INT, ndim=2] pap,
                     DTYPE_INT n) -> float:
    """
    Main method to compute the geodesic distance between two vertices

    :param a: first vertex
    :param b: second vertex
    :param pts: coordinates of all points in mesh
    :param pap: list of points ids around points padded with -1
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
    return geodesic(a, b, pts, pap, n)


def compute_distances(DTYPE_INT b,
                      np.ndarray[DTYPE_FLOAT, ndim=2] pts,
                      np.ndarray[DTYPE_INT, ndim=2] pap,
                      DTYPE_INT n):
    cdef double[:] dist = np.zeros(n)
    cdef DTYPE_INT i

    for i in range(n):
        dist[i] = geodesic(i, b, pts, pap, n)
    return dist


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_FLOAT geodesic(DTYPE_INT a,
                          DTYPE_INT b,
                          double[:, :] pts,
                          long[:, :] pap,
                          DTYPE_INT n):
    if a == b:
        return 0

    cdef double[:] d = np.zeros(n)
    cdef long[:] visited = np.zeros(n, dtype=int)
    visited[0] = True
    visited[a] = True
    cdef DTYPE_FLOAT dmin, dcur
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
                dcur = d[y] + n_(pts[x], pts[y])
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

    return np.min(endnodes)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_FLOAT n_(double[:] x, double[:] y):  # norm
    return ((y[0]-x[0])**2 + (y[1]-x[1])**2 + (y[2]-x[2])**2) ** 0.5
