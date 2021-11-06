from libcpp.vector cimport vector

cdef extern from "cla.h" namespace "geodesic" nogil:
    float c_geodesic_distance(unsigned int,
                              unsigned int,
                              unsigned int,
                              vector[vector[float]],
                              vector[vector[int]],
                              vector[float])
