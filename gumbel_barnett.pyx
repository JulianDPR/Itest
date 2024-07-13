# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:13:16 2024

@author: Julian
"""

# gumbel_barnett.pyx
import numpy as np
cimport numpy as np
cimport cython
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def inv_Gumbel_Barnett_cython(np.ndarray[double, ndim=1] u1, np.ndarray[double, ndim=1] c, double alpha):
    cdef int n = u1.shape[0]
    cdef np.ndarray[double, ndim=1] y = -np.log(1 - u1)
    cdef np.ndarray[double, ndim=1] x = np.zeros(n)
    cdef int i

    for i in range(n):
        x[i] = _find_root(y[i], c[i], alpha)

    return 1 - np.exp(-np.abs(x))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _find_root(double y, double c, double alpha):
    cdef double x
    cdef double f
    cdef double tol = 1e-6
    cdef double a = 0
    cdef double b = 20

    while True:
        x = (a + b) / 2
        f = 1 - (1 + alpha * x) * np.exp(-(1 + alpha * y) * x) - c

        if np.abs(f) < tol:
            break
        elif f < 0:
            a = x
        else:
            b = x

    return x
