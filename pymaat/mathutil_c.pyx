from pymaat.nputil import flat_view
from cython.parallel import prange

cimport cython
from libc.math cimport sqrt, erf, exp
from libc.math cimport M_SQRT1_2

DEF M_SQRT1_2PI = 0.398942280401432677939946059934381868475858631164934657665

cdef double _normcdf(double x) nogil:
    return 0.5*(1.+erf(x*M_SQRT1_2))

cdef double _normpdf(double x) nogil:
    return exp(-0.5*x*x)*M_SQRT1_2PI

@cython.boundscheck(False)
@cython.initializedcheck(False)
def normcdf(x, out):
    cdef:
        int n, N = x.size
        double[:] _x, _out
    _x = flat_view(x)
    _out = flat_view(out)
    for n in prange(N, nogil=True, schedule='static'):
        _out[n] = _normcdf(_x[n])

@cython.boundscheck(False)
@cython.initializedcheck(False)
def normpdf(x, out):
    cdef:
        int n, N = x.size
        double[:] _x, _out
    _x = flat_view(x)
    _out = flat_view(out)
    for n in prange(N, nogil=True, schedule='static'):
        _out[n] = _normpdf(_x[n])

