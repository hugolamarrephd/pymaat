import numpy as np
from scipy.stats import norm

from pymaat.garch.model import AbstractOneLagReturn
# from scipy.special import ndtr as normcdf
from pymaat.mathutil_c import normcdf
from pymaat.nputil import flat_view

cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt, log, exp, fabs, INFINITY
from pymaat.mathutil_c cimport _normcdf, _normpdf

np.import_array()

class CentralRiskPremium(AbstractOneLagReturn):

    def __init__(self, mu):
        self.mu = mu
        self.cimpl = _CentralRiskPremium(mu)

    def _one_step_generate(self, innovations, variances, volatilities):
        """
            `r_t = (mu - 0.5)*h_{t} + sqrt(h_{t}) * z_{t}`
        """
        return (self.mu-0.5)*variances + volatilities*innovations

    def _one_step_filter(self, returns, variances, volatilities):
        return (returns-(self.mu-0.5)*variances)/volatilities

    def _roots(
            self, prev_logprice, prev_variance, logprice, prev_volatility):
        shape = prev_logprice.shape
        flatten = (prev_logprice.size,)
        roots = np.empty(flatten)
        der = np.empty(flatten)
        self.cimpl.roots(
                np.ravel(prev_logprice),
                np.ravel(prev_variance),
                np.ravel(logprice),
                np.ravel(prev_volatility),
                roots, der)
        roots.shape = shape
        der.shape = shape
        return roots, der

    def _quantized_integral(
            self, prev_logprice, prev_variance, voronoi, I, d):
        self.cimpl.quantized_integral(
                prev_logprice, prev_variance, voronoi, *I, *d)


cdef class _CentralRiskPremium:
    cdef:
        double mu

    def __cinit__(self, double mu):
        self.mu = mu

    def __reduce__(self):
        return (self.__class__, (self.mu,))

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef roots(self,
            # Inputs
            double[:] prev_logprice,
            double[:] prev_variance,
            double[:] logprice,
            double[:] prev_volatility,
            double[:] root,
            double[:] derivative,
            ):
        cdef:
            int size = prev_logprice.size
            int i
        for i in range(size):
            self._roots(
                    prev_logprice[i],
                    prev_variance[i],
                    prev_volatility[i],
                    logprice[i],
                    &root[i],
                    &derivative[i]
                    )

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef quantized_integral(self,
            # Previous values
            double[:] prev_logprice, double[:] prev_variance,
            # Current values
            double[:] voronoi,
            # Results (prev_size, size)
            double[:,::1] I0,
            double[:,::1] I1,
            double[:,::1] I2,
            double[:,::1] D0,
            double[:,::1] D1,
            ):
        cdef:
            int prev_size = prev_logprice.size
            int size = voronoi.size-1
            int i

        for i in range(prev_size):
            self._do_integrate(size, &voronoi[0],
                    prev_logprice[i],
                    prev_variance[i],
                    &I0[i,0], &I1[i,0], &I2[i,0],
                    &D0[i,0], &D1[i,0])

    cdef void _do_integrate(self,
            int size, double *bounds,
            double prev_logprice, double prev_var,
            double *I0, double *I1, double *I2,
            double *D0, double *D1) nogil:
        cdef:
            double prev_vol
            int j
            double i0, i1, i2
            double der
            double lb, ub
        prev_vol = sqrt(prev_var)
        self._roots(
                prev_logprice, prev_var, prev_vol, bounds[0], &lb, &der)
        # Initialize derivative (assuming bounds[0] is 0)
        D0[0] = 0.
        D1[0] = 0.
        for j in range(size):
            # Compute roots
            self._roots(prev_logprice, prev_var, prev_vol,
                    bounds[j+1], &ub, &der)
            # Initialize
            I0[j] = I1[j] = I2[j] = 0.

            # To
            self._integral_until(prev_logprice, prev_var, prev_vol, ub,
                    &i0, &i1, &i2)
            I0[j] += i0; I1[j] += i1; I2[j] += i2

            # From
            self._integral_until(prev_logprice, prev_var, prev_vol, lb,
                    &i0, &i1, &i2)
            I0[j] -= i0; I1[j] -= i1; I2[j] -= i2

            D0[j+1] = 0.5*_normpdf(ub)*der
            D1[j+1] = D0[j+1]*bounds[j+1]
            lb = ub
        D0[size] = 0.
        D1[size] = 0.

    cdef void _integral_until(self,
            double prev_logprice,
            double prev_var, double prev_vol,
            double at,
            double *i0, double *i1, double *i2) nogil:
        cdef:
            double cdf = _normcdf(at)
            double pdf = _normpdf(at)
            double mutilde = prev_logprice + (self.mu-0.5)*prev_var
        i0[0] = cdf
        i1[0] = mutilde*cdf - prev_vol*pdf
        i2[0] = (mutilde*mutilde + prev_var)*cdf
        if fabs(at) == INFINITY:
            i2[0] -= 2.*mutilde*prev_vol*pdf
        else:
            i2[0] -= (2.*mutilde*prev_vol + at*prev_var) * pdf

    @cython.cdivision(True)
    cdef void _roots(self,
            double logprice, double variance, double volatility,
            double next_logprice,
            double* root, double* derivative) nogil:
        root[0] = (next_logprice-logprice-(self.mu-0.5)*variance)/volatility
        derivative[0] = 1./volatility
