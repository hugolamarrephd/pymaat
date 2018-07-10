import numpy as np
from scipy.stats import norm

from pymaat.garch.model import AbstractOneLagReturn
# from scipy.special import ndtr as normcdf
from pymaat.mathutil_c import normcdf

cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt, log, exp
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

    def _root_price_derivative(self, prices, variances, volatilities):
        return (prices * volatilities)**-1.

    def _first_order_integral(
            self, prices, variances, innovations, volatilities):
        out = np.empty_like(innovations)
        normcdf(innovations-volatilities, out)
        out *= np.exp(variances*self.mu)
        out *= prices
        return out

    def _second_order_integral(
            self, prices, variances, innovations, volatilities):
        out = np.empty_like(innovations)
        normcdf(innovations-2.*volatilities, out)
        out *= np.exp(variances*(2.*self.mu+1.))
        out *= prices
        out *= prices
        return out

cdef class _CentralRiskPremium:
    cdef:
        double mu

    def __cinit__(self, double mu):
        self.mu = mu

    def __reduce__(self):
        return (self.__class__, (self.mu,))

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef integrate(self,
            # Previous values
            double[:] prev_prices, double[:] prev_variances,
            # Current values
            double[:] bounds,
            # Results (prev_size, size)
            double[:,::1] I0,
            double[:,::1] I1,
            double[:,::1] I2,
            double[:,::1] D0,
            double[:,::1] D1,
            ):
        cdef:
            int prev_size = prev_prices.size
            int size = bounds.size-1
            int i

        for i in range(prev_size):
            self._do_integrate(size, &bounds[0],
                    prev_prices[i],
                    prev_variances[i],
                    &I0[i,0], &I1[i,0], &I2[i,0],
                    &D0[i,0], &D1[i,0])

    cdef void _do_integrate(self,
            int size, double *bounds,
            double prev_price, double prev_var,
            double *I0, double *I1, double *I2,
            double *D0, double *D1) nogil:
        cdef:
            double prev_vol
            int j
            double i0, i1, i2
            double der
            double lb, ub
        prev_vol = sqrt(prev_var)
        self._roots(prev_price, prev_var, prev_vol, bounds[0], &lb, &der)
        # Initialize derivative (assuming bounds[0] is 0)
        D0[0] = 0.
        D1[0] = 0.
        for j in range(size):
            # Compute roots
            self._roots(prev_price, prev_var, prev_vol,
                    bounds[j+1], &ub, &der)
            # Initialize
            I0[j] = I1[j] = I2[j] = 0.

            # To
            self._integral_until(prev_price, prev_var, prev_vol, ub,
                    &i0, &i1, &i2)
            I0[j] += i0; I1[j] += i1; I2[j] += i2

            # From
            self._integral_until(prev_price, prev_var, prev_vol, lb,
                    &i0, &i1, &i2)
            I0[j] -= i0; I1[j] -= i1; I2[j] -= i2

            D0[j+1] = 0.5*_normpdf(ub)*der
            D1[j+1] = D0[j+1]*bounds[j+1]
            lb = ub
        D0[size] = 0.
        D1[size] = 0.

    cdef void _integral_until(self,
            double prev_price, double prev_var, double prev_vol, double at,
            double *i0, double *i1, double *i2) nogil:
        i0[0] = _normcdf(at)
        i1[0] = prev_price * exp(prev_var*self.mu) * _normcdf(at-prev_vol)
        i2[0] = (prev_price*prev_price * exp(prev_var*(2.*self.mu+1.))
                * _normcdf(at-2.*prev_vol))

    @cython.cdivision(True)
    cdef void _roots(self,
            double price, double variance, double volatility,
            double next_price,
            double* root, double* derivative) nogil:
        root[0] = (log(next_price/price)-(self.mu-0.5)*variance)/volatility
        derivative[0] = 1./(next_price*volatility)
