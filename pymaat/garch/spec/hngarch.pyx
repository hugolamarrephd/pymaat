import numpy as np

from pymaat.garch.model import AbstractOneLagGarch
from pymaat.garch.spec.ret import CentralRiskPremium
from pymaat.nputil import flat_view, forced_elbyel

cimport numpy as np
cimport cython
from libc.math cimport sqrt, fmax, fmin, fabs, INFINITY
from pymaat.mathutil_c cimport _normcdf, _normpdf

np.import_array()

class HestonNandiGarch(AbstractOneLagGarch):

    def __init__(self, mu, omega, alpha, beta, gamma):
        super().__init__(CentralRiskPremium(mu))
        if (1.-beta-alpha*gamma**2. <= 0.):
            raise ValueError("Invalid HN-Garch parameters")
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Wraps C-level implementation...
        self.cimpl  = _HestonNandiGarch(omega, alpha, beta, gamma)

    def _equation(self, innovations, variances, volatilities):
        """
         ` h_{t+1} = omega + beta*h_{t} + alpha*(z_t-gamma*sqrt(h_{t}))^2`
        """
        return (self.omega + self.beta*variances
                + self.alpha*np.power(innovations-self.gamma*volatilities,2))

    def _real_roots(self, variances, next_variances):
        discr = (next_variances-self.omega-self.beta*variances)/self.alpha
        const = self.gamma * np.sqrt(variances)
        sqrtdiscr = np.maximum(discr, 0.)**0.5  # Real part only
        return [const + (pm * sqrtdiscr) for pm in [-1., 1.]]

    def _real_roots_unsigned_derivative(self, variances, next_variances):
        discr = next_variances-self.omega-self.beta*variances
        sqrtdiscr = (self.alpha*np.maximum(discr, 0.))**0.5  # Real part only
        out = np.zeros_like(sqrtdiscr)
        out[sqrtdiscr>0.] = 0.5 / sqrtdiscr[sqrtdiscr>0.]
        return out

    def _get_lowest_one_step_variance(self, variances, lb, ub):
        adj_lb = lb - self.gamma*np.sqrt(variances)
        adj_ub = ub - self.gamma*np.sqrt(variances)
        out = np.minimum(adj_lb**2., adj_ub**2.)
        out[np.logical_and(adj_lb<0., adj_ub>0.)] = 0.
        return self.omega + self.beta*variances + self.alpha*out

    def _get_highest_one_step_variance(self, variances, lb, ub):
        adj_lb = lb - self.gamma*np.sqrt(variances)
        adj_ub = ub - self.gamma*np.sqrt(variances)
        out = np.maximum(adj_lb**2., adj_ub**2.)
        return self.omega + self.beta*variances + self.alpha*out

    def _first_order_integral_factors(self, variances, innovations):
        p1 = 2.*self.gamma*self.alpha
        p2 = -self.alpha
        c1 = self.omega + self.alpha
        c2 = self.beta + self.alpha*self.gamma*self.gamma
        volatilities = np.sqrt(variances)
        pdf_factor = p1*volatilities + p2*innovations
        cdf_factor = c1 + c2*variances
        return pdf_factor, cdf_factor

    def _second_order_integral_factors(self, variances, innovations):
        volatilities = np.sqrt(variances)
        z = innovations
        z2 = z*z
        gv = self.gamma*volatilities
        gv2 = gv*gv
        minh = self.beta*variances + self.omega
        pdf_factor = self.alpha*(2.*(2.*gv-z)*minh + self.alpha*(
            2.*gv2*(2.*gv-3.*z) + 4.*gv*(z2+2.) - z*(z2+3.)))
        cdf_factor = (
                self.alpha**2.*(gv2*(gv2+6.)+3.)
                + 2.*self.alpha*(gv2+1.)*minh
                + minh*minh
                )
        return pdf_factor, cdf_factor

cdef class _HestonNandiGarch:

    cdef:
        double omega
        double alpha
        double beta
        double gamma
        double p1, p2, c1, c2

    def __cinit__(self,
            double omega, double alpha, double beta, double gamma):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Numerical efficiency optimization..
        # First order integral
        self.p1 = 2.*self.gamma*self.alpha
        self.p2 = -self.alpha
        self.c1 = self.omega + self.alpha
        self.c2 = self.beta + self.alpha*self.gamma*self.gamma

    def __reduce__(self):
        return (self.__class__,
                (self.omega, self.alpha, self.beta, self.gamma))

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef integrate(self,
            # Previous values
            double[:] prev_value, double[:] low, double[:] high,
            # Current values
            double[:] bounds, # bounds must have size (size+1)
            # Results (prev_size, size)
            double[:,::1] I0,
            double[:,::1] I1,
            double[:,::1] I2,
            double[:,::1] D0,
            double[:,::1] D1,
            ):
        cdef:
            int prev_size = prev_value.size
            int size = bounds.size-1
            int i
            double lb, ub

        for i in range(prev_size):
            self._do_integrate(size, &bounds[0],
                    prev_value[i], low[i], high[i],
                    &I0[i,0], &I1[i,0], &I2[i,0],
                    &D0[i,0], &D1[i,0])

    cdef void _do_integrate(self,
        int size, double *bounds,
            double prev_var, double lb, double ub,
            double *I0, double *I1, double *I2,
            double *D0, double *D1) nogil:
        cdef:
            double prev_vol
            int j
            double a, b, c, d
            double _a, _b, _c, _d
            double i0, i1, i2
            double der, _
            double pdf
        prev_vol = sqrt(prev_var)
        self._real_roots(prev_var, prev_vol, bounds[0], &b, &c, &der)
        # Initialize derivative
        D0[0] = 0.
        if b>lb and b<ub:
            D0[0] += 0.5*_normpdf(b)*der
        if c>lb and c<ub:
            D0[0] += 0.5*_normpdf(c)*der
        D1[0] = D0[0]*bounds[0]
        for j in range(size):
            # Compute roots
            self._real_roots(
                    prev_var, prev_vol, bounds[j+1], &a, &d, &der)
            # Bound roots
            _a = fmax(a, lb)
            _b = fmin(b, ub)
            _c = fmax(c, lb)
            _d = fmin(d, ub)
            # Initialize
            I0[j] = I1[j] = I2[j] = 0.
            D0[j+1] = 0.
            # Left branch
            if _b > _a:
                self._integral_until(
                        prev_var, prev_vol, _b, &i0, &i1, &i2, &pdf)
                I0[j] += i0; I1[j] += i1; I2[j] += i2
                self._integral_until(
                        prev_var, prev_vol, _a, &i0, &i1, &i2, &pdf)
                I0[j] -= i0; I1[j] -= i1; I2[j] -= i2
                if a > lb:  # Contributes to derivative
                    D0[j+1] += 0.5*pdf*der
            # Right branch
            if _d > _c:
                self._integral_until(
                        prev_var, prev_vol, _d, &i0, &i1, &i2, &pdf)
                I0[j] += i0; I1[j] += i1; I2[j] += i2
                if d < ub:  # Contributes to derivative
                    D0[j+1] += 0.5*pdf*der
                self._integral_until(
                        prev_var, prev_vol, _c, &i0, &i1, &i2, &pdf)
                I0[j] -= i0; I1[j] -= i1; I2[j] -= i2
            # Compute first order derivative
            D1[j+1] = D0[j+1]*bounds[j+1]
            # Prepare next iteration
            b = a
            c = d
        D1[size] = 0.

    @cython.cdivision(True)
    cdef void _real_roots(self,
            double variance, double volatility,
            double next_variance,
            double *left, double *right,
            double *unsigned_derivative) nogil:
        cdef:
            double discr, const, sqrtdiscr
        discr = (next_variance-self.omega-self.beta*variance)/self.alpha
        const = self.gamma*volatility
        sqrtdiscr = sqrt(fmax(discr, 0.))  # real part only!
        left[0] = const - sqrtdiscr
        right[0] = const + sqrtdiscr
        if sqrtdiscr>0.:
            unsigned_derivative[0] = 0.5/(self.alpha*sqrtdiscr)
        else:
            unsigned_derivative[0] = 0.

    cdef void _integral_until(self,
            double prev_var, double prev_vol, double at,
            double *i0, double *i1, double *i2, double *pdf) nogil:
        cdef:
            double cdf
            double pf, cf
        pdf[0] = _normpdf(at)
        cdf = _normcdf(at)
        i0[0] = cdf
        self._first_order_integral_factors(
                prev_var, prev_vol, at, &pf, &cf)
        i1[0] = pdf[0]*pf + cdf*cf
        self._second_order_integral_factors(
                prev_var, prev_vol, at, &pf, &cf)
        i2[0] = pdf[0]*pf + cdf*cf

    cdef void _first_order_integral_factors(self,
            double variance, double volatility,
            double innovation,
            double *pdf_factor, double *cdf_factor) nogil:
        if fabs(innovation) == INFINITY:
            pdf_factor[0] = 0.
        else:
            pdf_factor[0] = self.p1*volatility + self.p2*innovation
        cdf_factor[0] = self.c1 + self.c2*variance

    cdef void _second_order_integral_factors(self,
            double variance, double volatility,
            double innovation,
            double *pdf_factor, double *cdf_factor) nogil:
        cdef:
            double z, z2, gv, gv2, minh
        z = innovation
        z2 = z*z
        gv = self.gamma*volatility
        gv2 = gv*gv
        minh = self.beta*variance + self.omega
        if fabs(innovation) == INFINITY:
            pdf_factor[0] = 0.
        else:
            pdf_factor[0] = self.alpha*(2.*(2.*gv-z)*minh + self.alpha*(
                2.*gv2*(2.*gv-3.*z) + 4.*gv*(z2+2.) - z*(z2+3.)))
        cdf_factor[0] = (
                self.alpha**2.*(gv2*(gv2+6.)+3.)
                + 2.*self.alpha*(gv2+1.)*minh
                + minh*minh
                )
