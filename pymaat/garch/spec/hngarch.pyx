import numpy as np
from scipy.integrate import quad
from multiprocessing import Pool
from functools import partial

from pymaat.garch.model import AbstractOneLagGarch
from pymaat.garch.spec.ret import CentralRiskPremium
from pymaat.util import method_decorator
from pymaat.nputil import flat_view, ravel

cimport numpy as np
cimport cython
from libc.math cimport sqrt, fmax, fmin, fabs, INFINITY
from pymaat.mathutil_c cimport _normcdf, _normpdf

cdef extern from "complex.h":
    double complex clog(double complex)
    double complex cexp(double complex)

np.import_array()

class HestonNandiGarch(AbstractOneLagGarch):

    def __init__(self, mu, omega, alpha, beta, gamma):
        super().__init__(CentralRiskPremium(mu))
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bag2 = beta + alpha*gamma**2.
        if self.bag2 >= 1.:
            raise ValueError("Invalid HN-Garch parameters")
        # C-level implementation
        self.cimpl  = _HestonNandiGarch(mu, omega, alpha, beta, gamma)

    def _equation(self, innovations, variances, volatilities):
        """
         ` h_{t+1} = omega + beta*h_{t} + alpha*(z_t-gamma*sqrt(h_{t}))^2`
        """
        return (self.omega + self.beta*variances
                + self.alpha*np.power(innovations-self.gamma*volatilities,2))

    def _timeseries_filter(self, returns, variances, innovations):
        self.cimpl.timeseries_filter(returns, variances, innovations)
        return (variances, innovations)

    def _timeseries_generate_returns(
            self, innovations, variances, returns):
        self.cimpl.timeseries_generate_returns(
                innovations, variances, returns)
        return (variances, returns)

    def _timeseries_generate_logprices(
            self, innovations, variances, logprices):
        self.cimpl.timeseries_generate_logprices(
                innovations, variances, logprices)
        return (variances, logprices)

    @method_decorator(ravel)
    def option_price(
            self,
            price,
            strike,
            variance,
            *,
            T=21,
            put=False,
            parallel=True
            ):
        """
        Assumes preliminary change of numeraire (i.e. rfr is zero):
            price is S*exp(-(rfr-div)*t)
            strike is K*exp(-(rfr-div)*t)
        """
        if not (price.size == strike.size == variance.size):
            raise ValueError("Size mismatch")
        if self.retspec.mu != 0.:
            raise ValueError("The model must be risk-neutral")
        T = int(T)
        if T<=0:
            result = np.maximum(price-strike,0.)
        else:
            if parallel:
                p = Pool()
                f = partial(self._get_call_price, T)
                result = p.starmap(f, zip(price,strike,variance))
                p.close()
                result = np.array(result)
            else:
                result = np.empty_like(price)
                for (i,inputs) in enumerate(zip(price,strike,variance)):
                    result[i] = self._get_call_price(T,*inputs)
        if put:
            return result - price + strike # by put-call parity
        else:
            return result

    def _get_call_price(self,T,s,k,h):
        return self._P1(s,k,h,T)*s - self._P2(s,k,h,T)*k

    @method_decorator(ravel)
    def delta_hedge(
            self,
            price,
            strike,
            variance,
            portfolio,
            *,
            tau=21,
            put=False,
            parallel=True
            ):
        """
        Assumes preliminary change of numeraire (i.e. rfr is zero):
            price is S*exp(-(rfr-div)*t)
            strike is K*exp(-(rfr-div)*t)
        portfolio is not used here
        """
        if not (price.size == strike.size == variance.size):
            raise ValueError("Size mismatch")
        if self.retspec.mu != 0.:
            raise ValueError("The model must be risk-neutral")
        T = int(tau)
        if T<=0:
            result = np.double(price>=strike)
        else:
            if parallel:
                p = Pool()
                f = partial(self._get_call_delta, T)
                result = p.starmap(f, zip(price,strike,variance))
                p.close()
                result = np.array(result)
            else:
                result = np.empty_like(price)
                for (i,inputs) in enumerate(zip(price,strike,variance)):
                    result[i] = self._get_call_delta(T,*inputs)
        if put:
            return result - 1. # by put-call parity
        else:
            return result

    def _get_call_delta(self,T,s,k,h):
        return self._P1(s,k,h,T)

    def _P1(self, price, strike, variance, T):
        P1, err = quad(self.cimpl.P1_integrand, 0., np.inf,
                args=(price, strike, variance, T))
        P1 /= (np.pi*price)
        P1 += 0.5
        return P1

    def _P2(self, price, strike, variance, T):
        P2, err = quad(self.cimpl.P2_integrand, 0., np.inf,
                args=(price, strike, variance, T))
        P2 /= np.pi
        P2 += 0.5
        return P2

    def termstruct_variance(self, h0, until):
        assert np.isscalar(h0)
        assert np.isscalar(until)
        until = int(until)
        # (1) Expectation
        terms = np.power(self.bag2, np.arange(until))
        terms = np.insert(terms, 0, 0.)
        expected_variance = (
                h0 * np.power(self.bag2, np.arange(until+1))
                + (self.omega + self.alpha)*np.cumsum(terms)
                )

        # (2) Variance
        terms = np.power(self.bag2**2., np.arange(until))
        terms = np.insert(terms, 0, 0.)
        wexpect = 1.+2.*self.gamma**2.*expected_variance
        variance_variance = np.empty_like(terms)
        for i in range(variance_variance.size):
            variance_variance[i] = np.sum(
                    terms[:i+1]*np.flipud(wexpect[:i+1])
                    )
        variance_variance *= 2.*self.alpha**2.

        return expected_variance, variance_variance

    def termstruct_logprice(self, h0, until):
        assert np.isscalar(h0)
        assert np.isscalar(until)
        until = int(until)
        expected_variance, variance_variance = \
                self.termstruct_variance(h0, until)

        # (1) Expectation
        term = np.cumsum(expected_variance[:-1])
        expected_logprice = (self.retspec.mu-0.5)*term

        # (2) Variance
        terms = np.power(self.bag2, np.arange(until))

        covariance = 0.
        correction = 0.

        cov_factor = (self.retspec.mu-0.5)**2.
        corr_factor = -4.*self.alpha*self.gamma*(self.retspec.mu-0.5)

        variance_logprice = np.cumsum(expected_variance[:-1])
        for i in range(variance_logprice.size):
            wvar = np.flipud(terms[:i+1])*variance_variance[:i+1]
            wvar[:i] *= 2.
            covariance += cov_factor*np.sum(wvar)
            variance_logprice[i] += covariance

            wexpect = np.flipud(terms[:i])*expected_variance[:i]
            correction += corr_factor*np.sum(wexpect)
            variance_logprice[i] += correction

        assert variance_logprice[0] == h0
        return expected_logprice, variance_logprice


    def _one_step_bounds(self, variances, zlb, zub):
        xlb = zlb - self.gamma*np.sqrt(variances)
        xub = zub - self.gamma*np.sqrt(variances)
        brackets_zero = np.logical_and(xlb<0., xub>0.)
        xlb **= 2.
        xub **= 2.
        low = np.minimum(xlb, xub)
        low[brackets_zero] = 0.
        high = np.maximum(xlb, xub)
        return (
                self.omega + self.beta*variances + self.alpha*low,
                self.omega + self.beta*variances + self.alpha*high
                )

    def _real_roots(self, variances, next_variances):
        shape = variances.shape
        flatten = (variances.size,)
        roots = (
                np.empty(flatten),
                np.empty(flatten)
                )
        uder = np.empty(flatten)
        self.cimpl.real_roots(
                np.ravel(variances),
                np.ravel(next_variances),
                *roots, uder)
        roots[0].shape = shape
        roots[1].shape = shape
        uder.shape = variances.shape
        return roots, uder

    def _quantized_integral(self, prev_variances, voronoi, zlb, ulb, I, d):
        self.cimpl.quantized_integral(
                prev_variances, voronoi, zlb, ulb,  # Inputs
                *I, *d  # Outputs
                )
        return I, d




cdef class _HestonNandiGarch:

    cdef:
        double mu
        double omega
        double alpha
        double beta
        double gamma
        double p1, p2, c1, c2

    def __cinit__(self,
            double mu, double omega, double alpha, double beta, double gamma):
        self.mu = mu
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
                (self.mu, self.omega, self.alpha, self.beta, self.gamma))

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef timeseries_filter(self,
            double[:,:] returns,
            double[:,:] variances,
            double[:,:] innovations
            ):
        cdef:
            double[:] volatilities
            int nper = returns.shape[0]
            int npath = returns.shape[1]
            int t,n
        volatilities = np.empty((npath,))
        for n in range(npath):
            volatilities[n] = sqrt(variances[0,n])
        for t in range(nper):
            for n in range(npath):
                innovations[t,n] = (
                        returns[t,n]-(self.mu-0.5)*variances[t,n]
                        )/volatilities[n]
                variances[t+1,n] = (
                        self.omega + self.beta*variances[t,n]
                        + self.alpha*(
                            innovations[t,n]-self.gamma*volatilities[n]
                            )**2.
                        )
                volatilities[n] = sqrt(variances[t+1,n])

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef timeseries_generate_returns(self,
            double[:,:] innovations,
            double[:,:] variances,
            double[:,:] returns
            ):
        cdef:
            double[:] volatilities
            int nper = innovations.shape[0]
            int npath = innovations.shape[1]
            int t,n
        volatilities = np.empty((npath,))
        for n in range(npath):
            volatilities[n] = sqrt(variances[0,n])
        for t in range(nper):
            for n in range(npath):
                returns[t,n] = (
                        (self.mu-0.5)*variances[t,n]
                        + volatilities[n]*innovations[t,n]
                        )
                variances[t+1,n] = (
                        self.omega + self.beta*variances[t,n]
                        + self.alpha*(
                            innovations[t,n]-self.gamma*volatilities[n]
                            )**2.
                        )
                volatilities[n] = sqrt(variances[t+1,n])

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cpdef timeseries_generate_logprices(self,
            double[:,:] innovations,
            double[:,:] variances,
            double[:,:] logprices
            ):
        cdef:
            double[:] volatilities
            double return_tmp
            int nper = innovations.shape[0]
            int npath = innovations.shape[1]
            int t,n
        volatilities = np.empty((npath,))
        for n in range(npath):
            volatilities[n] = sqrt(variances[0,n])
        for t in range(nper):
            for n in range(npath):
                return_tmp = (
                        (self.mu-0.5)*variances[t,n]
                        + volatilities[n]*innovations[t,n]
                        )
                logprices[t+1,n] = logprices[t,n] + return_tmp
                variances[t+1,n] = (
                        self.omega + self.beta*variances[t,n]
                        + self.alpha*(
                            innovations[t,n]-self.gamma*volatilities[n]
                            )**2.
                        )
                volatilities[n] = sqrt(variances[t+1,n])


    ########################
    # Heston-Nandi pricing #
    ########################

    cpdef P1_integrand(self,
            double phi, double price, double strike, double variance, int T):
        cdef:
            double complex iphi=0.
            double complex result
        iphi.imag = phi
        result = strike**(-iphi)*self._moment_generating(
                iphi+1., price, variance, T)/iphi
        return result.real

    cpdef P2_integrand(self,
            double phi, double price, double strike, double variance, int T):
        cdef:
            double complex iphi=0.
            double complex result
        iphi.imag = phi
        result = strike**(-iphi)*self._moment_generating(
                iphi, price, variance, T)/iphi
        return result.real

    cdef double complex _moment_generating(self,
            double complex phi, double price, double variance, int T):
        cdef:
            double complex A=0., B=0.
            int t
        for t in range(T):
            A += B*self.omega - 0.5*clog(1.-2.*B*self.alpha)
            B = (
                    -0.5*phi + B*(self.beta+self.alpha*self.gamma**2.)
                    + (0.5*phi**2. + 2.*B*self.alpha*self.gamma
                            *(B*self.alpha*self.gamma - phi))
                    / (1-2.*B*self.alpha)
                    )
        return price**phi*cexp(A+B*variance)

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef real_roots(self,
            # Inputs
            double[:] value,
            double[:] next_value,
            # Outputs
            double[:] left,
            double[:] right,
            double[:] unsigned_derivative,
            ):
        cdef:
            int size = value.size
            int i
        for i in range(size):
            self._real_roots(
                    value[i],
                    sqrt(value[i]),
                    next_value[i],
                    &left[i],
                    &right[i],
                    &unsigned_derivative[i]
                    )

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef quantized_integral(self,
            # Previous values
            double[:] prev_value, double[:] low, double[:] high,
            # Current values
            double[:] voronoi, #  voronoi must have size (size+1)
            # Results (prev_size, size)
            double[:,::1] I0,
            double[:,::1] I1,
            double[:,::1] I2,
            double[:,::1] D0,
            double[:,::1] D1,
            ):
        cdef:
            int prev_size = prev_value.size
            int size = voronoi.size-1
            int i
            double lb, ub

        for i in range(prev_size):
            self._do_integrate(size, &voronoi[0],
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

