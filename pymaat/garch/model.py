from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

from pymaat.nputil import atleast_1d, elbyel
from pymaat.util import method_decorator

def get_garch_factory(retype='', vartype='hngarch'):
    #TODO
    pass

class AbstractOneLagReturn(ABC):

    def one_step_generate(self, innovations, variances):
        volatilities = np.sqrt(variances)
        return self._one_step_generate(innovations, variances, volatilities)

    def one_step_filter(self, returns, variances):
        volatilities = np.sqrt(variances)
        return self._one_step_filter(returns, variances, volatilities)

    def root_price_derivative(self, prices, variances):
        volatilities = np.sqrt(variances)
        with np.errstate(divide='ignore'):
            dX = self._root_price_derivative(
                    prices, variances, volatilities)
        # Limit cases
        # (1) Zero
        limit_index = np.broadcast_to(prices==0., dX.shape)
        dX[limit_index] = np.inf
        # (2) Inf
        limit_index = np.broadcast_to(prices==np.inf, dX.shape)
        dX[limit_index] = 0.
        # (3) Negative 
        limit_index = np.broadcast_to(prices<0., dX.shape)
        dX[limit_index] = np.nan
        return dX

    def price_integral_until(self, prices, variances, innovations, order=0):
        if order==0:
            return norm.cdf(innovations)
        elif order==1:
            volatilities = np.sqrt(variances)
            return self._first_order_integral(
                    prices, variances, innovations, volatilities)
        elif order==2:
            volatilities = np.sqrt(variances)
            return self._second_order_integral(
                    prices, variances, innovations, volatilities)
        else:
            raise ValueError('Only supports order 0, 1 and 2.')

    @abstractmethod
    def _one_step_generate(self, innovations, variances, volatilities):
     pass

    @abstractmethod
    def _one_step_filter(self, returns, variances, volatilities):
     pass

    @abstractmethod
    def _root_price_derivative(self, prices, variances, volatilities):
     pass

    @abstractmethod
    def _first_order_integral(self, variances, innovations):
        pass

    @abstractmethod
    def _second_order_integral(self, variances, innovations):
        pass


class AbstractOneLagGarch(ABC):

    def __init__(self, retspec):
        self.retspec = retspec

    @abstractmethod
    def _equation(self, innovations, variances, volatilities):
        pass

    @abstractmethod
    def _real_roots(self, variances, next_variances):
        pass

    @abstractmethod
    def _real_roots_unsigned_derivative(self, variances, next_variances):
        pass

    @abstractmethod
    def _first_order_integral_factors(self, variances, innovations):
        pass

    @abstractmethod
    def _second_order_integral_factors(self, variances, innovations):
        pass

    ##############
    # Timeseries #
    ##############

    @method_decorator(atleast_1d)
    def timeseries_filter(self, returns, first_variances):
        self._raise_value_error_if_any_invalid_variance(first_variances)
        # Initialize outputs
        variances = self._init_variances_like(returns, first_variances)
        innovations = np.empty_like(returns)
        # Do computations and return
        return self._timeseries_filter(returns, variances, innovations)

    def _timeseries_filter(self, returns, variances, innovations):
        """
        Performs filtering of returns from
            pre-allocated output arrays (variances, innovations).
        Rem. This is provided as a convenience only.
            Implementing classes are strongly encouraged to override
            (eg using cython) for efficiency.
        """
        for (t,(r,h)) in enumerate(zip(returns, variances)):
            (variances[t+1], innovations[t]) = self.one_step_filter(r,h)
        return (variances, innovations)

    @method_decorator(atleast_1d)
    def timeseries_generate(self, innovations, first_variances):
        self._raise_value_error_if_any_invalid_variance(first_variances)
        # Initialize ouputs
        variances = self._init_variances_like(innovations, first_variances)
        returns = np.empty_like(innovations)
        return self._timeseries_generate(innovations, variances, returns)

    def _timeseries_generate(self, innovations, variances, returns):
        """
        Generates time-series from innovations and
            pre-allocated output arrays (variances, returns).
        Rem. This is provided as a convenience only.
            Implementing classes are strongly encouraged to override
            (eg using cython) for efficiency.
        """
        for (t,(z,h)) in enumerate(zip(innovations, variances)):
            (variances[t+1], returns[t]) = self.one_step_generate(z,h)
        return (variances, returns)

    ############
    # One step #
    ############

    # TODO: send this to spec
    @method_decorator(elbyel)
    def get_lowest_one_step_variance(
            self, variances, lb=-np.inf, ub=np.inf):
        adj_lb = lb - self.gamma*np.sqrt(variances)
        adj_ub = ub - self.gamma*np.sqrt(variances)
        out = np.minimum(adj_lb**2., adj_ub**2.)
        out[np.logical_and(adj_lb<0., adj_ub>0.)] = 0.
        return self.omega + self.beta*variances + self.alpha*out

    @method_decorator(elbyel)
    def get_highest_one_step_variance(
            self, variances, lb=-np.inf, ub=np.inf):
        adj_lb = lb - self.gamma*np.sqrt(variances)
        adj_ub = ub - self.gamma*np.sqrt(variances)
        out = np.maximum(adj_lb**2., adj_ub**2.)
        return self.omega + self.beta*variances + self.alpha*out

    @method_decorator(elbyel)
    def one_step_filter(self, returns, variances):
        volatilities = np.sqrt(variances)
        innovations = self.retspec._one_step_filter(
                returns, variances, volatilities)
        next_variances = self._equation(innovations,
                variances, volatilities)
        return (next_variances, innovations)

    @method_decorator(elbyel)
    def one_step_generate(self, innovations, variances):
        volatilities = np.sqrt(variances)
        returns = self.retspec._one_step_generate(
                innovations, variances, volatilities)
        next_variances = self._equation(innovations,
                variances, volatilities)
        return (next_variances,returns)

    @method_decorator(elbyel)
    def real_roots(self, variances, next_variances):
        roots = self._real_roots(variances, next_variances)
        # Limit cases
        limit_index = np.broadcast_to(
                next_variances==np.inf, roots[0].shape)
        roots[0][limit_index] = -np.inf
        roots[1][limit_index] = np.inf
        # Output tuple
        return tuple(roots)

    @method_decorator(elbyel)
    def real_roots_unsigned_derivative(
            self, variances, next_variances):
        # Warning: returns zero at singularity
        #    although the theoritical value is inf
        dX = self._real_roots_unsigned_derivative(
                variances, next_variances)
        # Limit cases
        limit_index = np.broadcast_to(
                next_variances==np.inf, dX.shape)
        dX[limit_index] = 0.
        return dX

    @method_decorator(elbyel)
    def variance_integral_until(self, variances, innovations, *,
            order=0, _pdf=None, _cdf=None):
        """
        Integrate
            ```
            (_equation(z)**order)*gaussian_pdf(z)*dz
            ```
        from -infty until innovations.
        Rem. _pdf and _cdf may be used for efficiency.
        """
        if _pdf is None:
            _pdf = norm.pdf(innovations)
        if _cdf is None:
            _cdf = norm.cdf(innovations)
        pdf_factor, cdf_factor = self._variance_integral_factors(
                        variances, innovations, order=order)
        return pdf_factor*_pdf + cdf_factor*_cdf

    def _variance_integral_factors(self, variances, innovations, *,
            order=0):
        # Limit cases (innovations = +- infinity)
        #   PDF has exponential decrease towards zero that overwhelms
        #   any polynomials, e.g. `(z+z**2)*exp(-z**2)->0`
        limit_cases = np.isinf(innovations)
        # Compute PDF and CDF factors
        if order==0:
            pdf_factor = np.zeros_like(innovations)
            cdf_factor = np.ones_like(variances)
        elif order==1:
            (pdf_factor, cdf_factor) = \
                    self._first_order_integral_factors(
                    variances, innovations)
        elif order==2:
            (pdf_factor, cdf_factor) = \
                    self._second_order_integral_factors(
                    variances, innovations)
        else:
            raise ValueError('Only supports order 0,1 and 2.')
        # Treat limit cases
        limit_cases = np.broadcast_to(limit_cases, pdf_factor.shape)
        # Set PDF factor to zero to avoid `inf*zero` indetermination
        pdf_factor[limit_cases] = 0
        return (pdf_factor, cdf_factor)

    #############
    # Utilities #
    #############

    @staticmethod
    def _raise_value_error_if_any_invalid_variance(var):
        if np.any(np.isnan(var)) or np.any(var<=0):
            raise ValueError("Invalid variance detected.")

    @staticmethod
    def _init_variances_like(like, first_variances):
        var_shape = (like.shape[0]+1,) + like.shape[1:]
        variances = np.empty(var_shape)
        variances[0] = first_variances # Might broadcast here
        return variances

    # TODO: Send to estimator
    # def negative_log_likelihood(self,innovations ,variances):
    #     return 0.5 * (np.power(innovations , 2) + np.log(variances))
