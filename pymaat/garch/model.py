from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

from pymaat.nputil import atleast_1d, elbyel
from pymaat.util import method_decorator

def get_garch_factory(retype='', vartype='hngarch'):
    #TODO
    pass

class AbstractOneLagGarch(ABC):

    # TODO: send to return specification
    @abstractmethod
    def _one_step_return_equation(self, innovations, variances, volatilities):
        pass

    @abstractmethod
    def _one_step_innovation_equation(self, returns, variances, volatilities):
        pass

    # TODO: send to variance specification
    @abstractmethod
    def _one_step_equation(self, innovations, variances, volatilities):
        pass

    @abstractmethod
    def _get_one_step_roots(self, variances, next_variances):
        pass

    @abstractmethod
    def _get_one_step_roots_unsigned_derivative(self,
            variances, next_variances):
        pass

    @abstractmethod
    def _get_one_step_first_order_expectation_factors(self,
            variances, innovations):
        pass

    @abstractmethod
    def _get_one_step_second_order_expectation_factors(self,
            variances, innovations):
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

    @method_decorator(elbyel)
    def get_lowest_one_step_variance(self, variances):
        return self.omega + self.beta*variances

    @method_decorator(elbyel)
    def one_step_filter(self, returns, variances):
        volatilities = np.sqrt(variances)
        innovations = self._one_step_innovation_equation(returns,
                variances, volatilities)
        next_variances = self._one_step_equation(innovations,
                variances, volatilities)
        return (next_variances, innovations)

    @method_decorator(elbyel)
    def one_step_generate(self, innovations, variances):
        volatilities = np.sqrt(variances)
        returns = self._one_step_return_equation(innovations,
                variances, volatilities)
        next_variances = self._one_step_equation(innovations,
                variances, volatilities)
        return (next_variances,returns)

    @method_decorator(elbyel)
    def one_step_roots(self, variances, next_variances):
        limit_index = next_variances==np.inf
        # Complex roots are silently converted to NaNs.
        # For masked arrays, factor = np.inf gets automatically
        #    masked here
        roots = self._get_one_step_roots(variances, next_variances)
        # Handle limit case explicitly to support masked arrays
        limit_index = np.broadcast_to(limit_index, roots[0].shape)
        roots[0][limit_index] = -np.inf
        roots[1][limit_index] = np.inf
        # Output tuple
        return tuple(roots)

    @method_decorator(elbyel)
    def one_step_roots_unsigned_derivative(self, variances, next_variances):
        discr = next_variances-self.get_lowest_one_step_variance(variances)
        # Register limit cases
        limit_cases = {0.:next_variances==np.inf, np.inf:discr==0.}
        # Complex roots and division by zero
        #   are silently converted to NaNs.
        # For masked arrays, factor = 0. *and* np.inf may
        #   get automatically masked
        der = self._get_one_step_roots_unsigned_derivative(
                variances, next_variances)
        # Handle limit case explicitly to support masked arrays
        for lim, index in limit_cases.items():
            der[np.broadcast_to(index, der.shape)] = lim
        return der

    @method_decorator(elbyel)
    def one_step_expectation_until(self, variances, innovations, *,
            order=0, _pdf=None, _cdf=None):
        """
        Integrate
            ```
            (_one_step_equation(z)**order)*gaussian_pdf(z)*dz
            ```
        from -infty until innovations.
        Rem. _pdf and _cdf are used internally for efficiency.
        """
        if _pdf is None:
            _pdf = norm.pdf(innovations)
        if _cdf is None:
            _cdf = norm.cdf(innovations)
        pdf_factor, cdf_factor = self._get_one_step_expectation_factors(
                        variances, innovations, order=order)
        return pdf_factor*_pdf + cdf_factor*_cdf

    def _get_one_step_expectation_factors(self, variances, innovations, *,
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
                    self._get_one_step_first_order_expectation_factors(
                    variances, innovations)
        elif order==2:
            (pdf_factor, cdf_factor) = \
                    self._get_one_step_second_order_expectation_factors(
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
