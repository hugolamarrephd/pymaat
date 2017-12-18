from functools import partial, wraps

import numpy as np
from scipy.stats import norm

def print_vol(daily_variance):
    print(np.sqrt(daily_variance*252)*100)

# TODO: move to utilities
def instance_returns_numpy_or_scalar(output_type):
    '''
        Change all positional arguments to np.array, but leave
            keyword arguments untouched.
    '''
    def decorator(instance_method):
        @wraps(instance_method)
        def wrapper(self, *args, **kargs):
            # Formatting input
            *args, = [np.asarray(a) for a in args]
            scalar_input = np.all([a.ndim == 0 for a in args])
            if scalar_input:
                *args, = [a[np.newaxis] for a in args] # Makes x 1D
            # Call wrapped instance method
            out = instance_method(self, *args, **kargs)
            # Formatting output
            if scalar_input:
                if isinstance(output_type, tuple):
                    return tuple(t(o) for (t,o) in zip(output_type, out))
                else:
                    return output_type(out)
            else:
                return out
        return wrapper
    return decorator

class Garch():
    def __init__(self, mu, omega, alpha, beta, gamma):
        self.mu = mu
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if (1 - self.beta - self.alpha*self.gamma**2 <= 0):
            raise ValueError

    def _one_step_equation(self, innovations, variances, volatilities):
        # h_{t+1} = omega + beta*h_{t}
        #         + alpha*(z_t-gamma*sqrt(h_{t}))^2
        return (self.omega + self.beta*variances
                + self.alpha*np.power(innovations-self.gamma*volatilities,2))

    def timeseries_filter(self, returns, first_variances):
        self._raise_value_error_if_any_invalid_variance(first_variances)
        # Initialize outputs
        innovations = np.empty_like(returns)
        variances = self._init_variances_like(returns, first_variances)
        # Apply filter
        for (t,(r,h)) in enumerate(zip(returns, variances)):
            (variances[t+1], innovations[t]) = self.one_step_filter(r,h)
        return (variances, innovations)

    def timeseries_generate(self, innovations, first_variances):
        self._raise_value_error_if_any_invalid_variance(first_variances)
        # Initialize ouputs
        returns = np.empty_like(innovations)
        variances = self._init_variances_like(innovations, first_variances)
        # Apply filter
        for (t,(z,h)) in enumerate(zip(innovations, variances)):
            (variances[t+1], returns[t]) = self.one_step_generate(z,h)
        return (variances, returns)

    @instance_returns_numpy_or_scalar(output_type=(float,float))
    def one_step_filter(self, returns, variances):
        volatilities = np.sqrt(variances)
        innovations = (returns-(self.mu-0.5)*variances)/volatilities
        next_variances = self._one_step_equation(innovations,
                variances, volatilities)
        return (next_variances, innovations)

    @instance_returns_numpy_or_scalar(output_type=(float,float))
    def one_step_generate(self, innovations, variances):
        volatilities = np.sqrt(variances)
        returns = (self.mu-0.5)*variances + volatilities*innovations
        next_variances = self._one_step_equation(innovations,
                variances, volatilities)
        return (next_variances,returns)

    @instance_returns_numpy_or_scalar(output_type=(float,float))
    def one_step_roots(self, variances, next_variances):
        d = np.sqrt((next_variances-self.omega-self.beta*variances)
                / self.alpha)
        a = self.gamma * np.sqrt(variances)
        innov_left = a-d
        innov_right = a+d
        return (innov_left, innov_right)

    @instance_returns_numpy_or_scalar(output_type=(float,float))
    def one_step_roots_unsigned_derivative(self, variances, next_variances):
        denom = np.sqrt(self.alpha
                * (next_variances-self.omega-self.beta*variances))
        denom[denom < np.finfo(float).eps] = np.nan
        der = 0.5/denom
        return der

    @instance_returns_numpy_or_scalar(output_type=float)
    def one_step_expectation_until(self, variances, innovations, order=1):
        '''
            Integrate
            ```
            (_one_step_equation(z)**order)*gaussian_pdf(z)*dz
            ```
            from -infty to innovations
        '''
        # Compute factors
        if order==0:
            cdf_factor = np.ones_like(variances)
            pdf_factor = np.zeros_like(innovations)
        elif order==1:
            cdf_factor = (self.omega + self.alpha
                    + (self.beta+self.alpha*self.gamma**2) * variances)
            pdf_factor = 2*self.gamma*np.sqrt(variances)-innovations
            pdf_factor *= self.alpha
        elif order==2:
            vol = variances**0.5
            gamma_vol = self.gamma*vol
            gamma_vol_squared = gamma_vol**2.
            innovations_squared = innovations**2.
            betah_plus_omega = self.beta*variances + self.omega
            cdf_factor = (self.alpha**2.
                    *(gamma_vol_squared*(gamma_vol_squared+6.)+3.)
                    + 2.*self.alpha*(gamma_vol_squared+1.)*betah_plus_omega
                    + betah_plus_omega**2.)
            pdf_factor = (self.alpha
                        *(2.*gamma_vol_squared*(2.*gamma_vol-3.*innovations)
                        + 4.*gamma_vol*(innovations_squared+2.)
                        - innovations*(innovations_squared+3.))
                        + 2.*(2.*gamma_vol-innovations)*betah_plus_omega)
            pdf_factor *= self.alpha
        else:
            raise ValueError('Only supports order 0,1 and 2.')
        # Limit cases (innovations = +- infinity)
        # PDF has exponential decrease towards zero that overwhelms
        # any polynomials, e.g. `(z+z**2)*exp(-z**2)->0`
        # => Set PDF factor to zero to avoid `inf*zero` indeterminations
        limit_cases = np.isinf(innovations)
        # Numpy does not seem to handle broadcasting of boolean
        # indexes...
        limit_cases = np.broadcast_to(limit_cases, pdf_factor.shape)
        pdf_factor[limit_cases] = 0
        # Compute integral
        cdf = norm.cdf(innovations)
        pdf = norm.pdf(innovations)
        return cdf_factor*cdf + pdf_factor*pdf

    @instance_returns_numpy_or_scalar(output_type=bool)
    def one_step_has_roots(self,variances, next_variances):
        return next_variances<=self._get_lowest_one_step_variance(variances)

    def _get_lowest_one_step_variance(self, variances):
        return self.omega + self.beta*variances

    # TODO: Send to estimator
    # def negative_log_likelihood(self,innovations ,variances):
    #     return 0.5 * (np.power(innovations , 2) + np.log(variances))

    @staticmethod
    def _raise_value_error_if_any_invalid_variance(var):
        if np.any(var<=0):
            raise ValueError

    @staticmethod
    def _init_variances_like(like, first_variances):
        var_shape = (like.shape[0]+1,) + like.shape[1:]
        variances = np.empty(var_shape)
        variances[0] = first_variances
        return variances
