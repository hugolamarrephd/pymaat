from functools import partial, wraps

import numpy as np
import scipy.optimize
from scipy.stats import norm


# TODO: move to utilities
def instance_returns_numpy_or_scalar(output_type):
    def decorator(instance_method):
        @wraps(instance_method)
        def wrapper(self, *args):
            # Formatting input
            *args, = [np.asarray(a) for a in args]
            scalar_input = np.all([a.ndim == 0 for a in args])
            if scalar_input:
                *args, = [a[np.newaxis] for a in args] # Makes x 1D
            # Call wrapped instance method
            out = instance_method(self, *args)
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

    def timeseries_simulate(self, innovations, first_variances):
        self._raise_value_error_if_any_invalid_variance(first_variances)
        # Initialize ouputs
        returns = np.empty_like(innovations)
        variances = self._init_variances_like(innovations, first_variances)
        # Apply filter
        for (t,(z,h)) in enumerate(zip(innovations, variances)):
            (variances[t+1], returns[t]) = self.one_step_simulate(z,h)
        return (variances, returns)

    @instance_returns_numpy_or_scalar(output_type=(float,float))
    def one_step_filter(self, returns, variances):
        volatilities = np.sqrt(variances)
        innovations = (returns-(self.mu-0.5)*variances)/volatilities
        next_variances = self._one_step_equation(innovations,
                variances, volatilities)
        return (next_variances, innovations)

    @instance_returns_numpy_or_scalar(output_type=(float,float))
    def one_step_simulate(self, innovations, variances):
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

    @instance_returns_numpy_or_scalar(output_type=float)
    def one_step_expectation_until(self, variances,
            innovations=np.inf,
            constant=None):
        '''
            Integrate `_one_step_equation(z)*gaussian_pdf(z)*dz`
            from -infty to innovations

            constant is added to the GARCH equation prior
                to integrating, i.e.
                `(_one_step_equation(z)+constant)*gaussian_pdf(z)*dz`
        '''
        # Compute factors
        cdf_factor = (self.omega + self.alpha
                + (self.beta+self.alpha*self.gamma**2) * variances)
        pdf_factor = (self.alpha *
                (2*self.gamma*np.sqrt(variances)-innovations))
        # Treat limit cases (innovations = +- infinity)
        limit_cases = np.isinf(innovations)
        limit_cases = np.logical_and(limit_cases, # Hack for broadcasting
                np.ones_like(variances, dtype=bool))
        pdf_factor[limit_cases] = 0
        # Compute integral
        if constant is None:
            return (cdf_factor*norm.cdf(innovations)
                    + pdf_factor*norm.pdf(innovations))
        else:
            return ((cdf_factor+constant)*norm.cdf(innovations)
                    + pdf_factor*norm.pdf(innovations))

    @instance_returns_numpy_or_scalar(output_type=bool)
    def one_step_has_roots(self,variances, next_variances):
        return next_variances<=self._get_lowest_one_step_variance(variances)

    def _get_lowest_one_step_variance(self,variances):
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


class Quantizer():
    def __init__(self, model, init_innov, first_variances):
        self.model = model

        if init_innov.ndim == 2 and init_innov.shape[1]>1:
            self.n_per = init_innov.shape[0]+1
            self.n_quant = init_innov.shape[1]
        else:
            raise ValueError

        # Initialization of quantizer (incl. probabilities)
        (self.gamma,*_) = self.model.timeseries_simulate(init_innov, first_variances)
        self.gamma.sort(axis=-1)
        self.proba = np.zeros_like(self.gamma)
        self.proba[:,0] = 1
        self.trans = np.zeros((self.n_per-1, self.n_quant, self.n_quant))

    def quantize(self):
        self._one_step_quantize(self.gamma[0],self.proba[0], self.gamma[1])

    def _one_step_quantize(self, prev_gamma, prev_proba, init_gamma):
        # Optimize quantizer
        func_to_optimize = partial(self._one_step_gradient,
                prev_gamma,
                prev_proba)
        opt = scipy.optimize.root(func_to_optimize, init_gamma)
        # Compute transition probabilities
        #...
        print(opt.x)
        print(opt.success)
        print(opt.message)

    def _one_step_gradient(self, prev_gamma, prev_proba, gamma):
        print('In:')
        print(gamma)
        # Keep gamma in increasing order
        sort_id = np.argsort(gamma)
        gamma = gamma[sort_id]
        g = np.empty_like(gamma) # Initialize output
        # Warm-up for broadcasting
        gamma = gamma[np.newaxis,:]
        prev_gamma = prev_gamma[:,np.newaxis]
        # Compute integrals
        voronoi = self._get_voronoi(gamma)
        i = self._one_step_integrate(prev_gamma, voronoi)
        prev_proba = prev_proba[np.newaxis,:]
        # Compute gradient and put back in initial order
        g[sort_id] = -2 * prev_proba.dot(i).squeeze()
        print('Out:')
        print(g)
        assert(not np.any(np.isnan(g)))
        return g

    def _one_step_integrate(self, prev_gamma, voronoi):
        z = self.model.one_step_roots(prev_gamma, voronoi)
        z = (np.hstack((nan_column, z_, inf_column)) for z_ in z)
        integral = (self.model.one_step_expectation_until(
                prev_gamma,
                z_,
                -voronoi)
                for z in z_)
        i = i_right - i_left
        # Evaluate integral over intervals (voronoi[i], voronoi[i+1])
        i[np.isnan(i)] = 0
        return np.diff(i, n=1, axis=-1)

    @staticmethod
    def _get_voronoi(gamma):
        # assert gamma.shape[1]>1
        zero_column = np.zeros((gamma.shape[0],1))
        inf_column = np.full((gamma.shape[0],1), np.inf)
        return np.hstack((zero_column,
                gamma[:,:-1] + 0.5 * np.diff(gamma, n=1, axis=-1),
                inf_column))
