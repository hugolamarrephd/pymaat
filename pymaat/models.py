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
    def one_step_expectation_until(self, variances, innovations=np.inf):
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


class Quantizer():
    def __init__(self, model, init_innov, first_variances):
        self.model = model

        if init_innov.ndim == 2 and init_innov.shape[1]>1:
            self.n_per = init_innov.shape[0]+1
            self.n_quant = init_innov.shape[1]
        else:
            raise ValueError

        # Initialization of quantizer (incl. probabilities)
        (self.grid,*_) = self.model.timeseries_simulate(init_innov, first_variances)
        self.grid.sort(axis=1)
        self.proba = np.zeros_like(self.grid)
        self.proba[:,0] = 1
        self.trans = np.zeros((self.n_per-1, self.n_quant, self.n_quant))

    def quantize(self):
        for (prev_grid, prev_proba, grid, trans, proba) in zip(
                self.grid[:-1],
                self.proba[:-1],
                self.grid[1:],
                self.trans,
                self.proba[1:]):
            grid[:] = self._one_step_quantize_or_runtime_error(
                    prev_grid, prev_proba, grid) # [:] to fill row view
            trans[:,:] = self._transition_probability(prev_grid, grid)
            proba[:] = prev_proba.dot(trans)
            np.set_printoptions(linewidth=200)
            print(np.sqrt(252*grid)*100)
            # print(trans*100)
            # print(proba*100)

    def _one_step_quantize_or_runtime_error(self,
            prev_grid, prev_proba, init_grid):
        func_to_optimize = partial(self._one_step_gradient,
                prev_grid,
                prev_proba)
        opt = scipy.optimize.root(func_to_optimize, np.log(init_grid))
        # if opt.success:
        # print(opt.message)
        out = np.sort(opt.x)
        return np.exp(out)
        # else:
            # raise RuntimeError

    def _one_step_gradient(self, prev_grid, prev_proba, log_grid):
        grid = np.exp(log_grid)
        # Warm-up for broadcasting
        grid = grid[np.newaxis,:]
        prev_grid = prev_grid[:,np.newaxis]
        prev_proba = prev_proba[np.newaxis,:]
        # Keep grid in increasing order
        sort_id = np.argsort(grid[0])
        grid = grid[:,sort_id]
        # Compute integrals
        integrals = self._one_step_integrate(prev_grid, grid)
        # Compute gradient
        gradient = -2 * prev_proba.dot(integrals)
        # Compte dDist/dlog(Grid) = dDist/dGrid dGrid/dlog(Grid)
        gradient = gradient[0] * grid[0]
        # Put back in initial order
        rev_sort_id = np.argsort(sort_id)
        gradient = gradient[rev_sort_id]
        assert not np.any(np.isnan(gradient))
        # print(gradient)
        # print('-')
        return gradient

    def _transition_probability(self, prev_grid, grid):
        grid = grid[np.newaxis,:]
        prev_grid = prev_grid[:,np.newaxis]
        assert prev_grid.shape == (self.n_quant, 1)
        assert grid.shape == (1, self.n_quant)
        roots = self._get_voronoi_roots(prev_grid, grid)
        cdf = lambda z: norm.cdf(z)
        return self._over_range(cdf, roots)

    def _one_step_integrate(self, prev_grid, grid):
        assert prev_grid.shape == (self.n_quant, 1)
        assert grid.shape == (1, self.n_quant)
        roots = self._get_voronoi_roots(prev_grid, grid)
        model = lambda z: self.model.one_step_expectation_until(prev_grid, z)
        cdf = lambda z: norm.cdf(z)
        return (self._over_range(model, roots)
                - grid*self._over_range(cdf, roots))

    def _get_voronoi_roots(self, prev_grid, grid):
        v = self._get_voronoi(grid)
        roots = self.model.one_step_roots(prev_grid, v)
        return roots

    @staticmethod
    def _get_voronoi(grid):
        assert np.all(grid>0)
        zero_column = np.zeros((grid.shape[0],1))
        inf_column = np.full((grid.shape[0],1), np.inf)
        return np.hstack((zero_column,
                grid[:,:-1] + 0.5 * np.diff(grid, n=1, axis=-1),
                inf_column))

    @staticmethod
    def _over_range(integral, roots):
        out = integral(roots[1]) - integral(roots[0])
        out[np.isnan(out)] = 0
        out = np.diff(out, n=1, axis=1)
        return out
