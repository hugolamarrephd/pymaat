from functools import partial, wraps

import numpy as np
import scipy.optimize
from scipy.stats import norm

def print_as_vol(variance):
    print(np.sqrt(variance*252)*100)

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
    def __init__(self, model, *, nper=21, nquant=100):
        self.model = model
        self.nper = nper
        self.nquant = nquant
        self._init_innov = self._get_init_innov(self.nquant)

    def quantize(self, first_variance):
        # Check input
        first_variance = np.atleast_1d(first_variance)
        if first_variance.shape != (1,):
            raise ValueError
        grid, proba, trans = self._initialize(first_variance)
        for t in range(1,self.nper+1):
            success, grid[t] = self._one_step_quantize(grid[t-1], proba[t-1])
            if not success: # Early failure
                raise RuntimeError
            trans[t-1] = self._transition_probability(grid[t-1], grid[t])
            proba[t] = proba[t-1][np.newaxis,:].dot(trans[t-1])
            # print_as_vol(grid[t])
            # print(proba[t]*100)
            break
        # return quantization

    def _initialize(self, first_variance):
        grid = np.empty((self.nper+1, self.nquant), float)
        grid[0] = first_variance
        proba = np.empty_like(grid)
        proba[0] = 0
        proba[0,0] = 1
        trans = np.empty((self.nper, self.nquant, self.nquant), float)
        return (grid, proba, trans)

    def _one_step_quantize(self, prev_grid, prev_proba):
        func_to_optimize = partial(self._one_step_gradient_transformed,
                prev_grid,
                prev_proba)
        init_grid = self._init_grid_from_most_probable(prev_grid, prev_proba)
        init_x = self._optim_transform(init_grid, prev_grid)
        sol = scipy.optimize.newton_krylov(func_to_optimize, init_x)
                # f_tol=1e-19, line_search='wolfe')
        opt_grid = self._optim_inv_transform(np.sort(sol), prev_grid)
        return (True, opt_grid)

    def _one_step_gradient_transformed(self, prev_grid, prev_proba, x):
        # print(x)
        grid = self._optim_inv_transform(x, prev_grid)
        gradient = self._one_step_gradient(prev_grid, prev_proba, grid)
        # ddist/dx(Grid) = dsist/dgrid * dgrid/dx(grid)
        #                  = ddist/dgrid * grid
        gradient_wrt_log_grid = gradient * grid
        assert not np.any(np.isnan(gradient))
        # print(gradient_wrt_log_grid)
        # Catch limit case
        if 0.5 * (grid[0]+grid[1]) < self._get_minimal_variance(prev_grid):
            # Numerical hack to avoid leaving relevant space
            gradient_wrt_log_grid[0] = np.finfo().max
        return gradient_wrt_log_grid

    def _optim_transform(self, grid, prev_grid):
        x = np.log(grid)
        assert np.all(np.isfinite(x))
        return x

    def _optim_inv_transform(self, x, prev_grid):
        grid = np.exp(x)
        return grid

    def _one_step_gradient(self, prev_grid, prev_proba, grid):
        # Warm-up for broadcasting
        grid = grid[np.newaxis,:]
        prev_grid = prev_grid[:,np.newaxis]
        prev_proba = prev_proba[np.newaxis,:]
        # Keep grid in increasing order
        sort_id = np.argsort(grid[0])
        grid = grid[:,sort_id]
        # Compute integrals
        integrals = self._one_step_integrate(prev_grid, grid)
        # dDist/dGrid
        gradient = -2 * prev_proba.dot(integrals)
        gradient = gradient.squeeze()
        # print(gradient)
        # Put back in initial order
        rev_sort_id = np.argsort(sort_id)
        gradient = gradient[rev_sort_id]
        assert not np.any(np.isnan(gradient))
        return gradient

    def _transition_probability(self, prev_grid, grid):
        grid = grid[np.newaxis,:]
        prev_grid = prev_grid[:,np.newaxis]
        assert prev_grid.shape == (self.nquant, 1)
        assert grid.shape == (1, self.nquant)
        roots = self._get_voronoi_roots(prev_grid, grid)
        cdf = lambda z: norm.cdf(z)
        return self._over_range(cdf, roots)

    def _init_grid_from_most_probable(self, prev_grid, prev_proba):
        most_probable = np.argmax(prev_proba)
        most_probable_variance = prev_grid[most_probable]
        return self._init_grid_from(most_probable_variance)

    def _init_grid_from(self, variance_scalar):
        # Warm-up for broadcast
        variance_scalar = np.atleast_2d(variance_scalar)
        assert np.all(variance_scalar>0)
        # Initialization of quantizer (incl. probabilities)
        (grid,*_) = self.model.one_step_simulate(
                self._init_innov, variance_scalar)
        grid.sort(axis=1)
        return grid.squeeze()

    def _one_step_integrate(self, prev_grid, grid):
        assert prev_grid.shape == (self.nquant, 1)
        assert grid.shape == (1, self.nquant)
        roots = self._get_voronoi_roots(prev_grid, grid)
        model = lambda z: self.model.one_step_expectation_until(prev_grid, z)
        cdf = lambda z: norm.cdf(z)
        return (self._over_range(model, roots)
                - grid*self._over_range(cdf, roots))

    def _get_voronoi_roots(self, prev_grid, grid):
        v = self._get_voronoi(grid)
        # First (non-zero) voronoi point must be above lower variance bound
        assert v[0,1]>self._get_minimal_variance(prev_grid)
        roots = self.model.one_step_roots(prev_grid, v)
        return roots

    def _get_minimal_variance(self, prev_grid):
        return self.model.omega + self.model.beta*prev_grid[0]

    @staticmethod
    def _get_voronoi(grid):
        assert np.all(np.diff(grid)>0)
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

    @staticmethod
    def _get_init_innov(nquant):
        '''
        Returns 1-by-nquant 2D-array
        '''
        q = np.array(range(1, nquant+1), float) # (1, ..., nquant)
        q = q/(nquant+1) # (1/(nquant+1), ..., nquant/(nquant+1))
        init_innov = norm.ppf(q)
        init_innov = init_innov[np.newaxis,:]
        return init_innov
