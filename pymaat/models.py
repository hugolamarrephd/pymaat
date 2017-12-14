from functools import partial, wraps

import numpy as np
import scipy.optimize
from scipy.stats import norm

VAR_LEVEL = 0.18**2./252.

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

    @instance_returns_numpy_or_scalar(output_type=(float,float))
    def one_step_roots_unsigned_derivative(self, variances, next_variances):
        denom = np.sqrt(self.alpha
                * (next_variances-self.omega-self.beta*variances))
        denom[denom < np.finfo(float).eps] = np.nan
        der = 0.5/denom
        return der

    @instance_returns_numpy_or_scalar(output_type=float)
    def one_step_expectation_until(self, variances, innovations,
            order=1):
        '''
            Integrate
            ```
            (_one_step_equation(z)**order)*gaussian_pdf(z)*dz
            ```
            from -infty to innovations
        '''
        # Compute factors
        if order==1:
            cdf_factor_func = self._one_step_expectation_cdf_factor
            pdf_factor_func = self._one_step_expectation_pdf_factor
        elif order==2:
            cdf_factor_func = self._one_step_expectation_squared_cdf_factor
            pdf_factor_func = self._one_step_expectation_squared_pdf_factor
        else:
            raise ValueError
        cdf_factor = cdf_factor_func(variances)
        pdf_factor = pdf_factor_func(variances, innovations)
        # Limit cases (innovations = +- infinity)
        # PDF has exponential decrease towards zero that overwhelms
        # any polynomials, e.g. `(z+z**2)*exp(-z**2)->0`
        # => Set PDF factor to zero to avoid `inf*zero` indeterminations
        limit_cases = np.isinf(innovations)
        limit_cases = np.logical_and(limit_cases, # Hack for broadcasting
                np.ones_like(variances, dtype=bool))
        pdf_factor[limit_cases] = 0
        # Compute integral
        cdf = norm.cdf(innovations)
        pdf = norm.pdf(innovations)
        return cdf_factor*cdf + pdf_factor*pdf

    def _one_step_expectation_cdf_factor(self, variances):
        return (self.omega + self.alpha
                    + (self.beta+self.alpha*self.gamma**2) * variances)

    def _one_step_expectation_pdf_factor(self, variances, innovations):
        return (self.alpha *
                    (2*self.gamma*np.sqrt(variances)-innovations))

    def _one_step_expectation_squared_cdf_factor(self, variances):
        gamma_vol_squared = self.gamma**2.*variances
        betah_plus_omega = self.beta*variances + self.omega
        return (self.alpha**2.*(gamma_vol_squared*(gamma_vol_squared+6.)+3.)
                + 2.*self.alpha*(gamma_vol_squared+1.)*betah_plus_omega
                + betah_plus_omega**2.)

    def _one_step_expectation_squared_pdf_factor(self, variances, innovations):
        vol = variances**0.5
        gamma_vol = self.gamma*vol
        gamma_vol_squared = gamma_vol**2.
        innovations_squared = innovations**2.
        betah_plus_omega = self.beta*variances + self.omega
        return (self.alpha*(
            self.alpha*(2.*gamma_vol_squared*(2.*gamma_vol-3.*innovations)
            + 4.*gamma_vol*(innovations_squared+2.)
            - innovations*(innovations_squared+3.))
            + 2.*(2.*gamma_vol-innovations)*betah_plus_omega))

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
            print_vol(grid[t])
        # return quantization

    def _initialize(self, first_variance):
        grid = np.empty((self.nper+1, self.nquant), float)
        grid[0] = first_variance
        proba = np.empty_like(grid)
        proba[0] = 0
        proba[0,0] = 1
        trans = np.empty((self.nper, self.nquant, self.nquant), float)
        return (grid, proba, trans)

    def _one_step_quantize(self, prev_grid, prev_proba, *,
            init=None):
        if init is None:
            init = np.zeros((self.nquant))
        distortion = partial(
                self._one_step_distortion,
                prev_grid,
                prev_proba)
        opt = {'disp':True,
               'maxiter':np.inf,
               'gtol':1e-6}
        sol = scipy.optimize.minimize(distortion, init,
                method='trust-ncg',
                jac=True,
                hess=True,
                options=opt)
        # Format output
        opt_grid = self._optim_inv_transform(prev_grid, sol.x)
        return (sol.success, opt_grid)

    def _one_step_distortion(self, prev_grid, prev_proba, x):
        grid = self._optim_inv_transform(prev_grid, x)
        # Warm-up for broadcasting
        grid = grid[np.newaxis,:] #1-by-nquant
        prev_grid = prev_grid[:,np.newaxis] #nquant-by-1
        prev_proba = prev_proba[np.newaxis,:] #1-by-nquant
        # Compute Gradient
        distortion, gradient, hessian = self._do_one_step_distortion(
                prev_grid, prev_proba, grid)
        jacobian = self._optim_jacobian(prev_grid, x)
        gradient_wrt_x = gradient.dot(jacobian)
        hessian_wrt_x = (jacobian.T.dot(hessian).dot(jacobian) +
            gradient.dot(np.diag(np.diag(jacobian))))
        norm = VAR_LEVEL**2.
        return (distortion/norm, gradient_wrt_x/norm, hessian_wrt_x/norm)

    def _transition_probability(self, prev_grid, grid):
        grid = grid[np.newaxis,:]
        prev_grid = prev_grid[:,np.newaxis]
        assert prev_grid.shape == (self.nquant, 1)
        assert grid.shape == (1, self.nquant)
        roots = self._get_voronoi_roots(prev_grid, grid)
        cdf = lambda z: norm.cdf(z)
        return self._over_range(cdf, roots)

    def _do_one_step_distortion(self, prev_grid, prev_proba, grid):
        assert prev_grid.shape == (self.nquant, 1)
        assert grid.shape == (1, self.nquant)
        vor = self._get_voronoi(grid)
        # First (non-zero) voronoi point must be above lower variance bound
        assert vor[0,1]>=self._get_minimal_variance(prev_grid)
        roots = self._get_roots(prev_grid, vor)
        # Compute integrals
        I0 = self._get_integral(prev_grid, roots, order=0)
        I1 = self._get_integral(prev_grid, roots, order=1)
        I2 = self._get_integral(prev_grid, roots, order=2)
        # Distortion
        distortion = np.sum(I2 - 2.*I1*grid + I0*grid**2., axis=1)
        distortion = prev_proba.dot(distortion)
        distortion = distortion.squeeze()
        assert not np.isnan(distortion)
        # Gradient
        gradient = -2. * prev_proba.dot(I1-I0*grid)
        gradient = gradient.squeeze()
        assert not np.any(np.isnan(gradient))
        # Hessian
        factor = self._get_factor_of_integral_derivative(
                prev_grid, vor, roots)
        d0 = self._get_integral_derivative(factor, vor)
        d1 = self._get_integral_derivative(factor, vor, order=1)
        diagonal = -2. * prev_proba.dot(d1-d0*grid-I0)
        diagonal = diagonal.squeeze()
        d0 = self._get_integral_derivative(factor, vor, lag=-1)
        d1 = self._get_integral_derivative(factor, vor, order=1, lag=-1)
        off_diagonal = -2 * prev_proba.dot(d1-d0*grid)
        off_diagonal = off_diagonal.squeeze()
        hessian = np.zeros((self.nquant, self.nquant))
        for j in range(self.nquant):
            #TODO VECTORIZE
            hessian[j,j] = diagonal[j]
            if j>0:
                hessian[j-1,j] = off_diagonal[j]
                hessian[j,j-1] = hessian[j-1,j] # make symmetric
        assert not np.any(np.isnan(hessian))
        return distortion, gradient, hessian

    def _get_integral(self, prev_grid, roots, order=0):
        if order==0:
            integral_func = lambda z: norm.cdf(z)
        else:
            integral_func = lambda z: self.model.one_step_expectation_until(
                prev_grid, z, order=order)
        return self._over_range(integral_func, roots)

    def _get_integral_derivative(self, factor, vor, order=0, lag=0):
        if order==0:
            d = factor
        elif order==1:
            d = factor * vor
        else:
            d = factor * vor**order
        d[np.isnan(d)] = 0
        if lag==-1:
            d = -d[:,:-1]
            d[:,0] = np.nan # first column is meaningless
            return d
        elif lag==0:
            return np.diff(d, n=1, axis=1)
        elif lag==1:
            d = d[:,1:]
            d[:,-1] = np.nan # last column is meaningless
            return d

    def _get_factor_of_integral_derivative(self, prev_grid, vor, roots):
        root_derivative = self._get_roots_derivatives(prev_grid, vor)
        factor = (0.5 * root_derivative
                * (norm.pdf(roots[0])+norm.pdf(roots[1])))
        factor[np.isnan(factor)] = 0
        return factor

    def _get_voronoi_roots(self, prev_grid, grid):
        vor = self._get_voronoi(grid)
        return self._get_roots(prev_grid, vor)

    def _get_roots(self, prev_grid, vor):
        return self.model.one_step_roots(prev_grid, vor)

    def _get_roots_derivatives(self, prev_grid, vor):
        return self.model.one_step_roots_unsigned_derivative(prev_grid, vor)

    # TODO: wrap 3 following methods in reparametrization object
    def _optim_transform(self, prev_grid, grid):
        assert grid.ndim == 1
        assert prev_grid.ndim == 1
        assert grid.size == self.nquant
        assert prev_grid.size == self.nquant
        assert np.all(np.isfinite(grid))
        assert np.all(np.diff(grid)>0)
        assert np.all(grid>0)
        h_ = self._get_minimal_variance(prev_grid)
        assert h_>0
        # Transformation
        x = np.empty_like(grid)
        x[0] = np.log(grid[0]-h_)
        x[1:] = np.log(grid[1:]-grid[0:-1])
        assert np.all(np.isfinite(x))
        return x

    def _optim_inv_transform(self, prev_grid, x):
        assert x.ndim == 1
        assert prev_grid.ndim == 1
        assert x.size == self.nquant
        assert prev_grid.size == self.nquant
        h_ = self._get_minimal_variance(prev_grid)
        assert h_>0
        # Inverse transformation
        grid = np.cumsum(np.exp(x))
        grid = grid + h_
        assert np.all(np.isfinite(grid))
        assert np.all(np.diff(grid)>0)
        assert np.all(grid>0)
        return grid

    def _optim_jacobian(self, prev_grid, x):
        all_exp = np.exp(x[np.newaxis,:])
        mask = np.tril(np.ones((self.nquant, self.nquant), float))
        return all_exp*mask

    def _get_minimal_variance(self, prev_grid):
        return self.model.omega + self.model.beta*prev_grid[0]

    @staticmethod
    def _get_voronoi(grid):
        assert np.all(np.diff(grid)>=0)
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
