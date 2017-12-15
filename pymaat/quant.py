from functools import partial, wraps

import numpy as np
import scipy.optimize
from scipy.stats import norm
import collections

VAR_LEVEL = 0.18**2./252.

Quantization = collections.namedtuple('Quantization',
        ['values', 'probabilities', 'transition_probabilities'])

class MarginalVarianceQuantizer():
    def __init__(self, model, *, nper=21, nquant=100):
        self.model = model
        self.nper = nper
        self.nquant = nquant

    def quantize(self, first_variance):
        # Check input
        first_variance = np.atleast_1d(first_variance)
        if first_variance.shape != (1,):
            raise ValueError

        # Initialize results
        grid, proba, trans = self._initialize(first_variance)

        # Perform marginal variance quantization
        for t in range(1,self.nper+1):
            success, fun, grid[t] = self._one_step_quantize(
                    grid[t-1], proba[t-1])
            if not success: # Early failure
                raise RuntimeError
            trans[t-1] = self._transition_probability(grid[t-1], grid[t])
            proba[t] = proba[t-1][np.newaxis,:].dot(trans[t-1])

        return Quantization(grid, proba, trans)

    def _initialize(self, first_variance):
        grid = np.empty((self.nper+1, self.nquant), float)
        grid[0] = first_variance
        proba = np.empty_like(grid)
        proba[0] = 0
        proba[0,0] = 1
        trans = np.empty((self.nper, self.nquant, self.nquant), float)
        return (grid, proba, trans)

    def _one_step_quantize(self, prev_grid, prev_proba, *,
            init=None, recursion=0, max_try=5):
        if init is None:
            init = np.zeros(self.nquant)
        # Warm-up functions
        distortion = partial(self._transformed_distortion,
                prev_grid,
                prev_proba)
        gradient = partial(self._transformed_distortion_gradient,
                prev_grid,
                prev_proba)
        hessian = partial(self._transformed_distortion_hessian,
                prev_grid,
                prev_proba)
        # We let the minimization run until the predicted reduction
        #   in distortion hits zero (or less) by setting gtol=0
        #   and expect a status flag of 2.
        sol = scipy.optimize.minimize(distortion, init,
                method='trust-ncg',
                jac=gradient,
                hess=hessian,
                options={'disp':False, 'maxiter':np.inf, 'gtol':0})
        optim_fun = sol.fun
        optim_x = sol.x
        optim_grid = self._inverse_transform(prev_grid, optim_x)
        # Catch stall gradient areas
        if recursion<max_try and np.any(sol.jac == 0.):
            #Re-initialize components having stall gradients
            optim_x[sol.jac == 0.] = 0.
            success, new_fun, new_grid = self._one_step_quantize(
                    prev_grid, prev_proba,
                    init=optim_x,
                    recursion=recursion+1,
                    max_try=max_try)
            assert new_fun<=optim_fun
            return success, new_fun, new_grid
        # Format output
        return sol.status==2, optim_fun, optim_grid

    def _transformed_distortion(self, prev_grid, prev_proba, x):
        grid = self._inverse_transform(prev_grid, x)
        grid = grid[np.newaxis,:] #1-by-nquant
        prev_grid = prev_grid[:,np.newaxis] #nquant-by-1
        prev_proba = prev_proba[np.newaxis,:] #1-by-nquant
        distortion, _, _ = self._do_one_step_distortion(
                prev_grid, prev_proba, grid)
        return distortion/VAR_LEVEL**2.

    def _transformed_distortion_gradient(self, prev_grid, prev_proba, x):
        grid = self._inverse_transform(prev_grid, x)
        grid = grid[np.newaxis,:] #1-by-nquant
        prev_grid = prev_grid[:,np.newaxis] #nquant-by-1
        prev_proba = prev_proba[np.newaxis,:] #1-by-nquant
        _, gradient, _= self._do_one_step_distortion(
                prev_grid, prev_proba, grid)
        trans_jacobian = self._trans_jacobian(prev_grid.squeeze(), x)
        gradient_wrt_x = gradient.dot(trans_jacobian).squeeze()
        return gradient_wrt_x/VAR_LEVEL**2.

    def _transformed_distortion_hessian(self, prev_grid, prev_proba, x):
        grid = self._inverse_transform(prev_grid, x)
        grid = grid[np.newaxis,:] #1-by-nquant
        prev_grid = prev_grid[:,np.newaxis] #nquant-by-1
        prev_proba = prev_proba[np.newaxis,:] #1-by-nquant
        _, gradient, hessian = self._do_one_step_distortion(
                prev_grid, prev_proba, grid)
        trans_jacobian = self._trans_jacobian(prev_grid.squeeze(), x)
        trans_hessian_corr = self._trans_hessian_correction(
                prev_grid.squeeze(), gradient, x)
        hessian_wrt_x = (trans_jacobian.T.dot(hessian).dot(trans_jacobian)
                + trans_hessian_corr)
        return  hessian_wrt_x/VAR_LEVEL**2.

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
        assert vor[0,1]>=self._get_minimal_variance(prev_grid.squeeze())
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
    def _transform(self, prev_grid, grid):
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
        x = x + np.log(self.nquant) - np.log(h_)
        assert np.all(np.isfinite(x))
        return x

    def _inverse_transform(self, prev_grid, x):
        assert x.ndim == 1
        assert prev_grid.ndim == 1
        assert x.size == self.nquant
        assert prev_grid.size == self.nquant
        h_ = self._get_minimal_variance(prev_grid)
        assert h_>0
        # Inverse transformation
        grid = np.cumsum(np.exp(x))
        grid = h_ * (1.+grid/self.nquant)
        assert np.all(np.isfinite(grid))
        assert np.all(np.diff(grid)>0)
        assert np.all(grid>0)
        return grid

    def _trans_jacobian(self, prev_grid, x):
        assert x.ndim == 1
        assert prev_grid.ndim == 1
        assert x.size == self.nquant
        assert prev_grid.size == self.nquant
        h_ = self._get_minimal_variance(prev_grid)
        assert h_>0
        all_exp = np.exp(x[np.newaxis,:])*h_/self.nquant
        mask = np.tril(np.ones((self.nquant, self.nquant)))
        return all_exp*mask

    def _trans_hessian_correction(self, prev_grid, gradient, x):
        assert prev_grid.ndim == 1
        assert gradient.ndim == 1
        assert x.ndim == 1
        assert prev_grid.size == self.nquant
        assert gradient.size == self.nquant
        assert x.size == self.nquant
        h_ = self._get_minimal_variance(prev_grid)
        all_exp = np.exp(x)*h_/self.nquant
        corr = np.zeros((self.nquant,self.nquant))
        for i in range(self.nquant):
            mask = np.zeros(self.nquant)
            mask[:i+1] = 1.
            corr += gradient[i]*np.diag(mask*all_exp)
        return corr

    def _get_minimal_variance(self, prev_grid):
        assert prev_grid.ndim == 1
        assert prev_grid.size == self.nquant
        h_ = self.model.omega + self.model.beta*prev_grid[0]
        assert h_>0
        return h_

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
