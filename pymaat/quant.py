from functools import partial, wraps
import collections
import weakref

import numpy as np
import scipy.optimize
from scipy.stats import norm

MAX_VAR_QUANT_TRY = 5

Quantization = collections.namedtuple('Quantization',
        ['values', 'probabilities', 'transition_probabilities'])

def get_voronoi(grid, lb=-np.inf, ub=np.inf):
    voronoi = np.concatenate((np.array([lb]),
        grid[:-1] + 0.5*np.diff(grid, n=1),
        np.array([ub])))
    return voronoi

class VarianceQuantizer():
    def __init__(self, model, *, nper=21, nquant=100):
        self.model = model
        self.nper = nper
        self.nquant = nquant

    def quantize(self, first_variance):
        # Initialize placeholders for results
        grid, proba, trans = self._initialize(first_variance)

        # Perform marginal variance quantization
        for t in range(1,self.nper+1):
            # Initialize state of iteration
            state = _VarianceQuantizerState(self, grid[t-1], proba[t-1])
            success, optim_x = self._one_step_quantize(state)
            if not success: # Early failure if optimizer unsuccesfull
                raise RuntimeError
            # Gather results from state...
            grid[t] = state.get_grid(optim_x)
            trans[t-1] = state.get_transition_probability(optim_x)
            proba[t] = state.get_probability(optim_x)

        return Quantization(grid, proba, trans)

    def _initialize(self, first_variance):
        # Check input
        first_variance = np.atleast_1d(first_variance)
        if first_variance.shape != (1,):
            raise ValueError
        grid = np.empty((self.nper+1, self.nquant), float)
        grid[0] = first_variance
        proba = np.empty_like(grid)
        proba[0] = 0
        proba[0,0] = 1
        trans = np.empty((self.nper, self.nquant, self.nquant), float)
        return (grid, proba, trans)

    def _one_step_quantize(self, state, init=None, try_=0):
        if init is None:
            init = np.zeros(self.nquant)

        # We let the minimization run until the predicted reduction
        #   in distortion hits zero (or less) by setting gtol=0 and
        #   maxiter=inf. We hence expect a status flag of 2.
        sol = scipy.optimize.minimize(
                lambda x: state.get_distortion(x),
                init,
                method='trust-ncg',
                jac=lambda x: state.get_gradient(x),
                hess=lambda x: state.get_hessian(x),
                options={'disp':False, 'maxiter':np.inf, 'gtol':0})

        success = sol.status==2 or sol.status==0

        # Catch stall gradient areas
        if try_<MAX_VAR_QUANT_TRY and np.any(sol.jac == 0.):
            #Re-initialize components having stall gradients
            reinit_x = sol.x.copy()
            reinit_x[sol.jac == 0.] = 0.
            success, sol.x = self._one_step_quantize(state,
                    init=reinit_x,
                    try_=try_+1)

        # Format output
        return success, sol.x


class _VarianceQuantizerState():

    # Private class used by VarianceQuantizer

    def __init__(self, quant, prev_grid, prev_proba):
        self.nquant = quant.nquant
        self.model = quant.model
        self.prev_grid = prev_grid
        self.prev_proba = prev_proba
        self.h_min = self.model.omega + self.model.beta*self.prev_grid[0]
        # self.__cache = weakref.WeakValueDictionary()
        self.__cache = {}
        assert (self.prev_grid.ndim == 1
                and self.prev_grid.shape == (self.nquant,))
        assert (self.prev_proba.ndim == 1
                and prev_proba.shape == (self.nquant,))

    def eval_at(self, x):
        return _VarianceQuantizerEval(self, x)
        # Caching mechanism
        x.flags.writeable = False
        h = hash(x.tobytes())
        if h in self.__cache:
            assert np.all(x==self.__cache[h].x)
            return self.__cache[h]
        else:
            a_new_eval =  _VarianceQuantizerEval(self, x)
            self.__cache[h] = a_new_eval
            return a_new_eval

    def get_distortion(self, x):
        return self.eval_at(x).distortion

    def get_gradient(self, x):
        return self.eval_at(x).transformed_gradient

    def get_hessian(self, x):
        return self.eval_at(x).transformed_hessian

    def get_transition_probability(self, x):
        return self.eval_at(x).transition_probability

    def get_probability(self, x):
        return self.eval_at(x).probability

    def get_grid(self, x):
        return self.eval_at(x).grid

class lazyproperty:
    #TODO: move to utilities
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            # replace method by property...
            setattr(instance, self.func.__name__, value)
            return value

class _VarianceQuantizerEval():

    # Private class used by _VarianceQuantizerState

    def __init__(self, state, x):
        self.state = state
        self.x = x
        assert self.x.ndim == 1 and self.x.shape == (self.state.nquant,)

    @lazyproperty
    def distortion(self):
        out = np.sum(self.integrals[2]
                - 2.*self.integrals[1]*self._grid
                + self.integrals[0]*self._grid**2., axis=1)
        out = self.state.prev_proba.dot(out)
        assert not np.isnan(out)
        return out

    @lazyproperty
    def transformed_gradient(self):
        out = self.gradient.dot(self.jacobian)
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def transformed_hessian(self):
        out = (self.jacobian.T.dot(self.hessian).dot(self.jacobian)
                + self.hessian_correction)
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def transition_probability(self):
        out = self.integrals[0]
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def probability(self):
        out = self.state.prev_proba.dot(self.transition_probability)
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def grid(self):
        out = np.cumsum(np.exp(self.x))
        out = self.state.h_min * (1. + out/self.state.nquant)
        assert np.all(np.isfinite(out))
        assert np.all(out>0)
        assert np.all(np.diff(out)>0)
        return out

    @lazyproperty
    def voronoi(self):
        out = get_voronoi(self.grid, lb=0.)
        assert out[0] == 0.
        assert out[1]>=self.state.h_min
        return out

    @lazyproperty
    def roots(self):
        out = self.state.model.one_step_roots(self._prev_grid, self._voronoi)
        return out

    @lazyproperty
    def gradient(self):
        out = -2. * self.state.prev_proba.dot(
                self.integrals[1]-self.integrals[0]*self._grid)
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def hessian(self):
        diagonal = -2. * self.state.prev_proba.dot(
                self.integral_derivatives[0][1]
                -self.integral_derivatives[0][0]*self._grid
                -self.integrals[0])
        off_diagonal = -2. * self.state.prev_proba.dot(
                self.integral_derivatives[1][1]
                -self.integral_derivatives[1][0]*self._grid)
        # Build hessian
        out = np.zeros((self.state.nquant, self.state.nquant))
        for j in range(self.state.nquant):
            out[j,j] = diagonal[j]
            if j>0:
                out[j-1,j] = off_diagonal[j]
                out[j,j-1] =out[j-1,j] # make symmetric
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def integrals(self):
        return [self._get_integral(order=i) for i in range(3)]

    def _get_integral(self, *, order):
        integral_func = lambda z: self.state.model.one_step_expectation_until(
            self._prev_grid, z, order=order)
        out = integral_func(self.roots[1]) - integral_func(self.roots[0])
        out[np.isnan(out)] = 0
        out = np.diff(out, n=1, axis=1)
        return out

    @lazyproperty
    def integral_derivatives(self):
        factor = self._get_factor_for_integral_derivative()
        dI = [self._get_integral_derivative(factor, order=i, lag=0)
                for i in range(2)]
        dI_lagged = [self._get_integral_derivative(factor, order=i, lag=-1)
                for i in range(2)]
        return dI, dI_lagged

    def _get_factor_for_integral_derivative(self):
        root_derivative = self.state.model.one_step_roots_unsigned_derivative(
                self._prev_grid, self._voronoi)
        factor = (0.5 * root_derivative
                * (norm.pdf(self.roots[0])+norm.pdf(self.roots[1])))
        factor[np.isnan(factor)] = 0
        return factor

    def _get_integral_derivative(self, factor, order=0, lag=0):
        if order==0:
            d = factor
        elif order==1:
            d = factor * self._voronoi
        else:
            d = factor * self._voronoi**order
        d[np.isnan(d)] = 0
        if lag==-1:
            d = -d[:,:-1]
            d[:,0] = np.nan # first column is meaningless
        elif lag==0:
            d = np.diff(d, n=1, axis=1)
        elif lag==1:
            d = d[:,1:]
            d[:,-1] = np.nan # last column is meaningless
        else:
            assert False # should never happen
        return d

    @lazyproperty
    def jacobian(self):
        all_exp = np.exp(self._x)*self.state.h_min/self.state.nquant
        mask = np.tril(np.ones((self.state.nquant, self.state.nquant)))
        out = all_exp*mask
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def hessian_correction(self):
        all_exp = np.exp(self.x)*self.state.h_min/self.state.nquant
        out = np.zeros((self.state.nquant,self.state.nquant))
        for i in range(self.state.nquant):
            mask = np.zeros(self.state.nquant)
            mask[:i+1] = 1.
            out += self.gradient[i]*np.diag(mask*all_exp)
        assert not np.any(np.isnan(out))
        return out

    # Broadcast-ready formatting

    @lazyproperty
    def _prev_grid(self):
        return self.state.prev_grid[:,np.newaxis] #nquant-by-1

    @lazyproperty
    def _x(self):
        return self.x[np.newaxis,:] #1-by-nquant

    @lazyproperty
    def _grid(self):
        return self.grid[np.newaxis,:] #1-by-nquant

    @lazyproperty
    def _voronoi(self):
        return self.voronoi[np.newaxis,:]
