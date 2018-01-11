from functools import partial, wraps
import collections

import numpy as np
import scipy.optimize
from scipy.stats import norm
MAX_VAR_QUANT_TRY = 5

Quantization = collections.namedtuple('Quantization',
        ['values', 'probabilities', 'transition_probabilities'])

#TODO: move to utilities
class lazyproperty:
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

def quantize(self, model, first_variance, first_price=1., *,
        nper=21, nvar=100, nprice=100):
    pass

def quantize_variance(model, first_variance, *, nper=21, nquant=100):
    # Performs marginal variance quantization *only* and returns result
    # as a named-tuple containing values, probabilities and transition
    # probabilities.

    # Initialize placeholders for results
    grid = np.empty((nper+1, nquant)) #(t,i)
    proba = np.empty_like(grid) #(t,i)
    trans = np.empty((nper, nquant, nquant)) #(t,i,j)

    # Initialize first state
    state = _FirstVarianceQuantizerState(nquant, first_variance)

    for t in range(nper+1):
        # Gather results from previous state...
        grid[t] = state.grid
        proba[t] = state.probability
        if t>0:
            trans[t-1] = state.transition_probability
        # Advance to next time step (computations done here...)
        state = _OneStepVarianceQuantizer(model, nquant, state).advance()

    # Return "Quantization" named-tuple
    return Quantization(grid, proba, trans)

def get_voronoi_1d(grid, lb=-np.inf, ub=np.inf):
    voronoi = np.empty(grid.size+1)
    voronoi[0] = lb
    voronoi[1:-1] = grid[:-1] + 0.5*np.diff(grid, n=1)
    voronoi[-1] = ub
    return voronoi

# Variance Quantizer Internals

class _OneStepVarianceQuantizer():

    def __init__(self, model, nquant, prev_state):
        self.model = model
        self.nquant = nquant
        self.prev_state = prev_state
        # Initialize Cache
        self.__cache = {}

    def advance(self):
        success, optim_x = self._do_optimization()
        if not success: # Early failure if optimizer unsuccessful
            raise RuntimeError
        return self.eval_at(optim_x)

    def _do_optimization(self, init=None, try_number=0):
        if init is None:
            init = np.zeros(self.nquant)

        # We let the minimization run until the predicted reduction
        #   in distortion hits zero (or less) by setting gtol=0 and
        #   maxiter=inf. We hence expect a status flag of 2.
        sol = scipy.optimize.minimize(
                lambda x: self.eval_at(x).distortion,
                init,
                method='trust-ncg',
                jac=lambda x: self.eval_at(x).transformed_gradient,
                hess=lambda x: self.eval_at(x).transformed_hessian,
                options={'disp':False, 'maxiter':np.inf, 'gtol':0.})

        success = sol.status==2 or sol.status==0

        # Catch stall gradient areas
        if try_number<MAX_VAR_QUANT_TRY and np.any(sol.jac == 0.):
            #Re-initialize components having stall gradients
            reinit_x = sol.x.copy()
            reinit_x[sol.jac == 0.] = 0.
            success, sol.x = self._do_optimization(
                    init=reinit_x, try_number=try_number+1)

        # Format output
        return success, sol.x

    def eval_at(self, x):
        # Caching mechanism
        x.flags.writeable = False
        h = hash(x.tobytes())
        if h in self.__cache:
            assert np.all(x==self.__cache[h].x)
            return self.__cache[h]
        else:
            a_new_state =  _VarianceQuantizerState(
                    self.model, self.nquant, self.prev_state, x)
            self.__cache[h] = a_new_state
            return a_new_state

class _FirstVarianceQuantizerState():

    def __init__(self, nquant, first_variance):
        # Check input
        first_variance = np.atleast_1d(first_variance)
        if first_variance.shape != (1,):
            raise ValueError
        # Initialize
        self.grid = np.full((nquant), first_variance)
        self.probability = np.zeros_like(self.grid)
        self.probability[0] = 1.

class _VarianceQuantizerState():

    def __init__(self, model, nquant, prev_state, x):
        self.model = model
        self.nquant = nquant
        self.prev_state = prev_state
        self.x = x
        assert self.x.ndim == 1 and self.x.shape == (self.nquant,)

    @lazyproperty
    def h_min(self):
        return self.model.get_lowest_one_step_variance(
                self.prev_state.grid[0])

    @lazyproperty
    def distortion(self):
        out = np.sum(self.integrals[2]
                - 2.*self.integrals[1]*self._grid
                + self.integrals[0]*self._grid**2., axis=1)
        out = self.prev_state.probability.dot(out)
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
        out = self.prev_state.probability.dot(self.transition_probability)
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def grid(self):
        out = self.h_min + np.cumsum(self.all_exp_x)
        assert np.all(np.isfinite(out))
        assert np.all(out>0)
        assert np.all(np.diff(out)>0)
        return out

    @lazyproperty
    def voronoi(self):
        out = get_voronoi_1d(self.grid, lb=0.)
        assert out[0] == 0.
        assert out[1]>=self.h_min
        return out

    @lazyproperty
    def roots(self):
        out = self.model.one_step_roots(self._prev_grid, self._voronoi)
        return out

    @lazyproperty
    def gradient(self):
        out = -2. * self.prev_state.probability.dot(
                self.integrals[1]-self.integrals[0]*self._grid)
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def hessian(self):
        diagonal = -2. * self.prev_state.probability.dot(
                self.integral_derivatives[0][1]
                -self.integral_derivatives[0][0]*self._grid
                -self.integrals[0])
        off_diagonal = -2. * self.prev_state.probability.dot(
                self.integral_derivatives[1][1]
                -self.integral_derivatives[1][0]*self._grid)
        # Build hessian
        out = np.zeros((self.nquant, self.nquant))
        for j in range(self.nquant):
            out[j,j] = diagonal[j]
            if j>0:
                out[j-1,j] = off_diagonal[j]
                out[j,j-1] =out[j-1,j] # make symmetric
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def integrals(self):
        return [self._get_integral(order=i) for i in range(3)]

    @lazyproperty
    def pdf(self):
        return [norm.pdf(self.roots[i]) for i in range(2)]

    @lazyproperty
    def cdf(self):
        return [norm.cdf(self.roots[i]) for i in range(2)]

    def _get_integral(self, *, order):
        integral_func = self.model.one_step_expectation_until
        integral = [integral_func(self._prev_grid, self.roots[i],
                order=order, pdf=self.pdf[i], cdf=self.cdf[i])
                for i in range(2)]
        out = integral[1] - integral[0]
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
        derivative_func = self.model.one_step_roots_unsigned_derivative
        root_derivative = derivative_func(self._prev_grid, self._voronoi)
        factor = (0.5 * root_derivative
                * (self.pdf[0]+self.pdf[1]))
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
    def all_exp_x(self):
        return np.exp(self.x)*self.h_min/self.nquant

    @lazyproperty
    def jacobian(self):
        mask = np.tril(np.ones((self.nquant, self.nquant)))
        out = self.all_exp_x[np.newaxis,:]*mask
        assert not np.any(np.isnan(out))
        return out

    @lazyproperty
    def hessian_correction(self):
        out = np.zeros((self.nquant,self.nquant))
        mask = np.empty(self.nquant)
        for i in range(self.nquant):
            mask[:] = 0.
            mask[:i+1] = 1.
            out += self.gradient[i]*np.diag(mask*self.all_exp_x)
        assert not np.any(np.isnan(out))
        return out

    # Broadcast-ready formatting
    @lazyproperty
    def _prev_grid(self):
        return self.prev_state.grid[:,np.newaxis] #nquant-by-1

    @lazyproperty
    def _grid(self):
        return self.grid[np.newaxis,:] #1-by-nquant

    @lazyproperty
    def _voronoi(self):
        return self.voronoi[np.newaxis,:] #1-by-nquant

# Price Quantizer Internals

class _OneStepPriceQuantizer():
    pass

class _FirstPriceQuantizerState():
    pass

class _PriceQuantizerState():
    pass
