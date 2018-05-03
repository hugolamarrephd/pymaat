from abc import ABC, abstractmethod
from cachetools import LRUCache

import numpy as np
from scipy.linalg import norm as infnorm
from scipy.optimize import minimize, basinhopping

from pymaat.util import PymaatException
from pymaat.util import lazy_property, method_decorator, no_exception
from pymaat.nputil import workon_axis, diag_view, printoptions
from pymaat.nputil import icumsum
from pymaat.mathutil import logistic, dlogistic, ddlogistic
import pymaat.testing as pt

#############
# Utilities #
#############


def _check_value(value, bounds):
    if np.any(np.logical_not(np.isfinite(value))):
        raise ValueError('Invalid value: NaNs or infinite')
    elif np.any(np.logical_or(
            value <= bounds[0],
            value >= bounds[1])):
        raise ValueError('Invalid value: must be within bounds')


def _check_proba(probability):
    if np.any(np.logical_not(np.isfinite(probability))):
        raise ValueError('Invalid probabilities: NaNs or infinite')
    elif np.abs(np.sum(probability)-1.) > 1e-12:
        raise ValueError('Invalid probabilities: must sum to one')
    elif np.any(probability <= 0.):
        raise ValueError(
            'Invalid probabilities: must be striclty positive')


def _check_trans(transition_probability, ax=(1,)):
    if np.any(np.logical_not(np.isfinite(transition_probability))):
        raise ValueError('Invalid transition probabilities:'
                         'NaNs or infinite')
    elif np.any(transition_probability < 0.):
        raise ValueError('Invalid transition probabilities:'
                         ' must be positive')
    elif np.any(
            np.abs(np.sum(transition_probability, axis=ax)-1.)
            > 1e-12
    ):
        raise ValueError('Invalid transition probabilities:'
                         ' must sum to one')


def _check_match_shape(*args):
    s = None
    for a in args:
        if s is None:
            s = a.shape
        elif a.shape != s:
            raise ValueError('Shape mismatch')


def _check_dist(distortion):
    if distortion < 0.:
        raise ValueError("Invalid distortion: must be positive")


class AbstractCore(ABC):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def print_formatter(self, value): return value

    def optimize(self):
        # Warm-up
        all_shapes = self._get_all_shapes()
        do_advance = self._advance
        if self.verbose:  # Decorate with printing utility
            do_advance = self._print_decorator(do_advance)
        # Recursion
        q = self._initialize()
        for (self._t, s) in enumerate(all_shapes):
            q = do_advance(s, q)
        self._terminalize(q)
        return self  # for convenience

    def _initialize(self):
        self.all_quant = []
        return self._make_first_quant()

    def _advance(self, shape, quantizer):
        # Gather results from previous iteration...
        self.all_quant.append(quantizer)
        # Advance to next time step (computations done here...)
        return self._one_step_optimize(shape, quantizer)

    def _terminalize(self, quantizer):
        self.all_quant.append(quantizer)

    def quantize(self, t, values):
        return self.all_quant[t].quantize(values)

    @abstractmethod
    def _make_first_quant(self):
        pass

    @abstractmethod
    def _get_all_shapes(self):
        pass

    @abstractmethod
    def _one_step_optimize(self, shape, previous):
        pass

    # Print utilities

    NSEP = 75  # Match default of numpy

    def _print_decorator(self, do):
        def print_and_do(*args):
            self._print_iter_preamble()
            optimum = do(*args)
            self._print_iter_result(optimum)
            return optimum
        return print_and_do

    def _print_iter_preamble(self):
        header = "[t={0:d}] ".format(self._t+1)
        header += "Searching..."
        print("\n" + ("#"*self.NSEP) + "\n")  # print separator
        print(header)

    # @method_decorator(no_exception)
    def _print_iter_result(self, quant):
        print("\nOptimal quantizer (%/y):")
        with printoptions(precision=2, suppress=True):
            print(self.print_formatter(np.ravel(quant.value)))
        print("\n" + ("#"*self.NSEP) + "\n")  # print separator


###################
# 1D Quantization #
###################

class Quantization1D:

    def __init__(
            self,
            value,
            probability,
            transition_probability=None,
            distortion=None,
            bounds=[-np.inf, np.inf]):
        # Checks...
        _check_value(value, bounds)
        _check_proba(probability)
        if transition_probability is not None:
            _check_trans(transition_probability)
        if distortion is not None:
            _check_dist(distortion)
        _check_match_shape(probability, value)
        # Register values...
        self.value = value
        self.shape = self.value.shape
        self.size = self.value.size
        self.ndim = self.value.ndim
        self.probability = probability
        self.transition_probability = transition_probability
        self.distortion = distortion
        # Pre-compute Voronoi tiles
        self.voronoi = np.empty((self.size+1,))
        _voronoi_1d(
                self.value,
                bounds[0],
                bounds[1],
                self.voronoi)

    def quantize(self, values):
        if (
                np.any(values <= self.voronoi[0])
                or np.any(values >= self.voronoi[-1])
        ):
            raise ValueError("Queried value(s) outside quantization space")
        idx = np.digitize(values, self.voronoi)
        return self.value[idx-1]


class AbstractQuantizer1D(ABC):

    """
    Implementing classes must define following properties:
        (1) `bounds` e.g. (0, np.inf) for positive variables
        (2) `_integral`
        (3) `_delta`
    """

    bounds = [-np.inf, np.inf]

    def __init__(self, value, prev_proba):
        if value.ndim != 1:
            raise ValueError('Unexpected value dimension')
        self.size = value.size  # Register size
        if np.any(np.logical_not(np.isfinite(value))):
            raise ValueError('Invalid quantizer [not finite]')
        elif np.any(np.diff(value) < 0.):
            raise ValueError('Invalid quantizer [not increasing]')
        elif (
                np.any(value < self.bounds[0])
                or np.any(value > self.bounds[1])
        ):
            raise ValueError('Invalid quantizer [outside bounds]')
        # Compute Voronoi tiles
        voronoi = np.empty((self.size+1,))
        _voronoi_1d(
                value,
                self.bounds[0],
                self.bounds[1],
                voronoi)
        self.prev_proba = prev_proba

        # Previous dimension and broadcast-ready
        if self.prev_proba.ndim == 1:
            self.einidx = 'i,ij'
            self.value = value[np.newaxis, :]
            self.voronoi = voronoi[np.newaxis, :]
        elif self.prev_proba.ndim == 2:
            self.einidx = 'ij,ijk'
            self.value = value[np.newaxis, np.newaxis, :]
            self.voronoi = voronoi[np.newaxis, np.newaxis, :]
        else:
            raise ValueError('Max of 2 dimensions supported '
                             'for previous quantizers')

    # Results

    @property
    def probability(self):
        """
        Returns a self.shape np.array containing the time-t probability
        """
        return np.einsum(
            self.einidx,
            self.prev_proba,
            self.transition_probability)

    @property
    def transition_probability(self):
        """
        Returns a previous-shape-by-shape np.array
        containing the transition probabilities from time t-1 to time t
        """
        return self._integral[0]

    @property
    def distortion(self):
        """
        Returns the sum of distortion elements
        ```
            (\\mathcal{I}_{2,t}^{(ij)}
            - 2\\mathcal{I}_{1,t}^{(ij)} h_t^{(j)}
            + \\mathcal{I}_{0,t}^{(ij)} (h_t^{(j)})^2
        ```
        weighted by previous probabilities.
        """
        out = np.einsum(
            self.einidx,
            self.prev_proba,
            self._distortion_elements)
        return np.sum(out)

    @property
    def _distortion_elements(self):
        return (
            self._integral[2]
            - 2.*self._integral[1]*self.value
            + self._integral[0]*(self.value**2.)
        )

    @property
    def gradient(self):
        """
        Returns self.size np.array representing the gradient
        of self.distortion
        """
        elements = (
            self._integral[1]
            - self._integral[0] * self.value
        )
        return -2. * np.einsum(
            self.einidx,
            self.prev_proba,
            elements
        )

    @property
    def hessian(self):
        """
        Returns the Hessian of self.distortion
            as a self.size-by-self.size np.array
        """
        elements_under = (
            self._delta[1][..., :-1]
            - self._delta[0][..., :-1]*self.value
        )
        elements_over = (
            self._delta[1][..., 1:]
            - self._delta[0][..., 1:]*self.value
        )

        # Build Hessian
        hess = np.zeros((self.size, self.size))

        # Main-diagonal
        main_elements = elements_over - elements_under - self._integral[0]
        main_diagonal = -2. * np.einsum(
            self.einidx,
            self.prev_proba,
            main_elements
        )
        diag_view(hess, k=0)[:] = main_diagonal

        # Off-diagonal
        if self.size > 1:
            off_elements = (
                self._delta[1][..., 1:-1]
                - self._delta[0][..., 1:-1]*self.value[..., 1:]
            )
            off_diagonal = 2. * np.einsum(
                self.einidx,
                self.prev_proba,
                off_elements)
            diag_view(hess, k=1)[:] = off_diagonal
            diag_view(hess, k=-1)[:] = off_diagonal

        return hess


class ExpandBounds(PymaatException):
    pass

class ShrinkBounds(PymaatException):
    pass

class Optimizer1D:

    MIN_STEP = 1. + 1e-4  # Minimum decrease in distortion accepted

    def __init__(
            self, factory, size, *,
            robust=True,
            scale=None,
            maxiter=10000, niter=1000, niter_success=100,
            verbose=False,
            print_formatter=None
    ):
        self.raw_factory = factory
        self.factory = None
        self.size = size
        self.robust = robust
        self.maxiter = maxiter
        self.scale = scale
        self.niter = niter
        self.niter_success = niter_success
        self.verbose = verbose
        if print_formatter is not None:
            self.print_formatter = print_formatter
        # Internals
        self._x0 = np.zeros((self.size,))
        if self.robust:
            self._optimize = self._bh_optimize
        else:
            self._optimize = self._trust_ncg_optimize

    def print_formatter(self, value): return value

    def optimize(self, search_bounds):
        """
        Raises ExpandBounds when search_bounds needs to be expanded
        Raises ShrinkBounds when search_bounds needs to be shrunk
        """
        self._initialize(search_bounds)
        do_optim = self._optimize
        if self.verbose:
            do_optim = self._print_decorator(do_optim)
        optimum = do_optim()  # Perform optimization
        result = self._get_quantizer_at(optimum.x)  # Get optimal quantizer
        # Verify convergence
        self._raise_expand_if_tight(result)  # Solution is tight?
        self._raise_shrink_if_null_proba(result)  # Solution "wastes" knots?
        # Clear cache
        self.factory.clear()
        return result

    def _initialize(self, sb):
        if self.factory is not None:
            self.factory.clear()  # Flush existing cache if any
        self.factory = _OptimFactory1D(self.raw_factory, sb)
        if self.scale is None:  # Provide a default for scale
            self.scale = self._func(self._x0)
        if self.verbose:  # Initialize print utilities
            self._print_init()

    # To be optimized

    def _func(self, x):
        return self.factory.make(x).distortion

    def _jac(self, x):
        return self.factory.make(x).gradient

    def _hess(self, x):
        return self.factory.make(x).hessian

    # Optimization routines

    def _bh_optimize(self):
        return basinhopping(
            self._func,
            self._x0,
            niter=self.niter,
            disp=False,
            minimizer_kwargs={
                'method': 'trust-ncg',
                'jac': self._jac,
                'hess': self._hess,
                'options': self._format_trust_ncg_options()
            },
            accept_test = self._accept_test,
            callback = self._get_print_callback(),
            niter_success = self.niter_success
        )

    def _trust_ncg_optimize(self):
        return minimize(
            self._func,
            self._x0,
            method='trust-ncg',
            jac=self._jac,
            hess=self._hess,
            options=self._format_trust_ncg_options(),
            callback=self._get_print_callback()
        )

    def _get_quantizer_at(self, x):
        return self.factory.make(x).quantizer

    def _raise_expand_if_tight(self, result):
        v = np.ravel(result.value)
        sb = self.factory.search_bounds
        if self._is_tight(v):
            raise ExpandBounds("Search bounds are tight.")

    def _raise_shrink_if_null_proba(self, result):
        p = result.probability
        if self._has_null_proba(p):
            raise ShrinkBounds("Quantizers have null probability.")

    def _is_tight(self, value):
        sb = self.factory.search_bounds
        return (
                np.isclose(value[0], sb[0])
                or np.isclose(value[-1], sb[1])
                )

    def _has_null_proba(self, proba):
        return np.any(proba<=0.)

    def _format_trust_ncg_options(self):
        return {
            'maxiter': self.maxiter,
            'gtol': self._get_gtol(),
            'disp': False
        }

    def _get_gtol(self):
        return self.scale * 1e-6

    def _accept_test(self, f_new=None, x_new=None, f_old=None, x_old=None):
        q = self._get_quantizer_at(x_new)
        v = np.ravel(q.value)
        # Quantizer with repeated component(s) are sub-optimal
        strictly_increasing = np.all(np.diff(v) > 0.)
        # Quantizer is not diverging (towards search bounds)
        not_escaping = not self._is_tight(v)
        # Quantizer with zero probability are sub-optimal
        no_null_proba = not self._has_null_proba(q.probability)
        # print(no_null_proba)
        # Significant change in objective?
        significant_step = f_old > f_new*self.MIN_STEP
        return (
                strictly_increasing
                and not_escaping
                and no_null_proba
                and significant_step
                )

    # Print Utilities

    def _print_init(self):
        self._n = 0  # current iteration

    def _print_decorator(self, do):
        def print_and_do():
            self._print_preamble()
            optimum = do()
            self._print_afterword(optimum)
            return optimum
        return print_and_do

    def _get_print_callback(self):
        if self.verbose:
            return self._iteration_callback

    def _iteration_callback(self, x, f=None, a=None):
        q = self._get_quantizer_at(x)
        if self._n == 0:
            # Initialize running optimum
            self._distortion = q.distortion
            self._value = q.value
        else:
            if a is None:
                a = self._accept_test(
                f_new=q.distortion,
                f_old=self._distortion,
                x_new=x)
            if a:
                # New minimum found
                self._print_iteration(q)
                # Update running optimum
                self._distortion = q.distortion
                self._value = q.value
            else:
                self._print_void()
        # Increment iterations...
        self._n += 1

    def _print_preamble(self):
        msg = "\n"
        if self.robust:
            msg += "Basin-Hopping "
        else:
            msg += "Trust-NCG "
        sb = self.factory.search_bounds
        msg += "Optimization "
        msg += "(gtol={0:.6e}, ".format(self._get_gtol())
        msg += "search_bounds=[{0:.2f}%, {1:.2f}%]) ".format(
                self.print_formatter(sb[0]),
                self.print_formatter(sb[1]))
        print(msg)

    def _print_afterword(self, optimum):
        quant = self._get_quantizer_at(optimum.x)
        # Retrieve optimizer termination message
        msg = "\n"
        if isinstance(optimum.message, list):
            msg += optimum.message[0]
        else:
            msg += optimum.message
        # Append result
        msg += "\nConverged to "
        msg += "Distortion: {0:.6e}, ".format(quant.distortion)
        msg += "Gradient: {0:.6e}, ".format(infnorm(quant.gradient))
        msg += "Values (%/y):"
        print(msg)
        with printoptions(precision=2, suppress=True):
            print(self.print_formatter(np.ravel(quant.value)))

    def _print_iteration(self, quant):
        msg = "[Iter#{0:d}] ".format(self._n)
        msg += "df={0:+.2f}%, ".format(
            100.*(quant.distortion/self._distortion-1.)
        )
        msg += "dx={0:.2f}%".format(
            100.*np.amax(np.absolute(quant.value/self._value-1.))
        )
        print(msg)

    def _print_void(self):
        print('.', end='')


class _OptimFactory1D:

    def __init__(self, factory, search_bounds):
        self.factory = factory
        self.search_bounds = search_bounds
        self._cache = LRUCache(maxsize=1)

    def clear(self):
        self._cache.clear()

    def make(self, x):
        # Make sure x will not change between calls
        x.flags.writeable = False
        # Cache by object identity, not values e.g. sha1(x).hexdigest()
        key = id(x)
        if key not in self._cache:
            self._cache[key] = _OptimQuantizer1D(
                self.factory, self.search_bounds, x)
        return self._cache.get(key)


class _OptimQuantizer1D:

    OFFSET = 5.

    def __init__(self, factory, search_bounds, x):
        self.factory = factory
        self.search_bounds = search_bounds
        self.x = np.atleast_1d(x)

    @lazy_property
    def quantizer(self):
        return self.factory.make(self.value)

    @lazy_property
    def value(self):
        return self.search_bounds[0]+self._logistic

    @property
    def distortion(self):
        return self.quantizer.distortion

    @property
    def gradient(self):
        return self.quantizer.gradient.dot(self.jacobian)

    @property
    def hessian(self):
        # Contribution from objective hessian
        out = self.jacobian.T.dot(self.quantizer.hessian).dot(self.jacobian)
        # Contribution from first derivative of logistic function
        df = icumsum(self.quantizer.gradient*self._dlogistic)
        diag_view(out)[:] += df*self._kappa0
        # Contribution from second derivative of logistic function
        ddf = icumsum(self.quantizer.gradient*self._ddlogistic)
        ud = np.triu(  # Upper diagonal elements
                self._kappa1[np.newaxis, :]
                *self._kappa1[:, np.newaxis]
                * ddf[np.newaxis,:])
        out += ud + ud.T - np.diag(ud.diagonal())  # "Symmetrize"
        return out

    @lazy_property
    def jacobian(self):
        return (
            self._kappa1[np.newaxis, :]
            * self._dlogistic[:, np.newaxis]
            * np.tri(self.x.size, self.x.size)
        )

    @lazy_property
    def theta(self):
        x0 = self.x[0]-self.OFFSET
        grid = x0 + np.cumsum(self._scaled_exp_x)
        return np.insert(grid, 0, x0)

    # Internals...

    @lazy_property
    def _logistic(self):
        return self._span*logistic(self.theta)

    @lazy_property
    def _dlogistic(self):
        return self._span*dlogistic(self.theta)

    @lazy_property
    def _ddlogistic(self):
        return self._span*ddlogistic(self.theta)

    @lazy_property
    def _scaled_exp_x(self):
        return 2.*self.OFFSET*np.exp(self.x[1:])/(self.x.size-1.)

    @lazy_property
    def _kappa0(self):
        return np.insert(self._scaled_exp_x, 0, 0.)

    @lazy_property
    def _kappa1(self):
        return np.insert(self._scaled_exp_x, 0, 1.)

    @lazy_property
    def _span(self):
        return self.search_bounds[1] - self.search_bounds[0]

###########
# Voronoi #
###########


@workon_axis
def voronoi_1d(quantizer, lb=-np.inf, ub=np.inf):
    if quantizer.size == 0:
        raise ValueError
    shape = list(quantizer.shape)
    shape[0] += 1
    voronoi = np.empty(shape)
    _voronoi_1d(quantizer, lb, ub, voronoi)
    return voronoi

def _voronoi_1d(q, lb, ub, out):
    out[0] = lb
    out[1:-1] = q[:-1] + 0.5*np.diff(q, n=1, axis=0)
    out[-1] = ub

@workon_axis
def inv_voronoi_1d(voronoi, *, first_quantizer=None, with_bounds=True):
    if voronoi.ndim > 2:
        raise ValueError("Does not support dimension greater than 2")
    if np.any(np.diff(voronoi, axis=0) <= 0.):
        raise ValueError("Not strictly increasing Voronoi")
    if with_bounds:
        voronoi = voronoi[1:-1]  # Crop bounds, otherwise do nothing

    s = voronoi.shape
    broadcastable = (s[0],)+(1,)*(len(s)-1)
    # First, build (-1,+1) alternating vector
    alt_vector = np.empty((s[0],))
    alt_vector[::2] = 1.
    alt_vector[1::2] = -1.

    # Preliminary checks
    b = _get_first_quantizer_bounds(s, broadcastable, voronoi, alt_vector)
    if np.any(b[0] >= b[1]):
        raise ValueError("Has no inverse")
    if first_quantizer is None:
        if np.all(np.isfinite(b)):
            first_quantizer = 0.5 * (b[0] + b[1])
        else:
            raise ValueError("Could not infer first quantizer")
    elif np.any(first_quantizer >= b[1]) or np.any(first_quantizer <= b[0]):
        raise ValueError("Invalid first quantizer")

    # Initialize output
    inverse = np.empty((s[0]+1,)+s[1:])
    inverse[0] = first_quantizer  # May broadcast here
    # Solve using numpy matrix multiplication
    alt_matrix = np.empty((s[0], s[0]))
    for i in range(s[0]):
        diag_view(alt_matrix, k=-i)[:] = alt_vector[i]
    if voronoi.size > 0:
        inverse[1:] = 2.*np.dot(np.tril(alt_matrix), voronoi)
    # Correct for first element of quantizer
    inverse[1:] -= (np.reshape(alt_vector, broadcastable)
                    * inverse[np.newaxis, 0])
    assert np.all(np.diff(inverse, axis=0) > 0.)
    return inverse


def _get_first_quantizer_bounds(s, broadcastable, voronoi, alt_vector):
    lb = []
    ub = []
    term = np.cumsum(
        np.reshape(alt_vector, broadcastable)*voronoi,
        axis=0
    )

    for (i, v) in enumerate(voronoi):
        if i == 0:
            ub.append(v)
        else:
            if i % 2 == 0:
                ub.append(v-2.*term[i-1])
            else:
                lb.append(-v+2.*term[i-1])

    if len(lb) == 0:
        lb = -np.inf
    else:
        lb = np.max(np.array(lb), axis=0)

    if len(ub) == 0:
        ub = np.inf
    else:
        ub = np.min(np.array(ub), axis=0)

    return lb, ub

##############################
# Testing Helper Function(s) #
##############################


def _assert_valid_delta_at(factory, value, order, rtol=1e-6, atol=0.):
    quantizer = factory.make(value)
    it = np.nditer(factory.prev_proba, flags=['multi_index'])
    while not it.finished:
        def to_derivate(value):
            q = factory.make(value)
            return q._integral[order][it.multi_index]
        jac = np.zeros((quantizer.size, quantizer.size))
        deltas = np.ravel(quantizer._delta[order][it.multi_index])
        diag_view(jac, k=0)[:] = np.diff(deltas)
        if quantizer.size > 1:
            diag_view(jac, k=1)[:] = deltas[1:-1]
            diag_view(jac, k=-1)[:] = -deltas[1:-1]
        # Assertion
        pt.assert_jacobian_at(jac,
                              np.ravel(quantizer.value),
                              function=to_derivate,
                              rtol=rtol,
                              atol=atol)
        # Go to next previous value
        it.iternext()
