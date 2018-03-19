from abc import ABC, abstractmethod
from cachetools import LRUCache
from hashlib import sha1
import warnings

import numpy as np
from scipy.linalg import norm as infnorm
from scipy.optimize import minimize, brute, basinhopping
from scipy.stats import norm, truncnorm, uniform

from pymaat.util import lazy_property
from pymaat.nputil import workon_axis, diag_view

class AbstractQuantization(ABC):

    def __init__(self, model, print_formatter):
        self.model = model
        self.print_formatter = print_formatter

    def optimize(self, *, verbose=False):
        # Initialize recursion
        current_quantizer = self._make_first_quantizer()

        # Perform recursion
        self.all_quantizers = []
        for (t, s) in enumerate(self._get_all_shapes()):
            if verbose:
                header = "[t={0:d}] ".format(t)
                header += " Searching for optimal quantizer"
                sep = "\n" + ("#"*len(header)) + "\n"
                print(sep)
                print(header, end='')
            # Gather results from previous iteration...
            self.all_quantizers.append(current_quantizer)
            # Advance to next time step (computations done here...)
            current_quantizer = self._one_step_optimize(
                s, current_quantizer, verbose=verbose)
            if verbose:
                # Display results
                print("\n[Optimal distortion]\n{0:.6e}".format(
                    quant.distortion))
                print("\n[Optimal quantizer (%/y)]")
                with printoptions(precision=2, suppress=True):
                    print(self.print_formatter(quant.value))
                print(sep)

        # Handle last quantizer
        self.all_quantizers.append(current_quantizer)

        return self  # for convenience

    @abstractmethod
    def quantize(self, t, values):
        pass

    @abstractmethod
    def _make_first_quantizer(self):
        pass

    @abstractmethod
    def _get_all_shapes(self):
        pass

    @abstractmethod
    def _one_step_optimize(self, shape, previous, *, verbose=False):
        pass

class AbstractFactory(ABC):
    cache_size = 30

    def __init__(self, model, size, previous, target):
        if np.any(previous.probability <= 0.):
            raise ValueError(
                "Previous probabilities must be strictly positive")
        self.model = model
        self.ndim = 1
        self.size = size
        self.shape = (size,)
        self.previous = previous
        self.target = target

        # Setting up internal state..
        self.cache = LRUCache(maxsize=self.cache_size)

    @abstractmethod
    def get_distortion_scale(self):
        pass

    def make(self, x):
        x.flags.writeable = False
        key = sha1(x).hexdigest()
        if key not in self.cache:
            quant = self.target(self, x)
            self.cache[key] = quant
        return self.cache.get(key)

class AbstractQuantizer(ABC):

    def __init__(self, parent, value, *, lb=-np.inf, ub=np.inf):
        if not isinstance(parent, AbstractFactory):
            raise ValueError("Unexpected parent")
        else:
            self.parent = parent
        self.lb = lb
        self.ub = ub
        # Copy for convenience from parent
        self.model = parent.model
        self.shape = parent.shape
        self.size = parent.size
        self.previous = parent.previous
        # Process quantizer value
        value = np.asarray(value)

        if value.shape != self.shape:
            err_msg = 'Unexpected quantizer shape\n'
        elif not np.all(np.isfinite(value)):
            err_msg = 'Invalid quantizer (NaN or inf)\n'
        elif np.any(np.diff(value) <= 0.):
            err_msg = ('Unexpected quantizer value(s): '
                   'must be strictly increasing\n')
        elif np.any(value<=self.lb) or np.any(value>=self.ub):
            err_msg = ('Unexpected quantizer value(s): '
                   'outside specified bounds\n')

        if 'err_msg' in locals():
            raise ValueError(err_msg)
        else:
            self.value = value
            self.voronoi = voronoi_1d(self.value, lb=self.lb, ub=self.ub)

        # Previous dimension
        if  self.previous.ndim == 1:
            self.einidx = 'i,ij'
            self.broad_value = self.value[np.newaxis,:]
            self.broad_voronoi = self.voronoi[np.newaxis,:]
            self.broad_previous_value = self.previous.value[:, np.newaxis]
        elif self.previous.ndim == 2:
            self.einidx ='ij,ijk'
            self.broad_value = self.value[np.newaxis,np.newaxis,:]
            self.broad_voronoi = self.voronoi[np.newaxis,np.newaxis,:]
            self.broad_previous_value = self.previous.value[:, :, np.newaxis]
        else:
            raise ValueError('Unexpected previous dimension')

    # broad dictionnary with broad voronoi, value ,previous value ,

    @lazy_property
    def previous_probability(self):
        return self.previous.probability

    # Results

    @property
    def probability(self):
        """
        Returns a self.shape np.array containing the time-t probability
        """
        return np.einsum(
                self.einidx,
                self.previous_probability,
                self.transition_probability)

    @property
    def transition_probability(self):
        """
        Returns a self.previous.shape-by-self.shape np.array
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
                self.previous_probability,
                self._distortion_elements)
        return np.sum(out)

    @property
    def _distortion_elements(self):
        return (
                self._integral[2]
                - 2.*self._integral[1]*self.broad_value
                + self._integral[0]*(self.broad_value**2.)
                )

    @property
    def gradient(self):
        """
        Returns self.size np.array representing the gradient
        of self.distortion
        """
        elements = (
                self._integral[1]
                - self._integral[0] * self.broad_value
                )
        return -2. * np.einsum(
                self.einidx,
                self.previous_probability,
                elements
                )

    @property
    def hessian(self):
        """
        Returns the Hessian of self.distortion
            as a self.size-by-self.size np.array
        """
        elements_under = (
                self._delta[1][...,:-1]
                - self._delta[0][...,:-1]*self.broad_value
                )
        elements_over = (
                self._delta[1][...,1:]
                - self._delta[0][...,1:]*self.broad_value
                )

        # Build Hessian
        hess = np.zeros((self.size, self.size))

        # Main-diagonal
        main_elements = elements_over - elements_under - self._integral[0]
        main_diagonal = -2. * np.einsum(
                self.einidx,
                self.previous_probability,
                main_elements
                )
        diag_view(hess, k=0)[:] = main_diagonal

        # Off-diagonal
        if self.shape[0] > 1:
            off_elements = (
                self._delta[1][...,1:-1]
                - self._delta[0][...,1:-1]*self.broad_value[...,1:]
                )
            off_diagonal = 2. * np.einsum(
                    self.einidx,
                    self.previous_probability,
                    off_elements)
            diag_view(hess, k=1)[:] = off_diagonal
            diag_view(hess, k=-1)[:] = off_diagonal

        return hess

#############
# Optimizer #
#############

class AtSingularity(Exception):
    def __init__(self, idx):
        self.idx = np.unique(idx)

class Optimizer:

    method = 'trust-ncg'
    max_iter = 10000

    def __init__(self, factory, *, gtol=1e-4):
        self.gtol = gtol
        self.factory = factory
        self.func = lambda x: self.factory.make(x).distortion
        self.jac = lambda x: self.factory.make(x).gradient
        self.hess = lambda x: self.factory.make(x).hessian
        self.options = {
            'maxiter': self.max_iter,
            'disp': False,
            'gtol': self.factory.get_distortion_scale()*self.gtol
            }

    def optimize(
            self, *,
            niter=1000, niter_success=50, interval=5,
            seed=None, x0=None, verbose=True,
    ):
        """
        Default. Robust when gradient is discontinuous.
        """
        if niter_success is None:
            self._niter_success = niter+2
        else:
            self._niter_success = niter_success
        if verbose:
            msg = "with gtol={0:.6e}".format(self.options['gtol'])
            msg += ", niter={0:d}".format(niter)
            msg += ", niter_success={0:d}".format(self._niter_success)
            msg += " and interval={0:d}".format(interval)
            print(msg)
        minimizer_kwargs = {
            'method': self.method,
            'jac': self.jac,
            'hess': self.hess,
            'options': self.options
        }
        cb = lambda x,f,accept: self._bh_callback(x, verbose)

        def optimize_from(x0_): return basinhopping(
            self.func, x0_, niter=niter,
            T=self.factory.get_distortion_scale(),
            interval=interval,
            take_step=_OptimizerTakeStep(self.factory, seed=seed),
            disp=False,
            minimizer_kwargs=minimizer_kwargs,
            callback=cb,
        )
        return self._try_optimization(optimize_from, x0)

    def quick_optimize(self, *, x0=None, verbose=True):
        """
        Might stall on a local minimum when gradient is discontinuous.
        """
        if verbose:
            print("with gtol={0:.6e}".format(self.options['gtol']))

        def optimize_from(x0_): return minimize(
            self.func, x0_, method=self.method,
            jac=self.jac, hess=self.hess,
            options=self.options,
        )
        return self._try_optimization(optimize_from, x0)

    def brute_optimize(self, search_width=3.):
        """
        Solve by brute force.
        """
        if self.factory.size>3:
            raise ValueError(
                    "Cannot use brute force optimizer for large quantizers.")
        ranges = ((-search_width, search_width),)*self.factory.size
        sol = brute(self.func, ranges)
        return self.factory.make(sol)

    def _try_optimization(self, optimize_from, x0=None):
        if x0 is None:
            x0 = np.zeros(self.factory.shape)
        try:
            opt = optimize_from(x0)
        except AtSingularity as exc:
            # Optimizer met a singularity for indices idx
            # Move by arbitrary displacement
            x0[exc.idx] += 1e-2
            return self._try_optimization(optimize_from, x0)
        return self.factory.make(opt.x)

    def _bh_callback(self, x, verbose):
        q = self.factory.make(x)

        # Initialization
        if not hasattr(self, '_distortion'):
            self._distortion = q.distortion
            self._value = q.value
            self._niter = 0

        # If improve by at least 0.01%...
        if self._niter == 0 or q.distortion*(1.+1e-4) < self._distortion:
            if verbose:
                dobjective = 100.*(q.distortion/self._distortion-1.)
                dvalue = 100.*np.amax(np.absolute(q.value/self._value-1.))
                gradnorm = infnorm(q.gradient)
                msg = "[Iter#{0:d}] ".format(self._niter)
                if self._niter > 0:
                    msg += "New minimum ["
                    msg += "df={0:+.2f}%,".format(dobjective)
                    msg += "dx={1:.2f}%".format(dvalue)
                    msg += "]"
                msg += "Distortion: {0:.6e}, ".format(q.distortion)
                msg += "Gradient: {0:.6e}".format(gradnorm)
                print(msg)
            # Update
            self._niter_since_last = 0
            self._distortion = q.distortion
            self._value = q.value
        else:
            self._niter_since_last += 1

        self._niter += 1

        stop_flag = self._niter_since_last >= self._niter_success
        if verbose and stop_flag:
            msg = "Early termination: could not improve for "
            msg += "{0:d} consecutive tries".format(
                self._niter_since_last)
            print(msg)

        return stop_flag


class _OptimizerTakeStep():
# TODO major refactoring needed here
    def __init__(
            self, factory, *, seed=None, persistent=True):
        self.random_state = np.random.RandomState(seed)
        self.factory = factory
        self.stepsize = 1.
        self.persistent = persistent
        if self.factory.previous.size > 1:
            self.scale = np.mean(np.diff(self.factory.singularities))
        else:
            self.scale = self.factory.min_variance/self.factory.size

    def __call__(self, x):
        # Search bounds
        lb = self.factory.min_variance
        ub = self.factory.singularities[-1] + self.scale
        if self.persistent:
            value = self._persistent(self.stepsize, lb, ub, x)
        else:
            value = self._non_persistent(lb, ub)
        new_x = self.factory.target._inv_change_of_variable(self.factory, value)
        # Bound x to avoid numerical instability
        return np.clip(new_x, -10., 20.)

    def _persistent(self, step, lb, ub, x):
        _, value = self.factory.target._change_of_variable(self.factory, x)
        scale_ = 0.5*step*self.scale
        for i in range(value.size):
            # Compute truncated normal parameters
            loc_ = value[i]
            a_ = (lb-loc_)/scale_
            b_ = (ub-loc_)/scale_
            # Simulate
            tmp = truncnorm.rvs(
                a_, b_, loc=loc_, scale=scale_,
                random_state=self.random_state)
            # Check simulation consistency
            if not (tmp > lb and tmp < ub):
                # Fail-safe if numerical instability detected (eg. NaNs)
                # More likely to happen for large step sizes
                tmp = uniform.rvs(lb, ub-lb,
                                  size=value[i:].shape,
                                  random_state=self.random_state)
                tmp.sort()
                value[i:] = tmp
                msg = "Take-step failsafe: numerical instability "
                msg += " [stepsize={0:.2f}]".format(step)
                warnings.warn(msg, UserWarning)
                break
            else:
                # Commit new value and update lower bound
                lb = value[i] = tmp
        return value

    def _non_persistent(self, lb, ub):
        value = uniform.rvs(
            lb, ub-lb, size=self.parent.shape,
            random_state=self.random_state)
        value.sort()
        return value


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
    voronoi[0] = lb
    voronoi[1:-1] = quantizer[:-1] + 0.5*np.diff(quantizer, n=1, axis=0)
    voronoi[-1] = ub
    return voronoi


@workon_axis
def inv_voronoi_1d(voronoi, *, first_quantizer=None, with_bounds=True):
    if voronoi.ndim > 2:
        raise ValueError("Does not support dimension greater than 2")
    if np.any(np.diff(voronoi, axis=0)<=0.):
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
    if np.any(b[0]>=b[1]):
        raise ValueError("Has no inverse")
    if first_quantizer is None:
        if np.all(np.isfinite(b)):
            first_quantizer = 0.5 * (b[0] + b[1])
        else:
            raise ValueError("Could not infer first quantizer")
    elif np.any(first_quantizer>=b[1]) or np.any(first_quantizer<=b[0]):
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
    assert np.all(np.diff(inverse, axis=0)>0.)
    return inverse


def _get_first_quantizer_bounds(s, broadcastable, voronoi, alt_vector):
    lb = []; ub = []
    term = np.cumsum(
            np.reshape(alt_vector, broadcastable)*voronoi,
            axis=0
            )

    for (i,v) in enumerate(voronoi):
        if i==0:
            ub.append(v)
        else:
            if i%2 == 0:
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
