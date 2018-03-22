from abc import ABC, abstractmethod
from cachetools import LRUCache
from hashlib import sha1
import warnings

import numpy as np
from scipy.linalg import norm as infnorm
from scipy.optimize import minimize, brute, basinhopping
from scipy.stats import norm, uniform

from pymaat.util import lazy_property
from pymaat.nputil import workon_axis, diag_view, printoptions

class AbstractCore(ABC):

    print_formatter = lambda value: value

    def __init__(self, model):
        self.model = model

    def optimize(self, *, verbose=False, fast=False):
        # Initialize recursion
        current_quantizer = self._make_first_quant()

        # Perform recursion
        self.all_quant = []
        for (t, s) in enumerate(self._get_all_shapes()):
            if verbose:
                header = "[t={0:d}] ".format(t)
                header += " Searching for optimal quantizer"
                sep = "\n" + ("#"*len(header)) + "\n"
                print(sep)
                print(header)
            # Gather results from previous iteration...
            self.all_quant.append(current_quantizer)
            # Advance to next time step (computations done here...)
            current_quantizer = self._one_step_optimize(
                s, current_quantizer, verbose=verbose, fast=fast)
            if verbose:
                print("\n[Optimal quantizer (%/y)]")
                with printoptions(precision=2, suppress=True):
                    print(self.print_formatter(current_quantizer.value))
                print(sep)

        # Handle last quantizer
        self.all_quant.append(current_quantizer)

        return self  # for convenience

    def quantize(self, t, values):
        return self.all_quant[t].quantize(values)

    @abstractmethod
    def _make_first_quant(self):
        pass

    @abstractmethod
    def _get_all_shapes(self):
        pass

    @abstractmethod
    def _one_step_optimize(self, shape, previous, *,
            verbose=False, fast=False):
        pass

class AbstractQuantization(ABC):

    @abstractmethod
    def quantize(self, value):
        pass

class AbstractFactory1D(ABC):
    CACHE_SIZE = 30

    def __init__(
            self, model, size, unc_flag, *,
            voronoi_bounds, prev_proba, min_value, scale_value,
            singularities=np.array([])):
        self.model = model
        self.ndim = 1
        self.size = size
        self.shape = (size,)
        self.unc_flag = unc_flag  # Unconstrained flag
        # Voronoi bounds
        self.voronoi_bounds = voronoi_bounds
        # For change of variable...
        self.min_value = min_value
        self.scale_value = scale_value
        # For random searches...
        self.singularities = np.unique(singularities)

        # Previous quantizer properties
        self.prev_proba = prev_proba
        self.prev_ndim = prev_proba.ndim
        self.prev_size = prev_proba.size
        self.prev_shape = prev_proba.shape

        # Setting up internal state..
        self.cache = LRUCache(maxsize=self.CACHE_SIZE)
        self.distortion_scale = (self.scale_value**2.
                /(self.prev_size*self.size**2.))

    def clear(self):
        self.cache.clear()

    def make(self, x):
        x.flags.writeable = False
        key = sha1(x).hexdigest()
        if key not in self.cache:
            if self.unc_flag:  # Unconstrained
                quant = _Unconstrained1D(self, self.target, x)
            else:  # Straight-up quantizer
                quant = self.target(self, x)
            self.cache[key] = quant
        return self.cache.get(key)

    def change_of_variable(self, x):
        scaled_expx = self.scale_value*np.exp(x)/(self.size+1.)
        return self.min_value+np.cumsum(scaled_expx)

    def inv_change_of_variable(self, value):
        padded_values = np.insert(value, 0, self.min_value)
        expx = (self.size+1.)*np.diff(padded_values)/self.scale_value
        return np.log(expx)


class _Unconstrained1D:

    def __init__(self, parent, target, x):
        self.parent = parent
        self.shape = parent.shape
        self.size = parent.size
        self.ndim = parent.ndim
        x = np.asarray(x)
        self.x = x
        self.quantizer = target(parent, parent.change_of_variable(x))
        # Internals...
        self._scaled_expx = (
                parent.scale_value * np.exp(self.x)
                / (self.size+1.))

    @property
    def distortion(self):
        dist = self.quantizer.distortion

        if not np.isfinite(dist):
            raise AtSingularity()

        return dist

    @property
    def gradient(self):
        """
        Returns the transformed gradient for the distortion function, ie
        dDistortion/ dx = dDistortion/dGamma * dGamma/dx
        """
        grad = self.quantizer.gradient.dot(self._jacobian)

        if not np.all(np.isfinite(grad)):
            raise AtSingularity()

        return grad

    @property
    def hessian(self):
        """
        Returns the transformed Hessian for the distortion function
        """
        def get_first_term():
            jac = self._jacobian
            hess = self.quantizer.hessian
            return jac.T.dot(hess).dot(jac)

        def get_second_term():
            grad = self.quantizer.gradient
            inv_cum_grad = np.flipud(np.cumsum(np.flipud(grad)))
            return np.diag(self._scaled_expx*inv_cum_grad)

        hess = get_first_term() + get_second_term()

        if not np.all(np.isfinite(hess)):
            raise AtSingularity()

        return hess

    @property
    def _jacobian(self):
        """
        Returns the self.size-by-self.size np.array containing the jacobian
        of the change of variable, ie
        ```
        \\nabla_{ij} \\Gamma_{h,t}
        ```
        """
        mask = np.tril(np.ones((self.size, self.size)))
        return self._scaled_expx[np.newaxis, :]*mask

class AbstractQuantizer1D(ABC):

    def __init__(self, parent, value):
        if not isinstance(parent, AbstractFactory1D):
            raise ValueError("Unexpected parent")
        else:
            self.parent = parent

        value = np.asarray(value)
        # Verifications...
        if (
                np.any(~np.isfinite(value))
                or np.any(np.diff(value) <= 0.)
                or np.any(value<=self.parent.voronoi_bounds[0])
                or np.any(value>=self.parent.voronoi_bounds[1])
            ):
            raise AtSingularity()

        # Set value and compute Voronoi tiles
        self.value = value
        self.voronoi = voronoi_1d(
                self.value,
                lb=self.parent.voronoi_bounds[0],
                ub=self.parent.voronoi_bounds[1])

        # Previous dimension and broadcast
        if  self.parent.prev_ndim == 1:
            self.einidx = 'i,ij'
            self.broad_value = self.value[np.newaxis,:]
            self.broad_voronoi = self.voronoi[np.newaxis,:]
        elif self.parent.prev_ndim == 2:
            self.einidx ='ij,ijk'
            self.broad_value = self.value[np.newaxis,np.newaxis,:]
            self.broad_voronoi = self.voronoi[np.newaxis,np.newaxis,:]
        else:
            raise ValueError('Unexpected previous dimension')

    # Results

    @property
    def probability(self):
        """
        Returns a self.shape np.array containing the time-t probability
        """
        return np.einsum(
                self.einidx,
                self.parent.prev_proba,
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
                self.parent.prev_proba,
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
                self.parent.prev_proba,
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
        hess = np.zeros((self.parent.size, self.parent.size))

        # Main-diagonal
        main_elements = elements_over - elements_under - self._integral[0]
        main_diagonal = -2. * np.einsum(
                self.einidx,
                self.parent.prev_proba,
                main_elements
                )
        diag_view(hess, k=0)[:] = main_diagonal

        # Off-diagonal
        if self.parent.size > 1:
            off_elements = (
                self._delta[1][...,1:-1]
                - self._delta[0][...,1:-1]*self.broad_value[...,1:]
                )
            off_diagonal = 2. * np.einsum(
                    self.einidx,
                    self.parent.prev_proba,
                    off_elements)
            diag_view(hess, k=1)[:] = off_diagonal
            diag_view(hess, k=-1)[:] = off_diagonal

        return hess

#############
# Optimizer #
#############

class AtSingularity(Exception):
    pass

class Optimizer:

    def __init__(self, factory):
        self.factory = factory
        if not self.factory.unc_flag:
            raise ValueError(
                    "Optimizer only supports unconstrained factories.")
        self.func = lambda x: self.factory.make(x).distortion
        self.jac = lambda x: self.factory.make(x).gradient
        self.hess = lambda x: self.factory.make(x).hessian

    def optimize(
            self, *,
            gtol=1e-4, maxiter=10000, niter=1000, niter_success=50,
            seed=None, x0=None, verbose=False,
    ):
        """
        Default. Robust when gradient is discontinuous.
        """
        if niter_success is None:
            self._niter_success = niter+2
        else:
            self._niter_success = niter_success
        options = {
            'maxiter': maxiter,
            'disp': False,
            'gtol': self.factory.distortion_scale*gtol
            }
        minimizer_kwargs = {
            'method': 'trust-ncg',
            'jac': self.jac,
            'hess': self.hess,
            'options': options
        }
        cb = lambda x,f,accept: self._bh_callback(x, verbose)
        def optimize_from(x0_): return basinhopping(
            self.func, x0_, niter=niter,
            take_step=_TakeRandomStep(self.factory, seed=seed),
            disp=False,
            minimizer_kwargs=minimizer_kwargs,
            callback=cb,
        )

        if verbose:
            msg = "Optimize with gtol={0:.6e}".format(options['gtol'])
            msg += ", maxiter={0:d}".format(options['maxiter'])
            msg += ", niter={0:d}".format(niter)
            msg += ", niter_success={0:d}".format(self._niter_success)
            print(msg)

        out = self._try_optimization(optimize_from, x0)

        return out

    def quick_optimize(self, *,
            gtol=1e-4, maxiter=10000, x0=None, verbose=False):
        """
        Might stall on a local minimum when gradient is discontinuous.
        """
        options = {
            'maxiter': maxiter,
            'gtol': self.factory.distortion_scale*gtol,
            'disp': False
            }

        def optimize_from(x0_): return minimize(
            self.func, x0_, method='trust-ncg',
            jac=self.jac, hess=self.hess,
            options=options,
        )

        if verbose:
            msg = "Quick optimize with gtol={0:.6e}".format(options['gtol'])
            msg += ", maxiter={0:d}".format(options['maxiter'])
            print(msg)

        out = self._try_optimization(optimize_from, x0)

        if verbose:
            gradnorm = infnorm(out.gradient)
            msg = "\tDistortion: {0:.6e}, ".format(out.distortion)
            msg += "Gradient: {0:.6e}".format(gradnorm)
            print(msg)

        return out


    def brute_optimize(self, search_width=3.):
        """
        Solve by brute force.
        """
        if self.factory.size>3:
            raise ValueError(
                    "Cannot use brute force optimizer for large quantizers.")
        ranges = ((-search_width, search_width),)*self.factory.size
        sol = brute(self.func, ranges)
        return self.factory.make(sol).quantizer

    def _try_optimization(self, optimize_from, x0=None, _try=0):
        if x0 is None:
            x0 = np.zeros(self.factory.shape)
        try:
            opt = optimize_from(x0)
        except AtSingularity:
            x0 = np.random.normal(size=self.factory.shape)
            _try += 1
            if _try<10:
                return self._try_optimization(optimize_from, x0, _try)
            else:
                raise RuntimeError("Could not recover from bad Hessian")
        return self.factory.make(opt.x).quantizer

    def _bh_callback(self, x, verbose):
        q = self.factory.make(x).quantizer

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
                    msg += "dx={0:.2f}%".format(dvalue)
                    msg += "] "
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


class _TakeRandomStep():

    def __init__(self, factory, seed=None):
        self.random_state = np.random.RandomState(seed)
        self.factory = factory

    def __call__(self, x):
        # Search bounds
        if self.factory.singularities.size > 1:
            span = np.mean(np.diff(self.factory.singularities))
            lb = self.factory.singularities[0] - span
            ub = self.factory.singularities[-1] + span
        else:
            lb = self.factory.min_value
            ub = lb + 2.*self.factory.scale_value
        lb = max(lb, self.factory.min_value)
        # Randomize
        value = uniform.rvs(
            lb, ub-lb, size=self.factory.shape,
            random_state=self.random_state)
        value.sort()
        new_x = self.factory.inv_change_of_variable(value)
        # Bound x to avoid numerical instability
        return np.clip(new_x, -10., 20.)

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
