import warnings
from abc import ABC, abstractmethod
from cachetools import LRUCache
from math import pi, ceil
from hashlib import sha1
from types import GeneratorType
from functools import wraps, partial

import numpy as np
from scipy.linalg import norm as linalg_norm
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize
from scipy.spatial import Voronoi

import pymaat.quantutil_c as qutilc
from pymaat.util import PymaatException
from pymaat.util import lazy_property
from pymaat.nputil import workon_axis, diag_view, printoptions
from pymaat.nputil import icumsum
from pymaat.mathutil import logistic, dlogistic, ddlogistic
from pymaat.mathutil import ilogistic
from pymaat.nputil import ravel
from pymaat.util import method_decorator

import pymaat.testing as pt

import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar

CMAP = matplotlib.cm.get_cmap('gnuplot2').reversed()


class UnobservedState(PymaatException):
    pass


def chunked_generator(chunksize=100000):
    def decorator(f):
        @wraps(f)
        def simulator(nsim, *args):
            nchunk = np.int(np.ceil(nsim/chunksize))
            chunks = np.empty((nchunk,), np.uint64)
            chunks[:] = chunksize
            chunks[-1] = (1-nchunk) * chunksize + nsim
            for n in chunks:
                yield f(n, *args)
        return simulator
    return decorator


class AbstractCore:

    SEP = "\n" + ("#"*75) + "\n"

    def __init__(self, *,
                 shape=100,
                 nper=None,
                 verbose=False,
                 chunksize=100000,  # May be tuned for simulation performance
                 parallel=False,
                 **kwargs
                 ):
        """
        Inputs:
            * `first_quantization` is an instance of AbstractQuantization
            * `simulate(nsim)`
                return (first_sim, sim) where sim is a list of length
                [self.nper],
                and all elements contain np.arrays of shape [nsim, ndim]
        """
        self.shapes = self._get_shapes(shape, nper)
        self.nper = len(self.shapes)
        # First quantization...
        self.first_quantization = self.get_first_quantization()
        # ...bounds and dimension are constant...
        self.bounds = self.first_quantization.bounds
        self.ndim = self.first_quantization.ndim
        # Simulations
        self.chunksize = chunksize
        self.simulator = chunked_generator(self.chunksize)(self.simulate)
        # Process weights
        self.weights = self.get_weights()
        # Checks
        self._check_simulator()
        self._check_weights()
        # Get and set quantizations...
        self.verbose = verbose
        self.parallel = parallel
        self.quantizations = self.optimize()  # Computations here

    def _get_shapes(self, s, nper):
        s = np.atleast_2d(s)
        if nper is not None:
            shapes = np.broadcast_to(s, (nper, s.shape[1]))
        else:
            shapes = s
        return list(shapes)

    def get_quantized_at(self, t, values):
        return self.quantizations[t].get_quantized(values)

    def estimate_distortion(self, nsim):
        cumul_dist = np.zeros((self.nper,))
        for _, all_sim in self.simulator(nsim):
            for t, (sim, q) in enumerate(
                    zip(all_sim, self.quantizations)):
                cumul_dist[t] += np.sum(q.get_local_distortion(sim))
        return cumul_dist/nsim

    def plot_at(self, t, ax, **kwargs):
        raise NotImplementedError

    def simulate(self, nsim):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def get_first_quantization(self):
        raise NotImplementedError

    def _check_simulator(self):
        def _check_sim(sim):
            if sim.ndim != 2:
                raise ValueError(
                    "Simulation calls must return matrices "
                    "following (N,K)-numpy convention "
                    "i.e. with rows representing K-dim data points"
                )
            if sim.shape[0] != 1:
                raise ValueError(
                    "Unexpected number of simulations "
                    "from simulation call"
                )
            if sim.shape[1] != self.ndim:
                raise ValueError(
                    "Unexpected number of dimensions "
                    "from simulation call"
                )

        for first_sim, sim in self.simulator(1):
            if len(sim) != self.nper:
                raise ValueError(
                    "Unexpected number of periods "
                    "from simulation call"
                )
            _check_sim(first_sim)
            for s in sim:
                _check_sim(s)

    def _check_weights(self):
        if self.weights.shape != (self.nper, self.ndim):
            raise ValueError("Unexpected weights shape")


class AbstractMarkovianCore(AbstractCore):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def one_step_simulate(self, previous, nsim):
        '''
        `one_step_simulate(previous, nsim)`
            returns len-2 list [previous, current]
            containing np.arrays of shape [nsim, ndim]
        '''
        raise NotImplementedError

    def optimize(self):
        q = self.first_quantization
        result = []
        for n, (s, w) in enumerate(zip(self.shapes, self.weights)):
            # Forward time recursion
            if self.verbose:
                print(self.SEP + "[time={0:d}] ".format(n+1))
            q = self.one_step_optimize(s, q, w)
            result.append(q)  # Save result
            if self.verbose:
                print(self.SEP)
        return result

    @abstractmethod
    def one_step_optimize(self, shape, previous, weight):
        pass


#################
# Voronoi Cores #
#################

class _VoronoiCoreMixin:

    # Voronoi Core Quantization Utilities

    def __init__(
            self, *,
            last_step=1e-2,  # Used to determine number of simulations
            nlloyd=10,
            fullsplit=True,
            **kwargs
    ):
        # Check shapes
        # TODO: support different shapes for each timestep
        if last_step >= 1.:
            raise ValueError("Last step must be lower than 1")
        else:
            self.last_step = last_step
        self.nlloyd = nlloyd
        self.fullsplit = fullsplit
        super().__init__(**kwargs)

    def get_learning_parameters(self, size, distortion):
        # Learning parameters from:
        #   Pages G. (2003) Optimal quadratic
        #   quantization for numericals: the Gaussian case
        a = 4.*size**(1./self.ndim)
        b = pi**2*size**(-2./self.ndim)
        first_step = min(1., np.sqrt(distortion))
        return first_step, a, b

    def get_nsim(self, size):
        # Number of simulations from:
        #   Pages G. (2003) Optimal quadratic
        #   quantization for numericals: the Gaussian case
        _MAX = 1e9
        nsim = np.ceil(4.*size**1.5/(self.last_step*np.pi**2.))
        if nsim > _MAX:
            warnings.warn("Capping number of simulations to 1e9.")
            nsim = _MAX
        return np.int(nsim)


class AbstractMarginalVoronoiCore(_VoronoiCoreMixin, AbstractCore):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def optimize(self):
        if np.any(np.unique(self.shapes) != self.shapes[0][0]):
            raise ValueError("Unexpected shapes for Voronoi Quantization")
        # TODO: verbose?
        # Step 0: Get some random starting values
        #   (Remember that all shapes are the same)
        size = self.shapes[0][0]
        _, starting = self.simulate(size)
        # Step I: Do competitive learning
        if self.fullsplit:
            quantizer = [
                np.zeros((0, self.ndim))
                for n in range(self.nper)
            ]
            distortion = None
            for to_add in zip(*starting):
                quantizer = [
                    np.append(q, x[np.newaxis, :], axis=0)
                    for (q, x) in zip(quantizer, to_add)
                ]
                distortion = self._inplace_clvq(quantizer, distortion)
        else:
            quantizer = [s.copy() for s in starting]
            self._inplace_clvq(quantizer)
        # Step II: Do Llyod Method I
        for _ in range(self.nlloyd):
            self._inplace_lloyd1(quantizer)

        # Format result
        result = []
        for n in range(self.nper):
            result.append(
                VoronoiQuantization(
                    quantizer[n], self.bounds, self.weights[n, :])
            )
            # Link to previous
            if n == 0:
                result[-1].set_previous(self.first_quantization)
            else:
                result[-1].set_previous(result[-2])
        self._estimate_and_set_transition_probability(result)
        return result

    def _inplace_clvq(self, quantizer, distortion=None):
        if distortion is None:
            distortion = np.ones((self.nper,))
        size = quantizer[0].shape[0]
        N = np.zeros((self.nper,), np.uint64)
        cumul_distortion = np.zeros((self.nper,))
        nsim = self.get_nsim(size)
        for _, sim in self.simulator(nsim):
            def func(n):
                N[n], cumul_distortion[n] = qutilc.inplace_clvq(
                    sim[n],
                    quantizer[n],
                    self.weights[n, :],
                    *self.get_learning_parameters(size, distortion[n]),
                    N[n],
                    cumul_distortion[n]
                )
            if self.parallel:
                p = Pool()
                p.map(func, range(self.nper))
                p.close()
            else:
                for n in range(self.nper):
                    func(n)
        return cumul_distortion/N

    def _inplace_lloyd1(self, quantizer):
        size = quantizer[0].shape[0]
        count = [np.zeros((size,), dtype=np.uint64) for _ in range(self.nper)]
        cumul = [np.zeros((size, self.ndim)) for _ in range(self.nper)]
        # For consistency, use same number of simulations as clvq!
        nsim = self.get_nsim(size)
        for _, sim in self.simulator(nsim):
            def func(n):
                qutilc.lloyd1(
                    sim[n],
                    quantizer[n],
                    self.weights[n, :],
                    count[n],
                    cumul[n]
                )
            if self.parallel:
                p = Pool()
                p.map(func, range(self.nper))
                p.close()
            else:
                for n in range(self.nper):
                    func(n)
        # In-place modification of quantizer
        for n in range(self.nper):
            quantizer[n] += cumul[n]
            quantizer[n] /= (count[n]+1)[:, np.newaxis]

    def _estimate_and_set_transition_probability(
            self, quantizations, *, max_ntry=10):
        # TODO: use sparse matrices here
        nsim = self.get_nsim(self.shapes[0][0])

        def do(cumul=None, N=None, ntry=0):
            if cumul is None:
                cumul = [
                    q._init_joint_probability()
                    for q in quantizations
                ]
            if N is None:
                N = np.zeros((self.nper,), np.uint64)
            for first_sim, sim in self.simulator(nsim):
                def func(n):
                    if n > 0:
                        N[n] = quantizations[n]._cumul_joint_probability(
                            sim[n-1], sim[n], N[n], cumul[n])
                    else:
                        N[0] = quantizations[0]._cumul_joint_probability(
                            first_sim, sim[0], N[0], cumul[0])
                if self.parallel:
                    p = Pool()
                    p.map(func, range(self.nper))
                    p.close()
                else:
                    for n in range(self.nper):
                        func(n)
            for n in range(self.nper):
                try:
                    quantizations[n]._set_joint_probability(N[n], cumul[n])
                except UnobservedState:
                    if ntry >= max_ntry:
                        raise UnobservedState
                    else:  # Fail-safe: try again (i.e. more simulations)
                        ntry += 1
                        warnings.warn(
                            "Unobserved state(s) "
                            + "[Try #{0:d} out of {1:d}]".format(
                                ntry, max_ntry)
                        )
                        N, cumul = do(cumul, N, ntry)
            return N, cumul
        do()


class AbstractMarkovianVoronoiCore(
        _VoronoiCoreMixin,
        AbstractMarkovianCore
):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def one_step_optimize(self, shape, previous, weight):
        # TODO: verbose?
        simulate = partial(self.one_step_simulate, previous)
        simulator = chunked_generator(self.chunksize)(simulate)
        # Step 0: Get some random starting values (all shapes are the same)
        _, starting = simulate(self.shapes[0])
        # Step I: Do competitive learning
        if self.fullsplit:
            quantizer = np.zeros((0, self.ndim))
            distortion = 1.
            for x0 in starting:
                quantizer = np.append(quantizer, x0[np.newaxis, :], axis=0)
                distortion = self._inplace_clvq(
                    simulator, quantizer, weight, distortion)
        else:
            quantizer = starting.copy()
            self._inplace_clvq(simulator, quantizer, weight)
        # Step II: Do Llyod Method I
        for _ in range(self.nlloyd):
            self._inplace_lloyd1(simulator, quantizer, weight)
        # Step III: Format result
        result = VoronoiQuantization(quantizer, self.bounds, weight)
        result.set_previous(previous)
        result.estimate_and_set_transition_probability(
            simulate,  # TODO: pass simulator directly
            self.get_nsim(shape),
            chunksize=self.chunksize
        )
        return result

    def _inplace_clvq(self, simulator, quantizer, weight, distortion=1.):
        size = quantizer.shape[0]
        N = 0
        cumul_distortion = 0.
        nsim = self.get_nsim(size)
        for _, sim in simulator(nsim):
            N, cumul_distortion = qutilc.inplace_clvq(
                sim,
                quantizer,
                weight,
                *self.get_learning_parameters(size, distortion),
                N,
                cumul_distortion
            )
        return cumul_distortion/N

    def _inplace_lloyd1(self, simulator, quantizer, weight):
        size = quantizer.shape[0]
        count = np.zeros((size,), dtype=np.uint64)
        cumul = np.zeros((size, self.ndim))
        # For consistency, use same number of simulations as clvq!
        for _, sim in simulator(self.get_nsim(size)):
            qutilc.lloyd1(sim, quantizer, weight, count, cumul)
        quantizer += cumul
        quantizer /= (count+1)[:, np.newaxis]


#################
# Quantizations #
#################

class AbstractQuantization:

    def _digitize(self, values):
        raise NotImplementedError

    def _plot2d_tiles(ax, lim, colormap, colornorm, formatter1, formatter2):
        raise NotImplementedError

    def __init__(self, quantizer, bounds=None, weight=None):
        self.ndim = quantizer.shape[-1]
        self.shape = quantizer.shape[:-1]
        self.size = np.prod(self.shape)
        self._set_bounds(bounds)  # Must be set before quantizer
        self._set_quantizer(quantizer)
        self._set_weight(weight)
        self.previous = None
        self.probability = None
        self.transition_probability = None

    def estimate_distortion(self, simulator, nsim):
        simulator = chunked_generator(chunksize=100000)(simulator)
        cumul_dist = 0.
        for simul in simulator(nsim):
            cumul_dist += np.sum(self.get_local_distortion(simul))
        return cumul_dist/nsim

    def simulate(self, nsim):
        # Very important to have priory set probabilities
        # Otherwise calls to np.random.choice fail silently
        if self.probability is None:
            raise ValueError(
                "Probabilities must be set in order to simulate")
        states = np.random.choice(
            self.size,
            nsim,
            p=np.ravel(self.probability)
        )
        return self._ravel()[states, :]

    #########
    # State #
    #########

    def set_probability(self, probability, *, strict=True):
        probability = np.array(probability, ndmin=len(self.shape))
        eps = 1e-12
        if probability.shape != self.shape:
            raise ValueError(
                "Invalid probability: "
                + "expecting shape "
                + str(self.shape)
                + ", was "
                + str(probability.shape)
            )
        elif not np.all(np.isfinite(probability)):
            raise ValueError(
                'Invalid probability: NaN(s) or inf(s) detected')
        elif np.abs(np.sum(probability)-1.) > eps:
            raise ValueError(
                'Invalid probability: must sum to one')
        elif strict and np.any(probability <= 0.):
            raise ValueError(
                'Invalid probability: must be strictly positive')
        elif not strict and np.any(probability < 0.):
            raise ValueError(
                'Invalid probability: must be positive')
        else:
            self.probability = probability

    def set_transition_probability(self, trans):
        if self.previous is None:
            raise ValueError("Previous quantization must be set prior"
                             "to transition probabilities")
        current_ax = tuple(range(-len(self.shape), 0))
        eps = 1e-12
        if trans.shape != self.previous.shape + self.shape:
            raise ValueError(
                "Invalid transition probability: "
                + "expecting shape "
                + str(self.previous.shape + self.shape)
                + ", was "
                + str(trans.shape)
            )
        elif not np.all(np.isfinite(trans)):
            raise ValueError(
                "Invalid transition probability: "
                + "NaNs or infinite"
            )
        elif np.any(np.abs(np.sum(trans, axis=current_ax)-1.) > eps):
            raise ValueError('Invalid transition probability:'
                             ' must sum to one')
        elif np.any(trans < 0.):
            raise ValueError(
                "Invalid transition probability: "
                + " must be positive (or zero)"
            )
        else:
            self.transition_probability = trans

    def set_previous(self, previous):
        if not isinstance(previous, AbstractQuantization):
            raise ValueError("Must be a quantization")
        self.previous = previous

    ##########################
    # Probability Estimation #
    ##########################

    def estimate_and_set_transition_probability(
            self, one_step_simulate, nsim,
            *,
            max_ntry=10, chunksize=100000
    ):
        '''
        one_step_simulate(nsim) must return (prev_simul, simul) tuples
            containing arrays of shape [nsim, ndim]
        '''
        if self.previous is None:
            raise ValueError("Previous quantization must be set prior"
                             "to estimating transition probabilities")
        simulator = chunked_generator(chunksize)(one_step_simulate)

        def do(N=0, cumul=None, ntry=0):
            if cumul is None:
                cumul = self._init_joint_probability()
            for prev_simul, simul in simulator(nsim):
                N = self._cumul_joint_probability(
                    prev_simul, simul, N, cumul)
            try:
                self._set_joint_probability(N, cumul)
            except UnobservedState:
                if ntry >= max_ntry:
                    raise UnobservedState
                else:  # Fail-safe: try again (i.e. more simulations)
                    ntry += 1
                    warnings.warn(
                        "Unobserved state(s) "
                        + "[Try #{0:d} out of {1:d}]".format(
                            ntry, max_ntry)
                    )
                    N, cumul = do(N, cumul, ntry)
            return N, cumul
        do()

    def _init_joint_probability(self):
        return np.zeros(self.previous.shape + self.shape, np.uint64)

    def _cumul_joint_probability(self, prev_simul, simul, N, cumul):
        assert cumul.shape == self.previous.shape + self.shape
        prev_idx = self.previous._digitize(prev_simul)
        idx = self._digitize(simul)
        np.add.at(cumul, prev_idx + idx, 1)
        return N + prev_simul.shape[0]

    def _set_joint_probability(self, N, cumul):
        # TODO use sparse matrices for transition probabilities
        prev_ax = tuple(range(len(self.previous.shape)))
        current_ax = tuple(range(-len(self.shape), 0))
        joint = cumul/N
        # Compute marginal probabilities...
        # (1) ...of {current} by summing over previous states
        proba = np.sum(joint, axis=prev_ax)
        # (2) ...of {previous} by summing over current states
        prev_proba = np.sum(joint, axis=current_ax)
        # Check for unobserved states (marginal)
        if np.any(proba <= 0) or np.any(prev_proba <= 0):
            raise UnobservedState
        # Compute transition probabilities of {current \vert previous}
        broadcast_shape = self.previous.shape + (1,)*len(self.shape)
        transition = joint/np.reshape(prev_proba, broadcast_shape)
        # Set results
        self.set_probability(proba)
        self.set_transition_probability(transition)

    def estimate_and_set_probability(
            self, simulate, nsim,
            *,
            strict=True, max_ntry=10, chunksize=100000
    ):
        '''
        simulate(nsim) must return a np.array of shape [nsim, ndim]
        '''
        simulator = chunked_generator(chunksize)(simulate)

        def do(N=0, cumul=None, ntry=0):
            if cumul is None:
                cumul = self._init_probability()
            for simul in simulator(nsim):
                N = self._cumul_probability(simul, N, cumul)
                try:
                    self._set_probability(N, cumul, strict)
                except UnobservedState:
                    if ntry >= max_ntry:
                        raise UnobservedState
                    else:  # Fail-safe: try again (i.e. with more simulations)
                        ntry += 1
                        warnings.warn(
                            "Unobserved state(s) failsafe "
                            + "[Try #{0:d} out of {1:d}]".format(
                                ntry, max_ntry)
                        )
                        N, cumul = do(N, cumul, ntry)
            return N, cumul
        do()

    def _init_probability(self):
        return np.zeros(self.shape, np.uint64)

    def _cumul_probability(self, simul, N, cumul):
        idx = self._digitize(simul)
        np.add.at(cumul_proba, idx, 1)
        return N + simul.shape[0]

    def _set_probability(self, N, cumul, strict):
        proba = cumul/N
        if strict and np.any(proba <= 0):
            raise UnobservedState
        self.set_probability(proba, strict=strict)

    ###########
    # Getters #
    ###########

    def get_quantized(self, values):
        self._check_query(values)
        idx = self._digitize(values)
        result = self._quantize(idx)
        self._mask(values, result)
        return result

    def get_local_distortion(self, values):
        self._check_query(values)
        idx = self._digitize(values)
        result = self._quantize(idx)-values
        result *= self.weight
        result **= 2.
        self._mask(values, result)
        return result

    ########
    # Plot #
    ########

    def plot1d(self, ax, lim, plim):
        # TODO
        raise NotImplementedError

    def plot2d(self, ax, colorbar_ax=None, *,
               lim=None,  # First and second dimension range
               plim=None,  # Probability range
               colormap=CMAP,
               colorbar_orientation='vertical',
               quantizer_size=0,
               formatter1=None,
               formatter2=None
               ):

        if formatter1 is None:
            def formatter1(x): return x

        if formatter2 is None:
            def formatter2(x): return x

        if self.probability is None:
            raise ValueError("Probability must be set for plotting")

        if self.ndim != 2:
            raise ValueError("Quantization is not 2-D")

        def _get_lim(x, m):
            ptp = np.ptp(x)
            lim = [np.amin(x)-ptp*m, np.amax(x)+ptp*m]
            return np.array(lim)

        quantizer = self._ravel()
        if lim is None:
            lim1 = _get_lim(quantizer[:, 0], 0.1)
            lim2 = _get_lim(quantizer[:, 1], 0.1)
            lim = np.column_stack((lim1, lim2))
        if plim is None:
            plim = _get_lim(self.probability, 0.)
        plim = np.array(plim)*100.  # Probability displayed in percentage
        colornorm = matplotlib.colors.Normalize(*plim)
        self._plot2d_tiles(
            ax, lim, colormap, colornorm, formatter1, formatter2)

        # # Plot Quantizers
        if quantizer_size > 0:
            ax.scatter(formatter1(quantizer[:, 0]),
                       formatter2(quantizer[:, 1]),
                       s=quantizer_size,
                       c='k',
                       marker=".")

        # Adjust limits
        ax.set_xlim(formatter1(lim[:, 0]))
        ax.set_ylim(formatter2(lim[:, 1]))

        if colorbar_ax is not None:
            matplotlib.colorbar.ColorbarBase(
                colorbar_ax,
                cmap=colormap,
                norm=colornorm,
                orientation=colorbar_orientation
            )

    #############
    # Internals #
    #############

    def _quantize(self, idx):
        return self.quantizer[idx]

    def _mask(self, values, out, invalid=np.nan):
        # In-place masking of out with invalid for values outside bounds
        to_mask = np.logical_not(
            np.logical_and(
                values > self.bounds[0:1, :],
                values < self.bounds[1:2, :]
            )
        )
        out[np.any(to_mask, axis=1), ...] = invalid

    def _ravel(self):
        return self.quantizer.reshape((self.size, self.ndim))

    def _set_bounds(self, bounds):
        if bounds is None:
            self.bounds = np.empty((2, self.ndim))
            self.bounds[0] = -np.inf
            self.bounds[1] = np.inf
        elif bounds.shape != (2, self.ndim):
            raise ValueError("Invalid bounds: "
                             + "expecting shape " + str((2, self.ndim))
                             + ", was " + str(bounds.shape)
                             )
        elif np.any(bounds[0] >= bounds[1]):
            raise ValueError(
                "Invalid bounds: must be strictly increasing")
        else:
            self.bounds = bounds

    def _set_quantizer(self, quantizer):
        if quantizer.shape != self.shape + (self.ndim,):
            raise ValueError(
                "Invalid quantizer: "
                + "expecting shape "
                + str(self.shape+(self.ndim,))
                + ", was "
                + str(quantizer.shape)
            )
        self._check_all_dim(quantizer)
        self.quantizer = quantizer

    def _check_all_dim(self, x):
        for n in range(self.ndim):
            self._check_at_dim(x[..., n], n)

    def _check_at_dim(self, x, dim):
        if not np.all(np.isfinite(x)):
            raise ValueError(
                "Invalid value: "
                + "NaN(s) or inf(s) detected"
            )
        elif (
                np.any(x <= self.bounds[0, dim])
                or np.any(x >= self.bounds[1, dim])
        ):
            raise ValueError(
                "Invalid value: "
                + "must be strictly within specified bounds "
                + str(self.bounds[:, dim])
            )

    def _set_weight(self, weight):
        if weight is None:
            self.weight = np.ones((1, self.ndim))
        elif weight.size != self.ndim:
            raise ValueError("Invalid weight: "
                             + "expecting size " + str(self.ndim)
                             + ", was " + str(weight.size)
                             )
        elif not np.all(np.isfinite(weight)):
            raise ValueError(
                "Invalid weight: "
                + "NaN(s) or inf(s) detected"
            )
        elif not np.all(weight > 0.):
            raise ValueError("Invalid weight: "
                             + " must be strictly positive")
        else:
            self.weight = np.copy(weight)
        self.weight.shape = (1, self.ndim)

    def _check_query(self, values):
        if values.ndim != 2 or values.shape[1] != self.ndim:
            raise ValueError(
                "Expecting matrix with "
                + str(self.ndim)
                + " columns,"
                + " instead had shape "
                + str(values.shape)
            )

# def stochastic_optimization(
#         size,
#         simulator,
#         *,
#         last_step=1e-2,
#         nlloyd=0,
#         fullsplit=False,
#         nclvq=1,
#         weight=None,
#         bounds=None
# ):

#     if fullsplit:
#         nclvq = size

#     # Checks
#     starting = simulator(size)[1]
#     if starting.ndim != 2:
#         raise ValueError(
#             "simulator must return matrices "
#             "following (N,K)-numpy convention "
#             "i.e. with rows representing K-dim data points"
#         )
#     quantizer = starting.copy()

#     # Process inputs
#     ndim = quantizer.shape[1]

#     if weight is None:
#         weight = np.ones((ndim,))
#     else:
#         weight = np.ravel(weight)

#     def clvq(simulator, quantizer, distortion):
#         size = quantizer.shape[0]
#         g0, a, b, nsim = get_learning_parameters(
#                 size, ndim, last_step, distortion)
#         N = 0
#         cumul_dist = 0.
#         for sim in simulator(nsim):
#             N, d = qutilc.inplace_clvq(sim, quantizer, weight, g0, a, b, N)
#             cumul_dist += d

#         return cumul_dist/N

#     def lloyd1(simulator, quantizer, distortion):
#         size = quantizer.shape[0]
#         count = np.zeros((size,), dtype=np.uint64)
#         cumul = np.zeros((size, ndim), dtype=np.float_)
#         # For consistency, use same number of simulations as clvq
#         *_, nsim = get_learning_parameters(size, ndim, last_step, distortion)
#         for sim in simulator(nsim):
#             qutilc.lloyd1(sim, quantizer, weight, count, cumul)
#         quantizer += cumul
#         quantizer /= (count+1)[:,np.newaxis]

#     @chunked_generator(chunksize=100000)
#     def _simulator(nsim):
#         return simulator(nsim)[1]

#     # Optimization
#     distortion = 1.

#     # Step I: Do competitive learning
#     if nclvq > 0:
#         quantizer = np.zeros((0, ndim))
#         for x0 in np.array_split(starting, nclvq, axis=0):
#             if x0.size > 0:
#                 quantizer = np.append(quantizer, x0, axis=0)
#                 distortion = clvq(_simulator, quantizer, distortion)

#     # Step II: Do Llyod Method I
#     for _ in range(nlloyd):
#         lloyd1(_simulator, quantizer, distortion)

#     return VoronoiQuantization(quantizer, bounds, weight)


class VoronoiQuantization(AbstractQuantization):

    def _digitize(self, values):
        idx = self.tree.nearest(values, np.ravel(self.weight))
        return (idx,)

    def _plot2d_tiles(self, ax, lim, colormap, cnorm, formatter1, formatter2):
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        precision = 100
        t = np.linspace(0., 1., precision, endpoint=False)
        t = t[:, np.newaxis]
        # Hack to avoid infinite ridges...
        width = 10.
        span1 = width*(lim[1, 0]-lim[0, 0])
        span2 = width*(lim[1, 1]-lim[0, 1])
        far = np.array([
            [lim[0, 0]-span1, lim[0, 1]-span2],  # bottom-left
            [lim[0, 0]-span1, lim[1, 1]+span2],  # top-left
            [lim[1, 0]+span1, lim[1, 1]+span2],  # top-right
            [lim[1, 0]+span1, lim[0, 1]-span2],  # bottom-right
        ])
        boundary = np.empty((0, 2))
        for s, e in zip(far, np.roll(far, -1, axis=0)):
            s = s[np.newaxis, :]
            e = e[np.newaxis, :]
            boundary = np.append(boundary, s + (e-s)*t, axis=0)
        x = np.append(self.quantizer, boundary, axis=0)
        vor = Voronoi(x*self.weight)  # Interface with QHull
        patches = []
        for region in vor.point_region[:self.size]:
            idx1 = np.array(vor.regions[region])
            idx2 = np.roll(idx1, -1)
            if np.any(idx1 < 0):
                raise RuntimeError
            v = np.empty((0, 2))
            for i1, i2 in zip(idx1, idx2):
                start = vor.vertices[i1]
                start = start[np.newaxis, :]
                end = vor.vertices[i2]
                end = end[np.newaxis, :]
                pts = start + (end-start)*t
                v = np.append(v, pts, axis=0)
            v /= self.weight
            v[:, 0] = formatter1(v[:, 0])
            v[:, 1] = formatter2(v[:, 1])
            patches.append(Polygon(v))
        collect = PatchCollection(patches)
        collect.set_color(colormap(cnorm(self.probability*100.)))
        ax.add_collection(collect)

    def __init__(self, value, bounds=None, weight=None):
        if value.ndim != 2:
            raise ValueError(
                "Value must be a matrix "
                "following (N,K)-numpy convention "
                "i.e. with rows representing K-dim data points"
            )
        super().__init__(value, bounds, weight)
        self.tree = qutilc.KDTree(self.quantizer)


# Specialized Quantizations

class Quantization1D(AbstractQuantization):

    def _digitize(self, values):
        idx = np.digitize(np.ravel(values), self._voronoi)
        np.clip(idx, 1, self.size, out=idx)
        idx -= 1
        return (idx,)

    def __init__(self, value, bounds=None, weight=None):
        value = np.ravel(value)
        if bounds is not None:
            bounds = np.ravel(bounds)
            bounds.shape = (2, 1)
        super().__init__(value[:, np.newaxis], bounds, weight)
        self._voronoi = voronoi_1d(
            value,
            lb=self.bounds[0, 0],
            ub=self.bounds[1, 0]
        )


def _format_2d_bounds(bounds1, bounds2):
    bounds = np.empty((2, 2))
    bounds[0] = -np.inf
    bounds[1] = np.inf
    if bounds1 is not None:
        bounds[:, 0] = bounds1
    if bounds2 is not None:
        bounds[:, 1] = bounds2
    return bounds


class GridQuantization2D(AbstractQuantization):

    def _digitize(self, values):
        return (
            qutilc.digitize_1d(self._voronoi1, values[:, 0]),
            qutilc.digitize_1d(self._voronoi2, values[:, 1])
        )

    def _plot2d_tiles(self, ax, lim, colormap, colornorm, formatter1, formatter2):
        v1 = np.clip(self._voronoi1, *lim[:, 0])
        v2 = np.clip(self._voronoi2, *lim[:, 1])
        v1 = formatter1(v1)
        v2 = formatter2(v2)
        v1, v2 = np.meshgrid(v1, v2)
        p = self.probability.T*100
        ax.pcolor(v1, v2, p, cmap=colormap, norm=colornorm)

    def __init__(self, value1, value2,
                 bounds1=None, bounds2=None, weight=None):
        self.value1 = np.ravel(value1)
        self.value2 = np.ravel(value2)
        value = self._get_value()
        bounds = _format_2d_bounds(bounds1, bounds2)
        super().__init__(value, bounds, weight)
        self._voronoi1 = voronoi_1d(
            self.value1,
            lb=self.bounds[0, 0],
            ub=self.bounds[1, 0]
        )
        self._voronoi2 = voronoi_1d(
            self.value2,
            lb=self.bounds[0, 1],
            ub=self.bounds[1, 1]
        )

    def _get_value(self):
        s = (self.value1.size, self.value2.size)
        v1 = np.broadcast_to(self.value1[:, np.newaxis], s)
        v2 = np.broadcast_to(self.value2[np.newaxis, :], s)
        return np.stack((v1, v2), axis=2)


class ConditionalQuantization2D(AbstractQuantization):

    def _digitize(self, values):
        return qutilc.conditional_digitize_2d(
            self._voronoi1, self._voronoi2, values)

    def _plot2d_tiles(self, ax, lim, colormap, cnorm, formatter1, formatter2):
        v1 = np.clip(self._voronoi1, *lim[:, 0])
        v2 = np.clip(self._voronoi2, *lim[:, 1])
        v1 = formatter1(v1)
        v2 = formatter2(v2)
        p = self.probability*100.
        for idx in range(self.shape[0]):
            x = v1[idx:idx+2]
            y = v2[idx]
            x, y = np.meshgrid(x, y)
            z = p[idx][:, np.newaxis]
            ax.pcolor(x, y, z, cmap=colormap, norm=cnorm)

    def __init__(self, value1, value2,
                 bounds1=None, bounds2=None, weight=None):
        # Normalize inputs...
        value1 = np.ravel(value1)
        value2 = np.atleast_2d(value2)
        value = self._get_value(value1, value2)
        bounds = _format_2d_bounds(bounds1, bounds2)
        # Instantiate super...
        super().__init__(value, bounds, weight)
        # Set up conditional quantization...
        self._voronoi1 = voronoi_1d(
            value1,
            lb=self.bounds[0, 0],
            ub=self.bounds[1, 0]
        )

        self._voronoi2 = voronoi_1d(
            value2,
            lb=self.bounds[0, 1],
            ub=self.bounds[1, 1],
            axis=1
        )

    def _get_value(self, value1, value2):
        value1 = np.broadcast_to(value1[:, np.newaxis], value2.shape)
        return np.stack((value1, value2), axis=2)


################################
# One-Dimensional Quantization #
################################


class AbstractQuantizer1D(ABC):

    """
    Implementing classes must define following properties:
        (1) `bounds` e.g. (0, np.inf) for positive variables
        (2) `_integral`
        (3) `_delta`
    """

    bounds = [-np.inf, np.inf]

    @method_decorator(ravel)
    def __init__(self, value, prev_proba, *, norm=1.):
        # Save sizes
        self.size = value.size
        self.prev_size = prev_proba.size
        self.norm = norm
        # Compute voronoi tiles
        voronoi = np.empty((self.size+1,))
        _voronoi_1d(
            value,
            self.bounds[0],
            self.bounds[1],
            voronoi)
        # Save broadcast-ready arrays
        self.prev_proba = prev_proba
        self.value = value[np.newaxis, :]
        self.voronoi = voronoi[np.newaxis, :]

    # Results

    @property
    def probability(self):
        """
        Returns a self.shape np.array containing the time-t probability
        """
        return np.einsum(
            'i,ij',
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
            'i,ij',
            self.prev_proba,
            self._distortion_elements)
        return np.sum(out)/self.norm

    @property
    def conditional_expectation(self):
        out = np.einsum(
            'i,ij',
            self.prev_proba,
            self._integral[1]
        )/self.probability
        return out

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
            'i,ij',
            self.prev_proba,
            elements
        )/self.norm

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
            'i,ij',
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
                'i,ij',
                self.prev_proba,
                off_elements)
            diag_view(hess, k=1)[:] = off_diagonal
            diag_view(hess, k=-1)[:] = off_diagonal

        return hess/self.norm

    @property
    def _distortion_elements(self):
        return (
            self._integral[2]
            - 2.*self._integral[1]*self.value
            + self._integral[0]*(self.value**2.)
        )


class Optimizer1D:

    def __init__(
            self, factory, size, *,
            maxiter=10000,
            weight=1.,
            verbose=False,
            formatter=None
    ):
        self._factory = factory
        self.size = size
        self.maxiter = maxiter
        self.weight = 1
        self.verbose = verbose
        if formatter is not None:
            self.formatter = formatter

    def formatter(self, value): return value

    NSEP = 75  # Match default of numpy

    def optimize(self, search_bounds, starting=None):
        search_bounds = np.asarray(search_bounds)

        if self.verbose:
            print('*'*self.NSEP)  # print separator

        if starting is None:
            # Try to find suitable starting value...
            starting, search_bounds = self._get_starting(search_bounds)

        result = starting
        if self._is_optimal(result):  # Starting is already a solution!
            if self.verbose:
                self._print_important("Starting value already optimal")
        else:
            opt = _Optimizer1D(
                self._factory,
                result.value,
                search_bounds,
                self.maxiter,
                self.weight,
                self.verbose,
                self.formatter
            )
            # Constrained (global) optimization
            result = opt.do_bfgs()

            # Solution is tight?
            while opt.is_tight:
                if self.verbose:
                    self._print_important(
                        "Solution is tight. "
                        "Expand search bounds and re-try...")
                result = opt.do_bfgs()

            # Refine
            result = opt.do_newton()

        # Sanity Checks
        msg = ''
        if not self._has_null_gradient(result, 1e-12):
            grad = linalg_norm(result.gradient, ord=np.inf)
            msg += "\n(*) Gradient is non-null [{}]".format(grad)
        if not self._is_spd(result):
            if self.size > 1:
                ev = eigvalsh_tridiagonal(
                    diag_view(result.hessian),
                    diag_view(result.hessian, 1),
                    'i',
                    [0, 0]
                )
            else:
                ev = result.hessian
            ev = np.ravel(ev)[0]
            msg += "\n(*) Hessian is not SPD [{}]".format(ev)
        if not self._is_stat(result):
            msg += "\n(*) Quantizer is non-stationary"
        if len(msg) > 0:
            msg += "\n"
            # if self.verbose:
            self._print_important(msg)
            warnings.warn(msg)

        return result

    def _get_starting(self, search_bounds):
        """
        Find some reasonable starting points using fast Lloyd I iterations
        """
        starting = self._do_lloyd(np.mean(search_bounds), rtol=1e-2)
        stepsize = ceil(self.size/10)

        while starting.value.size != self.size:
            search_bounds = _Optimizer1D.expand_search_bounds(
                starting.value,
                search_bounds
            )
            proba = starting.probability
            val = starting.value[0, :]
            vor = starting.voronoi[0, :]
            vor[0] = search_bounds[0]
            vor[-1] = search_bounds[-1]

            # Select `s` most probable
            s = min(stepsize, proba.size)
            most = np.argpartition(proba, -s)[-s:]

            new = []
            for m in most:
                lb = vor[m]
                ub = vor[m+1]
                v0 = val[m]
                if v0-lb < ub-v0:  # Left split
                    new.append(0.5*(v0+lb))
                else:  # Right split
                    new.append(0.5*(v0+ub))

            new = np.append(val, new)
            if new.size >= self.size:
                new = new[:self.size]
            new.sort()

            starting = self._do_lloyd(new, rtol=1e-2)

        return starting, search_bounds

    def _do_lloyd(self, starting, rtol=1e-4, niter=10):
        old_ = np.ravel(starting)
        for _ in range(niter):
            quantizer = self._factory.make(old_)
            if np.all(quantizer.probability > 1e-16):
                next_ = quantizer.conditional_expectation
                if np.all(np.isclose(old_, next_, rtol=rtol)):
                    old_ = next_
                    break
                old_ = next_
            else:
                return None
        return self._factory.make(old_)

    # Sanity checks...

    def _is_optimal(self, result):
        return (
            self._is_spd(result)
            and self._has_null_gradient(result)
            and self._is_stat(result)
        )

    def _is_spd(self, result):
        if self.size > 1:
            ev = eigvalsh_tridiagonal(
                diag_view(result.hessian),
                diag_view(result.hessian, 1),
                'i',
                [0, 0]
            )
        else:
            ev = result.hessian
        return np.ravel(ev)[0] > 0.

    def _has_null_gradient(self, result, tol=None):
        if tol is None:
            tol = 2.*np.finfo(np.float).eps
        grad = linalg_norm(result.gradient, ord=np.inf)
        return grad < tol

    def _is_stat(self, quantizer, rtol=1e-6):
        return np.all(
            np.isclose(
                np.ravel(quantizer.value),
                np.ravel(quantizer.conditional_expectation),
                rtol=rtol,
            )
        )

    # Print utilities
    def _print_important(self, msg):
        print("### " + msg + " ###")


class _Optimizer1D:

    OFFSET = 50

    @staticmethod
    def expand_search_bounds(value, sb, expand_factor=0.5):
        value = np.ravel(value)
        sb = np.copy(sb)
        if value.size == 1.:
            ptp = np.diff(sb)
        else:
            ptp = np.ptp(value)
        if _Optimizer1D.is_tight(value[0], sb):
            sb[0] = value[0] - expand_factor*ptp
        if _Optimizer1D.is_tight(value[-1], sb):
            sb[1] = value[-1] + expand_factor*ptp
        return sb

    @staticmethod
    def is_tight(value, sb):
        assert np.isscalar(value)
        span = np.diff(sb)
        x = (value-sb[0])/span
        if x <= 0. or x >= 1.:
            return True
        theta = ilogistic(x)/_Optimizer1D.OFFSET
        if theta <= -0.5 or theta >= 0.5:
            return True
        theta = theta+0.5 if theta < 0 else 0.5-theta
        return np.log(theta) < -10.

    def __init__(
            self, factory, starting_value, search_bounds,
            maxiter, weight, verbose, formatter):
        self.factory = factory
        self.starting_value = np.ravel(starting_value)
        self.size = self.starting_value.size
        self.search_bounds = search_bounds
        self.maxiter = maxiter
        self.weight = weight
        self.verbose = verbose
        self.verbose_value = (
            self.verbose and self.starting_value.size < 100)
        self.formatter = formatter

    class Logger:

        def __init__(
                self, make, x0, search_bounds, verbose, formatter):
            self.make = make
            self.x = x0
            self.size = self.x.size
            self.search_bounds = search_bounds
            self.verbose = verbose
            self.verbose_value = self.verbose and self.size < 100
            self.formatter = formatter
            # Register starting quantizer
            self.starting = self.make(x0)
            self.current = self.starting
            self.niter = 0

        def __call__(self, x):
            self.niter += 1
            q = self.make(x)
            if q.distortion < self.current.distortion:
                # New running optimum
                self.x = x
                self.current = q

        def register_optimum(
                self, x, success=True, is_tight=False, msg=""):
            self(x)
            self.success = success
            self.is_tight = is_tight
            self.msg = msg

        # Print Utilities

        def _print_preamble(self, title):
            if self.verbose:
                msg = "\n" + title
                msg += "\n" + "-"*len(msg)
                msg += "\nSearch space: [{0:.2f},{1:.2f}]".format(
                    *self.formatter(self.search_bounds))
                msg += ". Starting from..."
                print(msg)
                self._print_current_state()

        def _print_current_state(self):
            if self.verbose:
                d = self.current.distortion
                msg = "\tDistortion: {0:e}, ".format(d)
                g = linalg_norm(self.current.gradient, ord=np.inf)
                msg += "Gradient: {0:e} ".format(g)
                print(msg)
            if self.verbose_value:
                print("\tValues (%/y):")
                print("\t", end='')
                with printoptions(precision=2, suppress=True):
                    v = np.ravel(self.current.value)
                    print(self.formatter(v))

        def _print_afterword(self):
            if self.verbose:
                if not self.success:
                    msg = "\tDid not converge after "
                else:
                    msg = "\tConverged in "
                msg += "{0:d} iteration(s)...".format(self.niter)
                if self.msg != "":
                    msg += "\n\t" + self.msg
                print(msg)
                self._print_change()
                self._print_current_state()

        def _print_change(self):
            if self.verbose:
                msg = "\t"
                # Change in Distortion
                start_dist = self.starting.distortion
                current_dist = self.current.distortion
                df = np.absolute(current_dist)/np.absolute(start_dist)-1.
                if current_dist < start_dist:
                    df = -df
                msg += "DDistortion={0:.6f}%, ".format(100.*df)
                # Change in Gradient
                start_gradient = np.absolute(self.starting.gradient)
                current_gradient = np.absolute(self.current.gradient)
                non_null = start_gradient > 0.
                dg = np.max(
                    current_gradient[non_null]
                    / start_gradient[non_null]
                    - 1.
                )
                msg += "DGradient(Worst)={0:.6f}%, ".format(100.*dg)
                # Change in Value
                dv = self.current.value/self.starting.value-1.
                largest = np.argmax(np.absolute(dv))
                dv = dv[largest]
                msg += "DValue(Largest)={0:.6f}%".format(100.*dv)
                print(msg)

    def do_bfgs(self):
        f = _ChangedFactory(
            self.factory,
            self.starting_value,
            self.search_bounds
        )
        log = self.Logger(
            f.make,
            f.x0,
            self.search_bounds,
            self.verbose,
            self.formatter
        )
        log._print_preamble("Changed Convex Optimization [BFGS]")

        def func(x): return f.make(x).distortion

        def jac(x): return f.make(x).gradient
        kwoptions = {
            'maxiter': self.maxiter,
            'gtol': 1e-4,
            'disp': False,
            'norm': np.inf
        }
        optimum = minimize(
            func,
            f.x0,
            method='bfgs',
            jac=jac,
            options=kwoptions,
            callback=log
        )
        log.register_optimum(
            optimum.x,
            success=optimum.success,
            msg=optimum.message
        )
        result = f.make(optimum.x)
        if result.distortion < log.starting.distortion:
            self.starting_value = np.ravel(result.value)
        log._print_afterword()
        # TODO Cleaner mechanism to trigger is tight!
        self.is_tight = f.is_tight(result)
        f.clear()  # memory management
        return result.quantizer

    def do_ncg(self):
        f = _ScaledFactory(
            self.factory,
            self.starting_value,
            self.weight
        )
        log = self.Logger(
            f.make, f.x0, self.search_bounds, self.verbose,
            self.formatter)
        log._print_preamble(
            "Scaled Convex Optimization "
            + "[Trust-Region Conjugate-Gradient]"
        )
        optimum = minimize(
            lambda x: f.make(x).distortion,
            f.x0,
            method='trust-ncg',
            jac=lambda x: f.make(x).gradient,
            hess=lambda x: f.make(x).hessian,
            options={
                'maxiter': self.maxiter,
                'gtol': 0.,
                'disp': False,
                'initial_trust_radius': 0.01
            },
            callback=log
        )
        log.register_optimum(optimum.x)
        if not optimum.success:
            log.register_optimum(
                log.x,
                success=False,
                msg=optimum.message
            )
            result = f.make(log.x)
        else:
            log.register_optimum(optimum.x)
            result = f.make(optimum.x)

        if result.distortion < log.starting.distortion:
            self.starting_value = np.ravel(result.value)

        log._print_afterword()
        f.clear()
        return result.quantizer

    def do_newton(self, niter=10):
        # Refine optimum with a few Newton's iteration(s)
        # ... Assuming Hessian is SPD!
        # We should be hitting quadratic convergence
        f = _ScaledFactory(
            self.factory,
            self.starting_value,
            self.weight
        )
        log = self.Logger(
            f.make,
            f.x0,
            self.search_bounds,
            self.verbose,
            self.formatter
        )
        log._print_preamble(
            "Scaled Convex Optimization [Newton's Method]")
        result = f.make(f.x0)
        starting = linalg_norm(result.gradient, ord=np.inf)
        best = starting
        for _ in range(niter):
            inv_hess = np.linalg.pinv(result.hessian)
            new = np.ravel(result.x) - inv_hess.dot(result.gradient)
            log(new)
            current = f.make(new)
            g = linalg_norm(current.gradient, ord=np.inf)
            if g < best:
                result = current
                best = g
        log.register_optimum(result.x)
        if best < starting:
            self.starting_value = np.ravel(result.value)
        log._print_afterword()
        f.clear()
        return result.quantizer

    # Print utilities
    def _print_important(self, msg):
        if self.verbose:
            print("### " + msg + " ###")


class _CachedFactory:

    def __init__(self, make=None):
        if make is not None:
            self._make = make
        self._cache = LRUCache(maxsize=10)

    def clear(self):
        self._cache.clear()

    def make(self, x):
        if not np.all(np.isfinite(x)):
            raise ValueError('Invalid x')
        key = sha1(x).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._make(x)
        out = self._cache.get(key)
        return out


class _ScaledFactory(_CachedFactory):

    class _Quantizer:

        def __init__(self, factory, scale, x):
            self.factory = factory
            self.scale = scale
            self.x = x
            self.size = self.x.size

        @lazy_property
        def quantizer(self):
            if self.value is not None:
                return self.factory.make(self.value)
            else:
                return None

        @lazy_property
        def value(self):
            value = self.scale*self.x
            if np.any(np.diff(value) <= 0.):
                return None
            else:
                return value

        @lazy_property
        def distortion(self):
            if self.quantizer is not None:
                return self.quantizer.distortion
            else:
                return np.finfo(np.float).max

        @lazy_property
        def gradient(self):
            if self.quantizer is not None:
                return self.scale*self.quantizer.gradient
            else:
                return np.finfo(np.float).max*np.ones((self.size,))

        @property
        def hessian(self):
            if self.quantizer is not None:
                return self.scale*self.scale*self.quantizer.hessian
            else:
                return np.zeros((self.size, self.size))

    def __init__(self, factory, v0, weight):
        super().__init__()
        self.factory = factory
        self.v0 = np.ravel(v0)
        self.scale = 1./weight
        self.size = self.v0.size
        self.x0 = self.invert(v0)

    def invert(self, value):
        return np.ravel(value)/self.scale

    def _make(self, x):
        return self._Quantizer(self.factory, self.scale, x)


class _ChangedFactory(_CachedFactory):

    class _Quantizer:

        def __init__(self, factory, search_bounds, x):
            self.OFFSET = _Optimizer1D.OFFSET
            self.factory = factory
            self.search_bounds = search_bounds
            self.x = x

        @lazy_property
        def quantizer(self):
            return self.factory.make(self.value)

        @lazy_property
        def value(self):
            return self.search_bounds[0]+self._logistic

        @lazy_property
        def distortion(self):
            return np.log(self.quantizer.distortion)

        @lazy_property
        def gradient(self):
            out = self.quantizer.gradient.dot(self.jacobian)
            out /= self.quantizer.distortion  # Log-transformation
            return out

        @property
        def hessian(self):
            # Contribution from objective hessian
            out = self.jacobian.T.dot(self.quantizer.hessian)
            out = out.dot(self.jacobian)
            # Contribution from first derivative of logistic function
            df = icumsum(self.quantizer.gradient*self._dlogistic)
            diag_view(out)[:] += df*self._scaled_exp_x
            # Contribution from second derivative of logistic function
            ddf = icumsum(self.quantizer.gradient*self._ddlogistic)
            ud = np.triu(  # Upper diagonal elements
                self._scaled_exp_x[np.newaxis, :]
                * self._scaled_exp_x[:, np.newaxis]
                * ddf[np.newaxis, :])
            out += ud + ud.T - np.diag(ud.diagonal())  # "Symmetrize"
            # Log-transformation...
            out /= self.quantizer.distortion
            out -= (
                self.gradient[:, np.newaxis]
                * self.gradient[np.newaxis, :]
            )
            return out

        @lazy_property
        def jacobian(self):
            out = (
                self._scaled_exp_x[np.newaxis, :]
                * self._dlogistic[:, np.newaxis]
                * np.tri(self.x.size, self.x.size)
            )
            return out

        @lazy_property
        def theta(self):
            return np.cumsum(np.exp(self.x)/(self.x.size+1.))-0.5

        # Internals...

        @lazy_property
        def _logistic(self):
            return self._span*logistic(self.OFFSET*self.theta)

        @lazy_property
        def _dlogistic(self):
            return self._span*dlogistic(self.OFFSET*self.theta)

        @lazy_property
        def _ddlogistic(self):
            return self._span*ddlogistic(self.OFFSET*self.theta)

        @lazy_property
        def _scaled_exp_x(self):
            return self.OFFSET*np.exp(self.x)/(self.x.size+1.)

        @lazy_property
        def _span(self):
            return self.search_bounds[1] - self.search_bounds[0]

    def __init__(self, factory, v0, search_bounds):
        self.OFFSET = _Optimizer1D.OFFSET
        super().__init__()
        v0 = np.ravel(v0)
        self.factory = factory
        self.v0 = v0
        self.size = self.v0.size
        self.search_bounds = _Optimizer1D.expand_search_bounds(
            self.v0,
            search_bounds
        )
        self.x0 = self.invert(v0)

    def _make(self, x):
        return self._Quantizer(self.factory, self.search_bounds, x)

    def invert(self, value):
        value = np.ravel(value)
        span = np.diff(self.search_bounds)
        P = self.size + 1.
        x = (value-self.search_bounds[0])/span
        x = ((ilogistic(x)/self.OFFSET)+0.5)*P
        x = np.diff(np.insert(x, 0, 0.))
        if np.any(x < 0.):
            raise ValueError("Value(s) to invert outside space")
        elif np.any(x == 0.):
            # Make sure values are strictly increasing...
            x[x == 0.] = 1e-4
        x = np.log(x)
        return x

    def is_tight(self, quantizer):
        value = np.ravel(quantizer.value)
        return (
            _Optimizer1D.is_tight(
                value[0], self.search_bounds)
            or _Optimizer1D.is_tight(
                value[-1], self.search_bounds)
        )


###########
# Voronoi #
###########


@workon_axis
def voronoi_1d(quantizer, *, lb=-np.inf, ub=np.inf):
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
    it = np.nditer(quantizer.prev_proba, flags=['multi_index'])
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
