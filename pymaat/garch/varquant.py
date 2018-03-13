from collections import namedtuple
from hashlib import sha1
import warnings
from cachetools import LRUCache
from functools import reduce

import numpy as np
from scipy.linalg import norm as infnorm
from scipy.optimize import minimize, brute, basinhopping
from scipy.stats import norm, truncnorm, uniform

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar
import matplotlib.lines

from pymaat.garch.quant import AbstractQuantization
from pymaat.mathutil import voronoi_1d
from pymaat.util import lazy_property
from pymaat.nputil import diag_view, printoptions


import math
def norm_var(v):
    return 100.*np.sqrt(252.*v)

def round_to_int(x, base=1, fcn=round):
    return int(base * fcn(float(x)/base))

class MarginalVariance(AbstractQuantization):

    def __init__(
        self, model, first_variance, size=100, first_probability=1.,
            nper=None):
        shape = self._init_shape(nper, size)
        first_quantizer = self._make_first_quantizer(
            first_variance,
            first_probability)
        super().__init__(model, shape, first_quantizer)

    def plot_distortion(self):
        all_quant = self.all_quantizers[1:]
        y = [norm_var(np.sqrt(q.distortion)) for q in all_quant]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        ax.plot(np.arange(len(y))+1, np.array(y))

    def plot_values(self):
        fig = plt.figure()
        main_ax = fig.add_axes([0.1,0.1,0.7,0.85])


        all_quant = self.all_quantizers[1:]
        all_vol = np.concatenate([norm_var(q.value) for q in all_quant])
        all_probas = np.concatenate([100.*q.probability for q in all_quant])

        def _get_bounds(values, lb=-np.inf, ub=np.inf):
            lowest = max(lb, round_to_int(np.amin(values)*0.9,
                base=5, fcn=math.floor))
            highest = min(ub, round_to_int(np.amax(values)*1.1,
                base=5, fcn=math.ceil))
            return (lowest, highest)

        vol_bounds = _get_bounds(all_vol, lb=0.)
        proba_bounds = _get_bounds(all_probas, lb=0., ub=100.)

        cmap = matplotlib.cm.get_cmap('gnuplot2')
        cmap = cmap.reversed()
        norm = matplotlib.colors.Normalize(vmin=proba_bounds[0],
                vmax=proba_bounds[1])

        # Main plot
        for (t, quant) in enumerate(all_quant):
            # Plot tiles
            x_tiles = np.array([t+0.5, t+1.5])
            y_tiles = norm_var(quant.voronoi)
            y_tiles[0] = norm_var(quant.parent.min_variance)
            y_tiles[-1] = vol_bounds[1]
            x_tiles, y_tiles = np.meshgrid(x_tiles, y_tiles)
            main_ax.pcolor(
                    x_tiles, y_tiles,
                    100.*quant.probability[:, np.newaxis],
                    cmap=cmap, norm=norm)

            # Plot singularities
            for s in norm_var(quant.parent.singularities):
                main_ax.plot(np.array([t+0.5, t+1.5]),
                        np.array([s, s]), '-r', linewidth=1)
            #  Plot quantizer
            y_pts = norm_var(quant.value)
            x_pts = np.broadcast_to(t+1, y_pts.shape)
            main_ax.scatter(x_pts, y_pts, c='k', s=4, marker=".")

        # Y-Axis
        main_ax.set_ylabel(r'Annualized Volatility (%)')
        main_ax.set_ylim(vol_bounds)
        # X-Axis
        main_ax.set_xlim(0.5, t+1.5)
        main_ax.set_xlim(17.5, 21.5)
        # main_ax.set_xlabel(r'Trading Days ($t$)')
        # tickspace = round_to_int(0.1*t, base=5, fcn=math.ceil)
        # ticks = np.arange(tickspace, t+1, tickspace)
        # main_ax.xaxis.set_ticks(ticks)
        # main_ax.xaxis.set_major_formatter(
        #         matplotlib.ticker.FormatStrFormatter('%d'))

        # Add colorbar
        cbar_ax = fig.add_axes([0.85,0.1,0.05,0.85])
        matplotlib.colorbar.ColorbarBase(
                cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cbar_ax.set_ylabel(r'%')

        return fig, main_ax, cbar_ax

    def visualize_transition(self, t=1):
        pass
        # if t == 0:
        #     raise ValueError
        # x = np.sqrt(252.*mc.values[t])*100.
        # selected = np.logical_and(x > 10., x < 30.)
        # x = x[selected, np.newaxis]
        # y = np.sqrt(252.*mc.values[t+1])*100.
        # y = y[np.newaxis, selected]
        # c = mc.transition_probabilities[t]*100.
        # c = c[selected][:, selected]
        # x, y = np.broadcast_arrays(x, y)

        # fig, ax = plt.subplots()

        # # Add scatter
        # cax = plt.scatter(x.flatten(), y.flatten(), c=c.flatten(),
        #                   marker=",", cmap=cmap,
        #                   s=ss, linewidths=0.15)

        # plt.xlabel(r'Annualized Volatility (%) at t=10', fontsize=fs)
        # plt.ylabel(r'Annualized Volatility (%) at t=11', fontsize=fs)
        # plt.xticks([10, 15, 20, 25, 30], fontsize=fs)
        # plt.yticks([10, 15, 20, 25, 30], fontsize=fs)
        # plt.xlim(10, 30)
        # plt.ylim(10, 30)
        # plt.tight_layout()

        # # Add colorbar
        # cbar = fig.colorbar(cax)
        # cbar.ax.set_title(r'$p_{h}^{(ij)}$(%)', fontsize=fs)

        # fig.set_size_inches(4, 3)
        # ax.grid(True, linestyle='dashed', alpha=0.5)
        # plt.tight_layout()

    def quantize(self, t, values):
        pass

    @staticmethod
    def _init_shape(nper, size):
        if nper is not None:
            shape = np.broadcast_to(size, (nper,))
        else:
            shape = np.atleast_1d(size)
        return shape

    @staticmethod
    def _make_first_quantizer(value, probability):
        # Set first values
        value = np.ravel(np.atleast_1d(value))
        # Set first probabilities
        probability = np.broadcast_to(probability, value.shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)
        return _QuantizerFactory.make_stub(value, probability)

    def _one_step_optimize(self, size, previous, *, verbose=True):
        factory = _QuantizerFactory(self.model, size, previous)
        if previous.size == 1:
            if verbose:
                print(" using trust-region", end='')
            quant = factory.optimize(verbose=verbose)
        else:
            if verbose:
                print(" using Basin-Hopping", end='')
            quant = factory.robust_optimize(verbose=verbose)
        return quant

class _QuantizerFactory:

    def __init__(self, model, size, previous):
        if np.any(previous.probability <= 0.):
            raise ValueError(
                "Previous probabilities must be strictly positive")
        self.model = model
        self.shape = (size,)
        self.size = size
        self.previous = previous
        self.variance_span = max(
            self.previous.value[-1]/self.min_variance-1., 1.)
        self._cache = LRUCache(maxsize=30)
        self._func = lambda x: self.make_unconstrained(x).distortion
        self._jac = lambda x: self.make_unconstrained(x).gradient
        self._hess = lambda x: self.make_unconstrained(x).hessian
        self._dist_scale = (
                self.min_variance**2./(self.previous.size*self.size**2.)
                )
        self._gtol = self._dist_scale * 1e-4
        self._method = 'trust-ncg'
        self._options = {
            'maxiter': 10000,
            'disp': False,
            'gtol': self._gtol} #  when using trust-ncg

    def optimize(self, *, x0=None, verbose=True):
        """
        Typical quantizer solver. Might stall on a local minimum.
        """
        if verbose:
            print(" with gtol={0:.6e}".format(self._gtol))

        def optimize_from(x0_): return minimize(
            self._func, x0_, method=self._method,
            jac=self._jac, hess=self._hess,
            options=self._options,
            callback=self._get_callback(verbose)
        )
        return self._try_optimization(optimize_from, x0)

    def robust_optimize(
            self, *,
            niter=1000, niter_success=10, interval=50,
            seed=None, x0=None, verbose=True
            ):
        """
        Use for large quantizers.
        """
        if niter_success is None:
            self._niter_success = niter+2
        else:
            self._niter_success = niter_success
        if verbose:
            msg = " with gtol={0:.6e}".format(self._gtol)
            msg += ", niter={0:d}".format(niter)
            msg += ", niter_success={0:d}".format(self._niter_success)
            msg += " and interval={0:d}".format(interval)
            print(msg)
        minimizer_kwargs = {
            'method': self._method,
            'jac': self._jac,
            'hess': self._hess,
            'options': self._options
        }
        def optimize_from(x0_): return basinhopping(
            self._func, x0_, niter=niter, T=self._dist_scale,
            take_step=_OptimizerTakeStep(self, seed=seed, persistent=False),
            disp=False,
            minimizer_kwargs=minimizer_kwargs,
            callback=self._get_callback(verbose),
            interval=interval,
        )
        return self._try_optimization(optimize_from, x0)

    def _try_optimization(self, optimize_from, x0=None):
        if x0 is None:
            if self.size == self.previous.size:
                # Start from previous values...
                x0 = _Unconstrained._inv_change_of_variable(
                        self, self.previous.value)
            else:
                x0 = np.zeros(self.shape)
        try:
            opt = optimize_from(x0)
        except _AtSingularity as exc:
            # Optimizer met a singularity for indices idx
            # Move by arbitrary displacement
            x0[exc.idx] += 1e-2
            return self._try_optimization(optimize_from, x0)
        return self.make_unconstrained(opt.x)

    def brute_optimize(self, search_width=3.):
        """
        Solve quantizer by brute force. Recommended use: only when size<=3.
        """
        ranges = ((-search_width, search_width),)*self.size
        sol = brute(self._func, ranges)
        return self.make_unconstrained(sol)

    def _get_callback(self, verbose):
        def cb(x, f=None, a=None):
            return self._callback(x, verbose)
        return cb

    def _callback(self, x, verbose):
        q = self.make_unconstrained(x)

        # Initialization
        if not hasattr(self, '_distortion'):
            self._distortion = q.distortion
            self._value = q.value
            self._niter = 0

        # If improve by at least 0.01%...
        if self._niter == 0 or q.distortion*(1.+1e-4) < self._distortion:
            if verbose:
                objective = 100.*(q.distortion/self._distortion-1.)
                value = 100.*np.amax(np.absolute(q.value/self._value-1.))
                msg = "[Iter#{0:d}] ".format(self._niter)
                if self._niter > 0:
                    msg += "New minimum [df={0:+.2f}%, dx={1:.2f}%] ".format(
                        objective, value)
                msg += "Distortion: {0:.6e}, ".format(q.distortion)
                msg += "Gradient: {0:.6e}".format(infnorm(q.gradient))
                print(msg)
            # Update
            self._niter_since_last = 0
            self._distortion = q.distortion
            self._value = q.value
        else:
            self._niter_since_last += 1

        self._niter += 1

        if hasattr(self, '_niter_success'):
            # Stop flag only used when using basin-hopping
            stop_flag = self._niter_since_last >= self._niter_success
            if verbose and stop_flag:
                msg = "Early termination: could not improve for "
                msg += "{0:d} consecutive tries".format(
                        self._niter_since_last)
                print(msg)
            return stop_flag
        else:
            return None

    def make_unconstrained(self, x):
        x.flags.writeable = False
        key = sha1(x).hexdigest()
        if key not in self._cache:
            quant = _Unconstrained(self, x)
            self._cache[key] = quant
        return self._cache.get(key)

    def make(self, value):
        return _Quantizer(self, value)

    @staticmethod
    def make_stub(value, probability):
        """
        Returns a minimal quantizer-like object which may be used
            as a previous quantizer when instantiating a subsequent
            factory.
        Rem. It is used for initializing recursions and testing
        """
        if np.any(np.isnan(value)) or np.any(~np.isfinite(value)):
            msg = 'Invalid value: NaNs or infinite'
            raise ValueError(msg)
        if np.any(value <= 0.):
            msg = 'Invalid value: variance must be strictly positive'
            raise ValueError(msg)
        if (
            np.abs(np.sum(probability)-1.) > 1e-12
            or np.any(probability < 0.)
        ):
            msg = 'Probabilities must sum to one and be positive'
            raise ValueError(msg)
        fact = namedtuple('_Quantizer',
                          ['value', 'probability', 'size', 'shape'])
        return fact(value=value, probability=probability,
                    size=value.size, shape=value.shape)

    # Utilities

    @lazy_property
    def min_variance(self):
        out = self.model.get_lowest_one_step_variance(
            np.amin(self.previous.value))
        # print(np.sqrt(252.*out)*100.)
        return out

    @lazy_property
    def singularities(self):
        return np.sort(
            self.model.get_lowest_one_step_variance(
                self.previous.value))


class _OptimizerTakeStep():

    def __init__(self, parent, seed=None, persistent=False):
        self.random_state = np.random.RandomState(seed)
        self.parent = parent
        self.stepsize = 1.
        self.persistent = persistent
        if self.parent.previous.size > 1:
            self.scale = np.mean(np.diff(self.parent.singularities))
        else:
            self.scale = self.parent.min_variance/self.parent.size

    def __call__(self, x):
        # Search bounds
        lb = self.parent.min_variance
        ub = self.parent.singularities[-1] + self.scale
        if self.persistent:
            value = self._persistent(self.stepsize, lb, ub, x)
        else:
            value = self._non_persistent(lb, ub)
        new_x = _Unconstrained._inv_change_of_variable(self.parent, value)
        # Bound x to avoid numerical instability
        return np.clip(new_x, -10., 20.)

    def _persistent(self, step, lb, ub, x):
        _, value = _Unconstrained._change_of_variable(self.parent, x)
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

class _Quantizer():

    def __init__(self, parent, value):
        if not isinstance(parent, _QuantizerFactory):
            msg = 'Quantizers must be instantiated from valid factory'
            raise ValueError(msg)
        else:
            self.parent = parent
            self.model = parent.model
            self.size = parent.size
            self.shape = parent.shape
            self.previous = parent.previous

        value = np.asarray(value)
        if value.shape != self.shape:
            msg = 'Unexpected quantizer shape'
            raise ValueError(msg)
        elif (np.any(np.isnan(value)) or np.any(~np.isfinite(value))):
            msg = 'Invalid quantizer (NaN or inf)'
            raise ValueError(msg)
        elif np.any(np.diff(value) <= 0.):
            msg = ('Unexpected quantizer value(s): '
                   'must be strictly increasing')
            raise ValueError(msg)
        else:
            self.value = value

    # Results
    @lazy_property
    def voronoi(self):
        """
        Returns a self.size+1 np.array representing the 1D voronoi tiles
        """
        return voronoi_1d(self.value, lb=0.)

    @property
    def probability(self):
        """
        Returns a self.size np.array containing the time-t probability
        """
        return self.previous.probability.dot(self.transition_probability)

    @property
    def transition_probability(self):
        """
        Returns a self.previous.size-by-self.size np.array
        containing the transition probabilities from time t-1 to time t
        """
        return self._integral[0]

    @property
    def distortion(self):
        """
        Returns a (scalar) distortion function to be optimized
        """
        elements = np.sum(self._distortion_elements, axis=1)
        return self.previous.probability.dot(elements)

    @property
    def gradient(self):
        """
        Returns self.size np.array representing the gradient
        of self.distortion
        """
        out = -2. * self.previous.probability.dot(
            self._integral[1]
            - self._integral[0]
            * self.value[np.newaxis, :])
        return out

    @property
    def hessian(self):
        """
        Returns a self.size-by-self.size np.array containing the
        Hessian of self.distortion
        """
        with np.errstate(invalid='ignore'):
            # (minus, plus)
            terms = ((self.value - self.voronoi[:-1])*self._delta[:, :-1],
                     (self.voronoi[1:] - self.value)*self._delta[:, 1:])
            # Catch inf*zero indetermination at voronoi=np.inf...
            terms[1][:, -1] = 0.

        # Build Hessian
        hess = np.zeros((self.size, self.size))
        main_diagonal = -2. * self.previous.probability.dot(
            terms[0] + terms[1] - self._integral[0])
        diag_view(hess, k=0)[:] = main_diagonal
        if self.size > 1:
            off_diagonal = -2. * self.previous.probability.dot(
                terms[0][:, 1:])
            diag_view(hess, k=1)[:] = off_diagonal
            diag_view(hess, k=-1)[:] = off_diagonal
        return hess

    #############
    # Internals #
    #############

    @property
    def _distortion_elements(self):
        """
        Returns the un-weighted (self.previous.size-by-self.size)
        elements of the distortion to be summed, namely
        ```
            (\\mathcal{I}_{2,t}^{(ij)} - 2\\mathcal{I}_{1,t}^{(ij)} h_t^{(j)}
            + \\mathcal{I}_{0,t}^{(ij)} (h_t^{(j)})^2
        ```
        Rem. Output only has valid entries, ie free of NaNs
        """
        return (self._integral[2]
                - 2.*self._integral[1]*self.value[np.newaxis, :]
                + self._integral[0]*self.value[np.newaxis, :]**2.)

    @lazy_property
    def _integral(self):
        """
        Returns of len-3 list indexed by order and filled with
        self.previous.size-by-self.size arrays containing
        ```
            \\mathcal{I}_{order,t}^{(ij)}
        ```
        for all i,j.
        Rem. Output only has valid entries, ie free of NaNs
        BOTTLENECK
        """
        roots = self._roots
        previous_value = self.previous.value[:, np.newaxis]
        cdf = self._cdf
        pdf = self._pdf

        def _get_integral(order):
            integral = [self.model.one_step_expectation_until(
                previous_value, roots[right], order=order,
                _pdf=pdf[right], _cdf=cdf[right])
                for right in [False, True]]
            out = integral[True] - integral[False]
            out[self._no_roots] = 0.0
            return np.diff(out, n=1, axis=1)
        return [_get_integral(order) for order in [0, 1, 2]]

    @lazy_property
    def _delta(self):
        """
        Returns a previous.size-by-size+1 array containing
        ```
            \\delta_{t}^{(ij\\pm)}
        ```
        for all i,j.
        Rem. Output only has valid entries, ie free of NaNs, but may be
            infinite (highly unlikely)
        In particular,
            (1) When no root exists (eg. voronoi==0), delta is zero
            (2) When voronoi==+np.inf, delta is zero
            (3) When voronoi is at a root singularity, delta is np.inf
        """
        unsigned_root_derivative = \
            self.model.one_step_roots_unsigned_derivative(
                self.previous.value[:, np.newaxis],
                self.voronoi[np.newaxis, :])
        limit_index = unsigned_root_derivative == np.inf
        if np.any(limit_index):
            warnings.warn(
                "Voronoi tiles at singularity detected",
                UserWarning
            )
        out = (0.5*unsigned_root_derivative
               * (self._pdf[True]+self._pdf[False]))
        # Limit cases
        # (1)  When a root does not exist, delta is 0
        out[self._no_roots] = 0.
        # (2) When voronoi is +np.inf (which always occur
        #   at the last column), delta is zero.
        out[:, -1] = 0.
        # (3) When roots approach the singularity, delta is np.inf
        out[limit_index] = np.inf
        return out

    @lazy_property
    def _pdf(self):
        """
        Returns a len-2 tuples containing (left, right) PDF of roots
        """
        roots = self._roots
        with np.errstate(invalid='ignore'):
            return [norm.pdf(roots[right]) for right in [False, True]]

    @lazy_property
    def _cdf(self):
        """
        Returns a len-2 tuples containing (left, right) CDF roots
        """
        roots = self._roots
        with np.errstate(invalid='ignore'):
            return [norm.cdf(roots[right]) for right in [False, True]]

    @lazy_property
    def _roots(self):
        """
        Returns a len-2 tuples containing (left, right) roots
        """
        return self.model.one_step_roots(
            self.previous.value[:, np.newaxis],
            self.voronoi[np.newaxis, :])

    @lazy_property
    def _no_roots(self):
        return np.isnan(self._roots[0])


class _AtSingularity(Exception):
    def __init__(self, idx):
        self.idx = np.unique(idx)


# class _UnconstrainedNaive(_Quantizer):
#     def __init__(self, parent, x):
#         if not isinstance(parent, _QuantizerFactory):
#             msg = 'Quantizers must be instantiated from valid factory'
#             raise ValueError(msg)
#         x = np.asarray(x)
#         self.x = x
#         if parent.size == parent.previous.size:
#             value = parent.previous.value + self.x
#         else:
#             value = np.linspace(
#                     parent.min_variance,
#                     parent.previous.value[-1],
#                     num=parent.size) + self.x
#         value.sort()
#         # Let super catch invalid value...
#         super().__init__(parent, value)

class _Unconstrained(_Quantizer):

    def __init__(self, parent, x):
        if not isinstance(parent, _QuantizerFactory):
            msg = 'Quantizers must be instantiated from valid factory'
            raise ValueError(msg)
        x = np.asarray(x)
        self.x = x
        self._scaled_exp_x, value = self._change_of_variable(parent, x)
        # Let super catch invalid value...
        super().__init__(parent, value)

    @staticmethod
    def _change_of_variable(parent, x):
        scaled_exp_x = (parent.variance_span * parent.min_variance
                        * np.exp(x) / (parent.size+1.))
        value = (parent.min_variance + np.cumsum(scaled_exp_x))
        return (scaled_exp_x, value)

    @staticmethod
    def _inv_change_of_variable(parent, value):
        padded_values = np.insert(value, 0, parent.min_variance)
        expx = (
            (parent.size+1.)*np.diff(padded_values)
            / (parent.min_variance * parent.variance_span)
        )
        return np.log(expx)

    @property
    def gradient(self):
        """
        Returns the transformed gradient for the distortion function, ie
        dDistortion/ dx = dDistortion/dGamma * dGamma/dx
        """
        grad = super().gradient
        out = grad.dot(self._jacobian)
        return out

    @property
    def hessian(self):
        """
        Returns the transformed Hessian for the distortion function
        """
        s = super()

        def get_first_term():
            jac = self._jacobian
            hess = s.hessian
            at_singularity = np.logical_not(np.isfinite(hess))
            if np.any(at_singularity):
                w = np.where(at_singularity)
                raise _AtSingularity(w[1])
            return jac.T.dot(hess).dot(jac)

        def get_second_term():
            inv_cum_grad = np.flipud(np.cumsum(np.flipud(s.gradient)))
            return np.diag(self._scaled_exp_x * inv_cum_grad)

        hess = get_first_term() + get_second_term()

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
        return self._scaled_exp_x[np.newaxis, :]*mask
