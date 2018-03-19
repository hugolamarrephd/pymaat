import warnings
from collections import namedtuple
import math

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar

from pymaat.quantutil import Optimizer
from pymaat.quantutil import AbstractQuantization
from pymaat.quantutil import AbstractFactory
from pymaat.quantutil import AbstractQuantizer
from pymaat.quantutil import AtSingularity
from pymaat.garch.format import variance_formatter
from pymaat.mathutil import round_to_int
from pymaat.util import lazy_property
from pymaat.nputil import printoptions
import pymaat.plotutil


stub_quantizer = namedtuple('_Quantizer',
                            ['value',
                             'probability',
                             'size',
                             'shape',
                             'ndim'])


def make_stub_quantizer(value, probability):
    """
    Returns a quantizer-like object which may be used
        as a previous quantizer e.g.
        for initializing recursions and testing
    """
    if np.any(~np.isfinite(value)):
        err_msg = 'Invalid value: NaNs or infinite\n'
    elif np.any(value <= 0.):
        err_msg = 'Invalid value: must be strictly positive\n'
    elif probability.size != value.size:
        err_msg = 'Invalid probabilities: size mismatch\n'
    elif np.abs(np.sum(probability)-1.) > 1e-12:
        err_msg = 'Invalid probabilities: must sum to one\n'
    elif np.any(probability < 0.):
        err_msg = 'Invalid probabilities: must be positive\n'

    if 'err_msg' in locals():
        raise ValueError(err_msg)
    else:
        return stub_quantizer(value=value, probability=probability,
                              size=value.size, shape=value.shape,
                              ndim=1)


class MarginalVariance(AbstractQuantization):

    def __init__(
        self, model, first_variance, size=100, first_probability=1.,
            nper=None, freq='daily'):
        super().__init__(model, variance_formatter(freq))
        self.nper = nper
        self.size = size
        self.first_variance = first_variance
        self.first_probability = first_probability

    def plot_distortion(self):
        all_quant = self.all_quantizers[1:]
        y = [q.distortion for q in all_quant]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        ax.plot(np.arange(len(y))+1, np.array(y))

    def plot_values(self):
        fig = plt.figure()
        main_ax = fig.add_axes([0.1, 0.1, 0.7, 0.85])

        # Compute bounds
        all_quant = self.all_quantizers[1:]
        all_vol = np.concatenate(
            [self.print_formatter(q.value) for q in all_quant])
        all_probas = np.concatenate(
            [100.*q.probability for q in all_quant])
        vol_bounds = pymaat.plotutil.get_lim(
            all_vol, lb=0.)
        proba_bounds = pymaat.plotutil.get_lim(
            all_probas, lb=0., ub=100.)

        # Initialize color map
        cmap = matplotlib.cm.get_cmap('gnuplot2')
        cmap = cmap.reversed()
        norm = matplotlib.colors.Normalize(vmin=proba_bounds[0],
                                           vmax=proba_bounds[1])

        # Main plot
        for (t, quant) in enumerate(all_quant):
            # Plot tiles
            x_tiles = np.array([t+0.5, t+1.5])
            y_tiles = self.print_formatter(quant.voronoi)
            y_tiles[0] = self.print_formatter(quant.parent.min_variance)
            y_tiles[-1] = vol_bounds[1]
            x_tiles, y_tiles = np.meshgrid(x_tiles, y_tiles)
            main_ax.pcolor(
                x_tiles, y_tiles,
                100.*quant.probability[:, np.newaxis],
                cmap=cmap, norm=norm)

            #  Plot quantizer
            y_pts = self.print_formatter(quant.value)
            x_pts = np.broadcast_to(t+1, y_pts.shape)
            main_ax.scatter(x_pts, y_pts, c='k', s=2, marker=".")

        # Y-Axis
        main_ax.set_ylabel(r'Annualized Volatility (%)')
        main_ax.set_ylim(vol_bounds)
        # X-Axis
        main_ax.set_xlim(0.5, t+1.5)
        main_ax.set_xlabel(r'Trading Days ($t$)')
        tickspace = round_to_int(0.1*t, base=5, fcn=math.ceil)
        ticks = np.arange(tickspace, t+1, tickspace)
        main_ax.xaxis.set_ticks(ticks)
        main_ax.xaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%d'))

        # Add colorbar
        cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.85])
        matplotlib.colorbar.ColorbarBase(
            cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cbar_ax.set_ylabel(r'%')

    def plot_transition_at(self, t=1):
        if t <= 0:
            raise ValueError
        fig = plt.figure()
        main_ax = fig.add_axes([0.1, 0.1, 0.7, 0.85])

        prev = self.all_quantizers[t-1]
        current = self.all_quantizers[t]

        x_pts = self.print_formatter(prev.value)[:, np.newaxis]
        y_pts = self.print_formatter(current.value)[np.newaxis, :]
        x_bounds = pymaat.plotutil.get_lim(x_pts, lb=0)
        y_bounds = pymaat.plotutil.get_lim(y_pts, lb=0)

        x_tiles = self.print_formatter(prev.voronoi)
        x_tiles[0] = self.print_formatter(prev.parent.min_variance)
        x_tiles[-1] = x_bounds[1]

        y_tiles = self.print_formatter(current.voronoi)
        y_tiles[0] = self.print_formatter(current.parent.min_variance)
        y_tiles[-1] = y_bounds[1]

        proba = current.transition_probability*100.
        proba_bounds = pymaat.plotutil.get_lim(proba, lb=0, ub=100)

        # Initialize color map
        cmap = matplotlib.cm.get_cmap('gnuplot2')
        cmap = cmap.reversed()
        norm = matplotlib.colors.Normalize(vmin=proba_bounds[0],
                                           vmax=proba_bounds[1])
        # Color tiles
        main_ax.pcolor(
            x_tiles, y_tiles, proba, cmap=cmap, norm=norm)

        # Plot quantizers
        x_pts, y_pts = np.broadcast_arrays(x_pts, y_pts)
        main_ax.scatter(x_pts, y_pts, c='k', s=2, marker=".")

        # X-Axis
        main_ax.set_xlabel('Annualized Volatility (%) on t={}'.format(t-1))
        main_ax.set_xlim(x_bounds)

        # Y-Axis
        main_ax.set_ylabel('Annualized Volatility (%) on t={}'.format(t))
        main_ax.set_ylim(y_bounds)

        # Add colorbar
        cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.85])
        matplotlib.colorbar.ColorbarBase(
            cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cbar_ax.set_ylabel(r'%')

    def quantize(self, t, values):
        pass

    def _make_first_quantizer(self):
        # Set first values
        value = np.ravel(self.first_variance)
        # Set first probabilities
        probability = np.broadcast_to(self.first_probability, value.shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)
        return self.make_stub(value, probability)

    def _get_all_shapes(self):
        if self.nper is not None:
            return np.broadcast_to(self.size, (self.nper,))
        else:
            return np.ravel(self.size)

    def _one_step_optimize(self, t, shape, previous, *, verbose=True):
        factory = _UncQuantizerFactory(self.model, shape, previous)
        optimizer = Optimizer(factory)
        return optimizer.optimize(verbose=verbose)

class _UncQuantizerFactory(AbstractFactory):

    def __init__(self, model, size, previous):
        super().__init__(model, size, previous, target=_Unconstrained)

    def get_distortion_scale(self):
        return self.min_variance**2./(self.previous.size*self.size**2.)

    @lazy_property
    def min_variance(self):
        out = self.model.get_lowest_one_step_variance(
            np.amin(self.previous.value))
        return out

    @lazy_property
    def singularities(self):
        return np.sort(
            self.model.get_lowest_one_step_variance(
                self.previous.value))

class _QuantizerFactory(AbstractFactory):

    def __init__(self, model, size, previous):
        super().__init__(model, size, previous, target=_Quantizer)

    def get_distortion_scale(self):
        return self.min_variance**2./(self.previous.size*self.size**2.)

    @lazy_property
    def min_variance(self):
        out = self.model.get_lowest_one_step_variance(
            np.amin(self.previous.value))
        return out

    @lazy_property
    def singularities(self):
        return np.sort(
            self.model.get_lowest_one_step_variance(
                self.previous.value))

class _Quantizer(AbstractQuantizer):

    def __init__(self, parent, value, *, root_lb=-np.inf, root_ub=np.inf):
        super().__init__(parent, value, lb=0.)
        self.root_lb = np.broadcast_to(root_lb, self.previous.shape)
        self.root_ub = np.broadcast_to(root_ub, self.previous.shape)
        if self.previous.ndim == 1:
            self.broad_root_lb = self.root_lb[:, np.newaxis]
            self.broad_root_ub = self.root_ub[:, np.newaxis]
        elif self.previous.ndim == 2:
            self.broad_root_lb = self.root_lb[:, :, np.newaxis]
            self.broad_root_ub = self.root_ub[:, :, np.newaxis]
        else:
            raise ValueError("Unexpected previous dimension")

    # Overridden to allow conditioning
    @lazy_property
    def previous_probability(self):
        return self.previous.probability / (
            norm.cdf(self.root_ub) - norm.cdf(self.root_lb)
        )

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
        """
        def bound_roots(right):
            a = self._roots[right][..., :-1]
            b = self._roots[right][..., 1:]
            if not right:
                a, b = b, a
            a = np.maximum(self.broad_root_lb, a)
            b = np.minimum(self.broad_root_ub, b)
            b = np.maximum(a, b)
            return (a, b)
        a, b = bound_roots(False)  # Left-branch intervals
        c, d = bound_roots(True)  # Right-branch intervals

        # Computational bottleneck here...
        bounds = [a, b, c, d]
        pdf = [norm.pdf(z) for z in bounds]
        cdf = [norm.cdf(z) for z in bounds]

        def I(p, i): return self.model.one_step_expectation_until(
            self.broad_previous_value,
            bounds[i],
            order=p,
            _pdf=pdf[i], _cdf=cdf[i])

        def integrate(p): return (I(p, 1)-I(p, 0)) + (I(p, 3)-I(p, 2))
        return [integrate(p) for p in [0, 1, 2]]

    @lazy_property
    def _delta(self):
        """
        Returns a previous.size-by-size+1 array containing
        ```
            \\delta_{t}^{(ij\\pm)}
        ```
        for all i,j.
        Rem. Output only has valid entries (ie free of NaNs or inf)
        In particular when voronoi==+np.inf, delta is zero
        """
        # Delta(0)
        dX = self.model.one_step_real_roots_unsigned_derivative(
            self.broad_previous_value,
            self.broad_voronoi)
        delta0 = np.zeros_like(dX)
        for right in [False, True]:
            idx = np.logical_and(
                self._roots[right] > self.broad_root_lb,
                self._roots[right] < self.broad_root_ub
            )
            pdf_tmp = norm.pdf(self._roots[right][idx])
            dX_tmp = dX[idx]
            delta0[idx] += 0.5 * pdf_tmp * dX_tmp

        # Delta(1)
        delta1 = np.empty_like(delta0)
        delta1[...,:-1] = delta0[...,:-1] * self.broad_voronoi[...,:-1]
        delta1[:, -1] = 0.  # is zero when voronoi is +np.inf (last column)

        return (delta0, delta1)

    @lazy_property
    def _roots(self):
        return self.model.one_step_real_roots(
            self.broad_previous_value,
            self.broad_voronoi
        )


class _Unconstrained(_Quantizer):

    def __init__(self, parent, x):
        x = np.asarray(x)
        self.x = x
        self._scaled_exp_x, value = self._change_of_variable(parent, x)
        # Let super catch invalid value...
        super().__init__(parent, value)

    @staticmethod
    def _change_of_variable(parent, x):
        scaled_exp_x = (parent.min_variance
                        * np.exp(x) / (parent.size+1.))
        value = (parent.min_variance + np.cumsum(scaled_exp_x))
        return (scaled_exp_x, value)

    @staticmethod
    def _inv_change_of_variable(parent, value):
        padded_values = np.insert(value, 0, parent.min_variance)
        expx = (
            (parent.size+1.)*np.diff(padded_values)
            / (parent.min_variance)
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
                raise AtSingularity(w[1])
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

# class _QuantizerFactory_Old:

#    def __init__(self, model, size, previous):
#        if np.any(previous.probability <= 0.):
#            raise ValueError(
#                "Previous probabilities must be strictly positive")
#        self.model = model
#        self.shape = (size,)
#        self.size = size
#        self.previous = previous
#        # Setting up internal state..
#        self._cache = LRUCache(maxsize=30)
#        self._func = lambda x: self.make_unconstrained(x).distortion
#        self._jac = lambda x: self.make_unconstrained(x).gradient
#        self._hess = lambda x: self.make_unconstrained(x).hessian
#        self._dist_scale = (
#            self.min_variance**2./(self.previous.size*self.size**2.)
#        )
#        self._gtol = self._dist_scale * 1e-4
#        self._method = 'trust-ncg'
#        self._options = {
#            'maxiter': 10000,
#            'disp': False,
#            'gtol': self._gtol}

#    def optimize(self, *, x0=None, verbose=True):
#        """
#        Typical quantizer solver. Might stall on a local minimum.
#        """
#        if verbose:
#            print("with gtol={0:.6e}".format(self._gtol))

#        def optimize_from(x0_): return minimize(
#            self._func, x0_, method=self._method,
#            jac=self._jac, hess=self._hess,
#            options=self._options,
#            callback=self._get_callback(verbose)
#        )
#        return self._try_optimization(optimize_from, x0)

#    def robust_optimize(
#            self, *,
#            niter=1000, niter_success=10, interval=50,
#            seed=None, x0=None, verbose=True
#    ):
#        """
#        Use for large quantizers.
#        """
#        if niter_success is None:
#            self._niter_success = niter+2
#        else:
#            self._niter_success = niter_success
#        if verbose:
#            msg = "with gtol={0:.6e}".format(self._gtol)
#            msg += ", niter={0:d}".format(niter)
#            msg += ", niter_success={0:d}".format(self._niter_success)
#            msg += " and interval={0:d}".format(interval)
#            print(msg)
#        minimizer_kwargs = {
#            'method': self._method,
#            'jac': self._jac,
#            'hess': self._hess,
#            'options': self._options
#        }

#        def optimize_from(x0_): return basinhopping(
#            self._func, x0_, niter=niter, T=self._dist_scale,
#            take_step=_OptimizerTakeStep(self, seed=seed, persistent=True),
#            disp=False,
#            minimizer_kwargs=minimizer_kwargs,
#            callback=self._get_callback(verbose),
#            interval=interval,
#        )
#        return self._try_optimization(optimize_from, x0)

#    def _try_optimization(self, optimize_from, x0=None):
#        if x0 is None:
#            if self.size == self.previous.size:
#                # Start from previous values...
#                x0 = _Unconstrained._inv_change_of_variable(
#                    self, self.previous.value)
#            else:
#                x0 = np.zeros(self.shape)
#        try:
#            opt = optimize_from(x0)
#        except _AtSingularity as exc:
#            # Optimizer met a singularity for indices idx
#            # Move by arbitrary displacement
#            x0[exc.idx] += 1e-2
#            return self._try_optimization(optimize_from, x0)
#        return self.make_unconstrained(opt.x)

#    def brute_optimize(self, search_width=3.):
#        """
#        Solve quantizer by brute force. Recommended use: only when size<=3.
#        """
#        ranges = ((-search_width, search_width),)*self.size
#        sol = brute(self._func, ranges)
#        return self.make_unconstrained(sol)

#    def _get_callback(self, verbose):
#        def cb(x, f=None, a=None):
#            return self._callback(x, verbose)
#        return cb

#    def _callback(self, x, verbose):
#        q = self.make_unconstrained(x)

#        # Initialization
#        if not hasattr(self, '_distortion'):
#            self._distortion = q.distortion
#            self._value = q.value
#            self._niter = 0

#        # If improve by at least 0.01%...
#        if self._niter == 0 or q.distortion*(1.+1e-4) < self._distortion:
#            if verbose:
#                objective = 100.*(q.distortion/self._distortion-1.)
#                value = 100.*np.amax(np.absolute(q.value/self._value-1.))
#                msg = "[Iter#{0:d}] ".format(self._niter)
#                if self._niter > 0:
#                    msg += "New minimum [df={0:+.2f}%, dx={1:.2f}%] ".format(
#                        objective, value)
#                msg += "Distortion: {0:.6e}, ".format(q.distortion)
#                msg += "Gradient: {0:.6e}".format(infnorm(q.gradient))
#                print(msg)
#            # Update
#            self._niter_since_last = 0
#            self._distortion = q.distortion
#            self._value = q.value
#        else:
#            self._niter_since_last += 1

#        self._niter += 1

#        if hasattr(self, '_niter_success'):
#            # Stop flag only used when using basin-hopping
#            stop_flag = self._niter_since_last >= self._niter_success
#            if verbose and stop_flag:
#                msg = "Early termination: could not improve for "
#                msg += "{0:d} consecutive tries".format(
#                    self._niter_since_last)
#                print(msg)
#            return stop_flag

#    def make_unconstrained(self, x):
#        x.flags.writeable = False
#        key = sha1(x).hexdigest()
#        if key not in self._cache:
#            quant = _Unconstrained(self, x)
#            self._cache[key] = quant
#        return self._cache.get(key)

#    def make(self, value):
#        return _Quantizer(self, value)

#    @staticmethod
#    def make_stub(value, probability):
#        """
#        Returns a quantizer-like object which may be used
#            as a previous quantizer e.g.
#            for initializing recursions and testing
#        """
#        msg = ''
#        if np.any(~np.isfinite(value)):
#            msg += 'Invalid value: NaNs or infinite\n'
#        if np.any(value <= 0.):
#            msg += 'Invalid value: must be strictly positive\n'
#        if probability.size != value.size:
#            msg += 'Invalid probabilities: size mismatch\n'
#        if np.abs(np.sum(probability)-1.) > 1e-12:
#            msg += 'Invalid probabilities: must sum to one\n'
#        if np.any(probability < 0.):
#            msg += 'Invalid probabilities: must be positive\n'

#        if len(msg)>0:
#            raise ValueError(msg)
#        else:
#            fact = namedtuple('_Quantizer',
#                              ['value', 'probability', 'size', 'shape'])
#            return fact(value=value, probability=probability,
#                        size=value.size, shape=value.shape)

#    # Utilities

#    @lazy_property
#    def min_variance(self):
#        out = self.model.get_lowest_one_step_variance(
#            np.amin(self.previous.value))
#        return out

#    @lazy_property
#    def singularities(self):
#        return np.sort(
#            self.model.get_lowest_one_step_variance(
#                self.previous.value))

# class _Quantizer_Old():

#    def __init__(self, parent, value):
#        if not isinstance(parent, _QuantizerFactory):
#            msg = 'Quantizers must be instantiated from valid factory'
#            raise ValueError(msg)
#        else:
#            self.parent = parent
#            self.model = parent.model
#            self.size = parent.size
#            self.shape = parent.shape
#            self.previous = parent.previous

#        value = np.asarray(value)
#        if value.shape != self.shape:
#            msg = 'Unexpected quantizer shape'
#            raise ValueError(msg)
#        elif (np.any(np.isnan(value)) or np.any(~np.isfinite(value))):
#            msg = 'Invalid quantizer (NaN or inf)'
#            raise ValueError(msg)
#        elif np.any(np.diff(value) <= 0.):
#            msg = ('Unexpected quantizer value(s): '
#                   'must be strictly increasing')
#            raise ValueError(msg)
#        else:
#            self.value = value

#    # Results
#    @lazy_property
#    def voronoi(self):
#        """
#        Returns a self.size+1 np.array representing the 1D voronoi tiles
#        """
#        return voronoi_1d(self.value, lb=0.)

#    @property
#    def probability(self):
#        """
#        Returns a self.size np.array containing the time-t probability
#        """
#        return self.previous.probability.dot(self.transition_probability)

#    @property
#    def transition_probability(self):
#        """
#        Returns a self.previous.size-by-self.size np.array
#        containing the transition probabilities from time t-1 to time t
#        """
#        return self._integral[0]

#    @property
#    def distortion(self):
#        """
#        Returns a (scalar) distortion function to be optimized
#        """
#        elements = np.sum(self._distortion_elements, axis=1)
#        return self.previous.probability.dot(elements)

#    @property
#    def gradient(self):
#        """
#        Returns self.size np.array representing the gradient
#        of self.distortion
#        """
#        out = -2. * self.previous.probability.dot(
#            self._integral[1]
#            - self._integral[0]
#            * self.value[np.newaxis, :])
#        return out

#    @property
#    def hessian(self):
#        """
#        Returns a self.size-by-self.size np.array containing the
#        Hessian of self.distortion
#        """
#        with np.errstate(invalid='ignore'):
#            # (minus, plus)
#            terms = ((self.value - self.voronoi[:-1])*self._delta[:, :-1],
#                     (self.voronoi[1:] - self.value)*self._delta[:, 1:])
#            # Catch inf*zero indetermination at voronoi=np.inf...
#            terms[1][:, -1] = 0.

#        # Build Hessian
#        hess = np.zeros((self.size, self.size))
#        main_diagonal = -2. * self.previous.probability.dot(
#            terms[0] + terms[1] - self._integral[0])
#        diag_view(hess, k=0)[:] = main_diagonal
#        if self.size > 1:
#            off_diagonal = -2. * self.previous.probability.dot(
#                terms[0][:, 1:])
#            diag_view(hess, k=1)[:] = off_diagonal
#            diag_view(hess, k=-1)[:] = off_diagonal
#        return hess

#    #############
#    # Internals #
#    #############

#    @property
#    def _distortion_elements(self):
#        """
#        Returns the un-weighted (self.previous.size-by-self.size)
#        elements of the distortion to be summed, namely
#        ```
#            (\\mathcal{I}_{2,t}^{(ij)} - 2\\mathcal{I}_{1,t}^{(ij)} h_t^{(j)}
#            + \\mathcal{I}_{0,t}^{(ij)} (h_t^{(j)})^2
#        ```
#        Rem. Output only has valid entries, ie free of NaNs
#        """
#        return (self._integral[2]
#                - 2.*self._integral[1]*self.value[np.newaxis, :]
#                + self._integral[0]*self.value[np.newaxis, :]**2.)

#    @lazy_property
#    def _integral(self):
#        """
#        Returns of len-3 list indexed by order and filled with
#        self.previous.size-by-self.size arrays containing
#        ```
#            \\mathcal{I}_{order,t}^{(ij)}
#        ```
#        for all i,j.
#        Rem. Output only has valid entries, ie free of NaNs
#        BOTTLENECK
#        """
#        roots = self._roots
#        previous_value = self.previous.value[:, np.newaxis]
#        cdf = self._cdf
#        pdf = self._pdf

#        def _get_integral(order):
#            integral = [self.model.one_step_expectation_until(
#                previous_value, roots[right], order=order,
#                _pdf=pdf[right], _cdf=cdf[right])
#                for right in [False, True]]
#            out = integral[True] - integral[False]
#            out[self._no_roots] = 0.0
#            return np.diff(out, n=1, axis=1)
#        return [_get_integral(order) for order in [0, 1, 2]]

#    @lazy_property
#    def _delta(self):
#        """
#        Returns a previous.size-by-size+1 array containing
#        ```
#            \\delta_{t}^{(ij\\pm)}
#        ```
#        for all i,j.
#        Rem. Output only has valid entries, ie free of NaNs, but may be
#            infinite (highly unlikely)
#        In particular,
#            (1) When no root exists (eg. voronoi==0), delta is zero
#            (2) When voronoi==+np.inf, delta is zero
#            (3) When voronoi is at a root singularity, delta is np.inf
#        """
#        unsigned_root_derivative = \
#            self.model.one_step_roots_unsigned_derivative(
#                self.previous.value[:, np.newaxis],
#                self.voronoi[np.newaxis, :])
#        limit_index = unsigned_root_derivative == np.inf
#        if np.any(limit_index):
#            warnings.warn(
#                "Voronoi tiles at singularity detected",
#                UserWarning
#            )
#        out = (0.5*unsigned_root_derivative
#               * (self._pdf[True]+self._pdf[False]))
#        # Limit cases
#        # (1)  When a root does not exist, delta is 0
#        out[self._no_roots] = 0.
#        # (2) When voronoi is +np.inf (which always occur
#        #   at the last column), delta is zero.
#        out[:, -1] = 0.
#        # (3) When roots approach the singularity, delta is np.inf
#        out[limit_index] = np.inf
#        return out

#    @lazy_property
#    def _pdf(self):
#        """
#        Returns a len-2 tuples containing (left, right) PDF of roots
#        """
#        roots = self._roots
#        with np.errstate(invalid='ignore'):
#            return [norm.pdf(roots[right]) for right in [False, True]]

#    @lazy_property
#    def _cdf(self):
#        """
#        Returns a len-2 tuples containing (left, right) CDF roots
#        """
#        roots = self._roots
#        with np.errstate(invalid='ignore'):
#            return [norm.cdf(roots[right]) for right in [False, True]]

#    @lazy_property
#    def _roots(self):
#        """
#        Returns a len-2 tuples containing (left, right) roots
#        """
#        return self.model.one_step_roots(
#            self.previous.value[:, np.newaxis],
#            self.voronoi[np.newaxis, :])

#    @lazy_property
#    def _no_roots(self):
#        return np.isnan(self._roots[0])
