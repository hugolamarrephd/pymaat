import warnings
from collections import namedtuple
import math

import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar

import pymaat.quantutil
from pymaat.garch.format import variance_formatter
from pymaat.mathutil import round_to_int
from pymaat.util import lazy_property
from pymaat.nputil import printoptions
import pymaat.plotutil



class Core(pymaat.quantutil.AbstractCore):

    def __init__(
        self, model, first_variance, size=100, first_probability=1.,
            nper=None, freq='daily'):
        super().__init__(model)
        self.print_formatter = variance_formatter(freq)
        self.nper = nper
        self.size = size
        self.first_variance = first_variance
        self.first_probability = first_probability

    # TODO: send to superclass
    def plot_distortion(self):
        all_quant = self.all_quant[1:]
        y = [q.distortion for q in all_quant]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        ax.plot(np.arange(len(y))+1, np.array(y))

    def plot_values(self):
        fig = plt.figure()
        main_ax = fig.add_axes([0.1, 0.1, 0.7, 0.85])

        # Compute bounds
        all_quant = self.all_quant[1:]
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

        prev = self.all_quant[t-1]
        current = self.all_quant[t]

        x_pts = self.print_formatter(prev.value)[:, np.newaxis]
        y_pts = self.print_formatter(current.value)[np.newaxis, :]
        x_bounds = pymaat.plotutil.get_lim(x_pts, lb=0)
        y_bounds = pymaat.plotutil.get_lim(y_pts, lb=0)

        x_tiles = self.print_formatter(prev.voronoi)
        x_tiles[-1] = x_bounds[1]

        y_tiles = self.print_formatter(current.voronoi)
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

    def _make_first_quant(self):
        # Set first values
        value = np.ravel(self.first_variance)
        # Set first probabilities
        probability = np.broadcast_to(self.first_probability, value.shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)
        return Quantization(value, probability)

    def _get_all_shapes(self):
        if self.nper is not None:
            return np.broadcast_to(self.size, (self.nper,))
        else:
            return np.ravel(self.size)

    def _one_step_optimize(self, shape, previous, *,
            verbose=False, fast=False):
        factory = _Factory(self.model, shape, True,
                                    prev_proba=previous.probability,
                                    prev_value=previous.value)
        optimizer = pymaat.quantutil.Optimizer(factory)
        if fast:
            quant = optimizer.quick_optimize(verbose=verbose)
        else:
            quant = optimizer.optimize(verbose=verbose)
        result = Quantization.make_from(quant)
        factory.clear()
        return result


class Quantization(pymaat.quantutil.AbstractQuantization):

    def __init__(self, value, probability,
            transition_probability=None,
            distortion=None):
        # Check values
        if np.any(~np.isfinite(value)):
            err_msg = 'Invalid value: NaNs or infinite\n'
        elif np.any(value <= 0.):
            err_msg = 'Invalid value: must be strictly positive\n'

        # Check probabilities
        if probability.size != value.size:
            err_msg = 'Invalid probabilities: size mismatch\n'
        elif np.abs(np.sum(probability)-1.) > 1e-12:
            err_msg = 'Invalid probabilities: must sum to one\n'
        elif np.any(probability <= 0.):
            err_msg = 'Invalid probabilities: must be striclty positive\n'

        # Check transition probabilities
        if transition_probability is not None:
            if np.any(transition_probability < 0.):
                err_msg = ('Invalid transition probabilities:'
                    ' must be positive\n')
            elif np.any(
                    np.abs(np.sum(transition_probability, axis=1)-1.)
                    > 1e-12
                    ):
                err_msg = ('Invalid transition probabilities:'
                    ' must sum to one\n')

        if distortion is not None:
            if distortion<0.:
                err_msg = "Invalid distortion: must be positive\n"

        if 'err_msg' in locals():
            raise ValueError(err_msg)

        self.value = value
        self.shape = self.value.shape
        self.size = self.value.size
        self.ndim = self.value.ndim
        self.probability = probability
        self.transition_probability = transition_probability
        self.distortion = distortion
        self.voronoi = pymaat.quantutil.voronoi_1d(
                self.value, lb=0., ub=np.inf)

    def quantize(self, values):
        idx = np.digitize(values, self.voronoi)
        return self.value[idx-1]

    @staticmethod
    def make_from(quant):
        return Quantization(
                        quant.value.copy(),
                        quant.probability.copy(),
                        quant.transition_probability.copy(),
                        np.float_(quant.distortion)
                        )

class _Quantizer(pymaat.quantutil.AbstractQuantizer1D):

    @lazy_property
    def _integral(self):
        """
        Returns of len-3 list indexed by order and filled with
        previous-size-by-size arrays containing
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
            a = np.maximum(self.parent.root_bounds[0], a)
            b = np.minimum(self.parent.root_bounds[1], b)
            b = np.maximum(a, b)
            return (a, b)
        a, b = bound_roots(False)  # Left-branch intervals
        c, d = bound_roots(True)  # Right-branch intervals

        # Computational bottleneck here...
        bounds = [a, b, c, d]
        pdf = [norm.pdf(z) for z in bounds]
        cdf = [norm.cdf(z) for z in bounds]

        def I(p, i): return self.parent.model.variance_integral_until(
            self.parent.prev_value,
            bounds[i],
            order=p,
            _pdf=pdf[i], _cdf=cdf[i])

        def integrate(p): return (I(p, 1)-I(p, 0)) + (I(p, 3)-I(p, 2))
        return [integrate(p) for p in [0, 1, 2]]

    @lazy_property
    def _delta(self):
        """
        Returns a previous-size-by-size+1 array containing
        ```
            \\delta_{t}^{(ij\\pm)}
        ```
        for all i,j.
        Rem. Output only has valid entries (ie free of NaNs or inf)
        In particular when voronoi==+np.inf, delta is zero
        """
        # Delta(0)
        dX = self.parent.model.real_roots_unsigned_derivative(
            self.parent.prev_value,
            self.broad_voronoi)
        delta0 = np.zeros_like(dX)
        for right in [False, True]:
            idx = np.logical_and(
                self._roots[right] > self.parent.root_bounds[0],
                self._roots[right] < self.parent.root_bounds[1]
            )
            pdf_tmp = norm.pdf(self._roots[right][idx])
            dX_tmp = dX[idx]
            delta0[idx] += 0.5 * pdf_tmp * dX_tmp

        # Delta(1)
        delta1 = np.empty_like(delta0)
        delta1[..., :-1] = delta0[..., :-1] * self.broad_voronoi[..., :-1]
        delta1[..., -1] = 0.  # is zero when voronoi is +np.inf (last column)

        return (delta0, delta1)

    @lazy_property
    def _roots(self):
        return self.parent.model.real_roots(
            self.parent.prev_value,
            self.broad_voronoi
        )


class _Factory(pymaat.quantutil.AbstractFactory1D):

    target = _Quantizer

    @staticmethod
    def make_from_prev(model, size, unc_flag, previous):
        return _Factory(model, size, unc_flag,
                prev_proba = previous.probability,
                prev_value = previous.value)

    def __init__(self, model, size, unc_flag, *,
            prev_proba, prev_value, root_bounds=None):
        if root_bounds is None:
            root_bounds = np.empty((2,)+prev_proba.shape)
            root_bounds[0] = -np.inf; root_bounds[1] = np.inf
        # Conditional probabilities...
        prev_proba = prev_proba / (
            norm.cdf(root_bounds[1]) - norm.cdf(root_bounds[0])
        )

        singularities = model.get_lowest_one_step_variance(
                prev_value, root_bounds[0], root_bounds[1])
        singularities = singularities.ravel()
        singularities = np.unique(singularities)
        min_value = np.amin(singularities)
        max_value = np.amax(model.get_highest_one_step_variance(
                prev_value, root_bounds[0], root_bounds[1]))
        if np.isfinite(max_value):
            scale_value = max_value-min_value
        else:
            scale_value = min_value

        super().__init__(model, size, unc_flag,
                         voronoi_bounds=[0, np.inf],
                         prev_proba=prev_proba,
                         min_value=min_value,
                         scale_value=scale_value,
                         singularities=singularities)

        # Broadcast for convenience
        if self.prev_ndim == 1:
            self.root_bounds = root_bounds[:, :, np.newaxis]
            self.prev_value = prev_value[:, np.newaxis]
        elif self.prev_ndim == 2:
            self.root_bounds = root_bounds[:, :, :, np.newaxis]
            self.prev_value = prev_value[:, :, np.newaxis]
        else:
            raise ValueError("Unexpected previous dimension")
