import math

import numpy as np
from scipy.special import ndtr

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar

import pymaat.quantutil as qutil
from pymaat.garch.format import variance_formatter
from pymaat.mathutil import round_to_int, normpdf
from pymaat.util import lazy_property
import pymaat.plotutil


class Core(qutil.AbstractCore):

    def __init__(
            self, model, first_variance,
            size=100,
            first_probability=1.,
            nper=None,
            freq='daily',
            verbose = False):
        super().__init__(verbose)
        self.model = model
        self.print_formatter = variance_formatter(freq)
        self.nper = nper
        self.size = size
        self.first_variance = first_variance
        self.first_probability = first_probability

    def _make_first_quant(self):
        # Set first values
        value = np.ravel(self.first_variance)
        # Set first probabilities
        probability = np.broadcast_to(self.first_probability, value.shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)
        return qutil.Quantization1D(
            value, probability, bounds=[0., np.inf])

    def _get_all_shapes(self):
        if self.nper is not None:
            return np.broadcast_to(self.size, (self.nper,))
        else:
            return np.ravel(self.size)

    def _one_step_optimize(self, shape, previous):
        factory = _Factory(previous.probability, self.model, previous.value)
        optimizer = qutil.Optimizer1D(factory, shape,
                robust=False,
                scale=previous.distortion,
                verbose=self.verbose,
                print_formatter=self.print_formatter)
        quant = self._try_optimization(optimizer)
        return qutil.Quantization1D(
            np.ravel(quant.value).copy(),
            np.ravel(quant.probability).copy(),
            quant.transition_probability.copy(),
            np.float_(quant.distortion),
            bounds=[0., np.inf]
        )

    def _try_optimization(self, optimizer, crop=None):
        if crop is None:
            crop = 5.
        sb = optimizer.raw_factory.get_search_bounds(crop)
        try:
            return optimizer.optimize(sb)
        except qutil.ExpandBounds:
            crop += 1.
        except qutil.ShrinkBounds:
            crop -= 1.
        if crop>0. and crop<10.:
            return self._try_optimization(optimizer, crop)
        else:
            raise RuntimeError("Failed to converge")


##################
# Plot Utilities #
##################

def plot_distortion(core):
    all_quant = core.all_quant[1:]
    y = [q.distortion for q in all_quant]
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.plot(np.arange(len(y))+1, np.array(y))

def plot_values(core):
    fig = plt.figure()
    main_ax = fig.add_axes([0.1, 0.1, 0.7, 0.85])

    # Compute bounds
    all_quant = core.all_quant[1:]
    all_vol = np.concatenate(
        [core.print_formatter(q.value) for q in all_quant])
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
        y_tiles = core.print_formatter(quant.voronoi)
        y_tiles[-1] = vol_bounds[1]
        x_tiles, y_tiles = np.meshgrid(x_tiles, y_tiles)
        main_ax.pcolor(
            x_tiles, y_tiles,
            100.*quant.probability[:, np.newaxis],
            cmap=cmap, norm=norm)

        #  Plot quantizer
        y_pts = core.print_formatter(quant.value)
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

def plot_transition_at(core, t=1):
    if t <= 0:
        raise ValueError
    fig = plt.figure()
    main_ax = fig.add_axes([0.1, 0.1, 0.7, 0.85])

    prev = core.all_quant[t-1]
    current = core.all_quant[t]

    x_pts = core.print_formatter(prev.value)[:, np.newaxis]
    y_pts = core.print_formatter(current.value)[np.newaxis, :]
    x_bounds = pymaat.plotutil.get_lim(x_pts, lb=0)
    y_bounds = pymaat.plotutil.get_lim(y_pts, lb=0)

    x_tiles = core.print_formatter(prev.voronoi)
    x_tiles[-1] = x_bounds[1]

    y_tiles = core.print_formatter(current.voronoi)
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


#############
# Internals #
#############

class _Factory:

    def __init__(self, prev_proba, model, prev_value,
                 low_z=None, high_z=None, z_proba=None):
        qutil._check_proba(prev_proba)
        self.prev_proba = prev_proba
        # Process inputs
        low_z, high_z, z_proba = self._process_innov_bounds(
            low_z, high_z, z_proba, prev_proba.shape)
        qutil._check_value(prev_value, [0., np.inf])
        qutil._check_match_shape(
            prev_value, prev_proba, low_z, high_z, z_proba)
        # Initialize object
        self._normalize_prev_proba(z_proba)
        self.model = model
        self.low_z = low_z
        self.high_z = high_z
        self.prev_value = prev_value

    def get_search_bounds(self, crop):
        cropped_low_z = np.clip(self.low_z, -crop, crop)
        cropped_high_z = np.clip(self.high_z, -crop, crop)
        lb = np.amin(self.model.get_lowest_one_step_variance(
            self.prev_value, cropped_low_z, cropped_high_z))
        ub = np.amax(self.model.get_highest_one_step_variance(
            self.prev_value, cropped_low_z, cropped_high_z))
        return (lb, ub)

    def make(self, value):
        return _Quantizer(value,
                          self.prev_proba,
                          self.model,
                          self.prev_value,
                          self.low_z,
                          self.high_z)

    def _process_innov_bounds(self, lb, ub, p, s):
        if lb is None or ub is None:
            # Default root bounds
            lb = np.full(s, -np.inf)
            ub = np.full(s, np.inf)
            p = np.ones(s)
        elif p is None:
            # Compute transition probability
            # By-pass if known for efficiency
            p = ndtr(ub) - ndtr(lb)
        # Check root bounds consistency
        if np.any(ub < lb):
            raise ValueError("Invalid root bounds (UB lower than LB)")
        else:
            return (lb, ub, p)

    def _normalize_prev_proba(self, p):
        c = np.sum(self.prev_proba*p)
        if c <= 0.:
            raise ValueError(
                'Invalid transition: state can not be reached')
        else:
            self.prev_proba /= c


class _Quantizer(qutil.AbstractQuantizer1D):

    bounds = [0., np.inf]

    def __init__(self, value, prev_proba, model, prev_value, low_z, high_z):
        super().__init__(value, prev_proba)
        self.model = model
        self.prev_value = prev_value[..., np.newaxis]
        self.low_z = low_z[..., np.newaxis]
        self.high_z = high_z[..., np.newaxis]

    @lazy_property
    def _integral(self):
        s = self.prev_value.shape[:-1] + (self.size,)
        prev_value = np.broadcast_to(self.prev_value, s)

        def get_bounds(right):
            _a = self._roots[right][..., :-1]
            _b = self._roots[right][..., 1:]
            if not right:
                _a, _b = _b, _a  # a<b from now on...
            _a = np.maximum(self.low_z, _a)
            _b = np.minimum(self.high_z, _b)
            return (_a,_b)

        (a,b) = get_bounds(False)  # Left branch bounds
        (c,d) = get_bounds(True)  # Right branch bounds

        out = [np.zeros(s),  np.zeros(s), np.zeros(s)]

        def integrate(at, idx, add):
            _at = at[idx]
            _pv = prev_value[idx]
            # Computational bottleneck: use fast variants!
            _pdf = normpdf(_at)
            _cdf = ndtr(_at)
            for _p in [0,1,2]:
                tmp = self.model._variance_integral_until(
                            _pv, _at, _p, _pdf, _cdf)
                if add:
                    out[_p][idx] += tmp
                else:
                    out[_p][idx] -= tmp

        b_gtr_a = b>a
        integrate(b, b_gtr_a, True)
        integrate(a, b_gtr_a, False)
        d_gtr_c = d>c
        integrate(d, d_gtr_c, True)
        integrate(c, d_gtr_c, False)

        return out

    @lazy_property
    def _delta(self):
        _s = self.prev_value.shape[:-1] + (self.size+1,)
        pv = np.broadcast_to(self.prev_value, _s)
        vor = np.broadcast_to(self.voronoi, _s)
        def get_delta0(right):
            out = np.zeros(_s)
            idx = np.logical_and(  # delta null on boundary...
                self._roots[right] > self.low_z,  # strictly "inside"
                self._roots[right] < self.high_z
            )
            pdf_tmp = normpdf(self._roots[right][idx])
            dX_tmp = self.model.real_roots_unsigned_derivative(
                pv[idx], vor[idx])

            out[idx] = 0.5 * pdf_tmp * dX_tmp
            return out
        delta0 = get_delta0(False) + get_delta0(True)
        delta1 = np.empty_like(delta0)
        delta1[..., :-1] = delta0[..., :-1] * self.voronoi[..., :-1]
        delta1[..., -1] = 0.  # is zero when voronoi is +np.inf (last column)
        return [delta0, delta1]

    @lazy_property
    def _roots(self):
        return self.model.real_roots(self.prev_value, self.voronoi)
