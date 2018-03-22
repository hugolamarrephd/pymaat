import math

import numpy as np
from scipy.stats import norm

from pymaat.nputil import printoptions
from pymaat.util import lazy_property

import pymaat.quantutil
import pymaat.garch.varquant
import pymaat.garch.format

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar
import pymaat.plotutil
from pymaat.mathutil import round_to_int

class Core(pymaat.quantutil.AbstractCore):


    def __init__(self, model, first_variance,
                 price_size=100, variance_size=100,
                 first_price=1., first_probability=1.,
                 nper=None, freq='daily'):
        super().__init__(model)
        self.nper = nper
        self.variance_size = variance_size
        self.price_size = price_size
        self.first_variance = first_variance
        self.first_price = first_price
        self.first_probability = first_probability
        self.variance_formatter = \
            pymaat.garch.format.variance_formatter(freq)
        self.price_formatter = pymaat.garch.format.base_price

    def print_formatter(self, value):
        value = value.copy()
        value['price'] = self.price_formatter(value['price'])
        value['variance'] = self.variance_formatter(value['variance'])
        return value

    def plot_values_3_by_3(self):
        fig = plt.figure()

        # Compute bounds
        all_quant = self.all_quant[1:] # Disregard time 0
        all_vol = np.concatenate(
            [self.variance_formatter(q.variance).ravel() for q in all_quant])
        all_price = np.concatenate(
            [self.price_formatter(q.price) for q in all_quant])
        all_probas = np.concatenate(
            [100.*q.probability.ravel() for q in all_quant])
        m = 0
        vol_bounds = pymaat.plotutil.get_lim(
                all_vol, lb=0., margin=m)
        price_bounds = pymaat.plotutil.get_lim(
                all_price, lb=0., margin=m)
        proba_bounds = pymaat.plotutil.get_lim(
                all_probas, lb=0., ub=100., margin=m)

        # Initialize color map
        cmap = matplotlib.cm.get_cmap('gnuplot2')
        cmap = cmap.reversed()
        norm = matplotlib.colors.Normalize(vmin=proba_bounds[0],
                                           vmax=proba_bounds[1])

        step = math.floor(len(all_quant)/9)
        selected = list(range(0, len(all_quant), step))
        fd = {'fontsize':8}

        for t in range(9):
            ax = fig.add_subplot(3,3,t+1)
            quant = all_quant[selected[t]]
            # Plot tiles
            vprice = self.price_formatter(quant.voronoi_price)
            vprice[0] = price_bounds[0]
            vprice[-1] = price_bounds[1]
            for idx in range(quant.shape[0]):
                x_tiles = vprice[idx:idx+2]
                y_tiles = self.variance_formatter(
                        quant.voronoi_variance[idx])
                y_tiles[0] = vol_bounds[0]
                y_tiles[-1] = vol_bounds[1]
                x_tiles, y_tiles = np.meshgrid(x_tiles, y_tiles)
                ax.pcolor(
                    x_tiles, y_tiles,
                    100.*quant.probability[idx][:, np.newaxis],
                    cmap=cmap, norm=norm)

            #  Plot quantizer
            bprice = np.broadcast_to(quant.price[:,np.newaxis], quant.shape)
            x_pts = self.price_formatter(bprice)
            y_pts = self.variance_formatter(quant.variance)
            ax.scatter(x_pts, y_pts, c='k', s=2, marker=".")

            # Title
            ax.set_title('t={}'.format(selected[t]+1), fontdict=fd, y=0.95)

            # Y-Axis
            if t in [0,3,6]:
                lb = round_to_int(vol_bounds[0], base=5, fcn=math.floor)
                ub = round_to_int(vol_bounds[1], base=5, fcn=math.ceil)
                tickspace = round_to_int((ub-lb)/5, base=5, fcn=math.ceil)
                ticks = np.arange(lb, ub, tickspace)
                ticks = np.unique(np.append(ticks, ub))
                labels = ['{0:d}'.format(yt) for yt in ticks]
                ax.set_ylabel(r'Annual Vol. (%)', fontdict=fd)
                ax.set_ylim(vol_bounds)
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels, fontdict=fd)
            else:
                ax.yaxis.set_ticks([])

            # X-Axis
            if t in [6,7,8]:
                lb = round_to_int(price_bounds[0], base=5, fcn=math.floor)
                ub = round_to_int(price_bounds[1], base=5, fcn=math.ceil)
                tickspace = round_to_int((ub-lb)/5, base=5, fcn=math.ceil)
                ticks = np.arange(lb, ub, tickspace)
                ticks = np.unique(np.append(ticks, ub))
                labels = ['{0:d}'.format(xt) for xt in ticks]
                ax.set_xlabel(r'Price (%)', fontdict=fd)
                ax.set_xlim(price_bounds)
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels, fontdict=fd)
            else:
                ax.xaxis.set_ticks([])

        # Add colorbar
        cbar_ax = fig.add_axes([0.925, 0.1, 0.025, 0.79])
        matplotlib.colorbar.ColorbarBase(
            cbar_ax, cmap=cmap, norm=norm,
            orientation='vertical')
        cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), fontdict=fd)
        # cbar_ax.set_xlabel(r'%', fontdict=fd)
        cbar_ax.set_title('Proba. (%)', fontdict=fd)


    # Internals
    def _get_all_shapes(self):
        if self.nper is not None:
            price_size = np.broadcast_to(
                self.price_size, (self.nper,))
            variance_size = np.broadcast_to(
                self.variance_size, (self.nper,))
        else:
            price_size = np.ravel(self.price_size)
            variance_size = np.ravel(self.variance_size)

        return zip(price_size, variance_size)

    def _make_first_quant(self):
        # Set first price (i)
        price = np.ravel(self.first_price)

        # Set first variance (il)
        variance = np.atleast_2d(self.first_variance)
        shape = (price.size, variance.shape[1])
        variance = np.broadcast_to(variance, shape)

        # Set first probabilities
        probability = np.broadcast_to(self.first_probability, shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)

        return Quantization(price, variance, probability)

    def _one_step_optimize(self, shape, previous, *,
            verbose=True, fast=True):

        def optimize_price_quantizer():
            factory = _PriceFactory(self.model, shape[0], True,
                    prev_proba=previous.probability,
                    prev_variance=previous.variance,
                    prev_price=previous.price)
            optimizer = pymaat.quantutil.Optimizer(factory)
            if fast:
                result = optimizer.quick_optimize(verbose=verbose)
            else:
                result = optimizer.optimize(verbose=verbose)
            factory.clear()
            return result

        if verbose:
            print("[Marginal Price Quantizer]")
        price_quant = optimize_price_quantizer()

        def optimize_variance_quantizer_conditional_on(idx):
            # Compute root bounds
            rb = price_quant._roots[:,:,idx:idx+2]
            rb = np.moveaxis(rb, 2, 0)
            factory = pymaat.garch.varquant._Factory(
                    self.model, shape[1], True,
                    prev_proba = previous.probability,
                    prev_value = previous.variance,
                    root_bounds = rb)
            optimizer = pymaat.quantutil.Optimizer(factory)
            if fast:
                result = optimizer.quick_optimize(verbose=verbose)
            else:
                result = optimizer.optimize(verbose=verbose)
            factory.clear()
            return result

        # TODO: parallelize
        all_var_quant = [None]*shape[0]
        for idx in range(shape[0]):
            if verbose:
                msg = "[idx={0}] Conditional Variance Quantizer".format(idx)
                msg += " (price={0:.2f}%)".format(
                        self.price_formatter(price_quant.value[idx]))
                print(msg)
            all_var_quant[idx] = \
                    optimize_variance_quantizer_conditional_on(idx)

        # Merge results
        price = price_quant.value
        variance = np.empty(shape)
        transition_probability = np.empty(previous.shape + shape)
        for (idx,q) in enumerate(all_var_quant):
            variance[idx,:] = q.value
            transition_probability[:,:,idx,:] = q.transition_probability
        probability = np.einsum(
                'ij,ijkl',
                previous.probability,
                transition_probability)

        return Quantization(price, variance, probability,
                transition_probability)


class Quantization(pymaat.quantutil.AbstractQuantization):

    dtype = np.dtype([('price', np.float_), ('variance', np.float_)])

    def __init__(
            self, price, variance, probability, transition_probability=None):
        if np.any(~np.isfinite(price)):
            err_msg = 'Invalid prices: NaNs or infinite\n'
        elif np.any(price <= 0.):
            err_msg = 'Invalid prices: must be strictly positive\n'
        elif np.any(~np.isfinite(variance)):
            err_msg = 'Invalid variances: NaNs or infinite\n'
        elif np.any(variance <= 0.):
            err_msg = 'Invalid variances: must be strictly positive\n'
        elif probability.shape != variance.shape:
            err_msg = 'Invalid probabilities: size mismatch\n'
        elif np.abs(np.sum(probability)-1.) > 1e-12:
            err_msg = 'Invalid probabilities: must sum to one\n'
        elif np.any(probability <= 0.):
            err_msg = 'Invalid probabilities: must be strictly positive\n'
        if transition_probability is not None:
            if np.any(transition_probability < 0.):
                err_msg = ('Invalid transition probabilities:'
                    ' must be positive\n')
            elif np.any(
                    np.abs(np.sum(transition_probability, axis=(2,3))-1.)
                    > 1e-12
                    ):
                err_msg = ('Invalid transition probabilities:'
                    ' must sum to one\n')
        if 'err_msg' in locals():
            raise ValueError(err_msg)
        self.price = price
        self.variance = variance
        self.probability = probability
        self.transition_probability = transition_probability
        self.shape = self.variance.shape
        self.size = self.variance.size
        self.ndim = self.variance.ndim
        self.voronoi_price = pymaat.quantutil.voronoi_1d(self.price)
        self.voronoi_variance = np.empty((self.shape[0], self.shape[1]+1))
        for idx in range(self.shape[0]):
            self.voronoi_variance[idx,:] = pymaat.quantutil.voronoi_1d(
                    self.variance[idx,:])

    @property
    def value(self):
        out = np.empty(self.shape, self.dtype)
        out['price'] = self.price[:, np.newaxis]
        out['variance'] = self.variance
        return out

    def quantize(self, values):
        out = np.empty_like(values)
        price_idx = np.digitize(values['price'] , self.voronoi_price)
        out['price'] = self.price[price_idx-1]
        # TODO: QUANTIZE VARIANCE
        return out

class _PriceQuantizer(pymaat.quantutil.AbstractQuantizer1D):

    @lazy_property
    def _integral(self):
        """
        Returns of len-3 list indexed by order and filled with
        prev_size-by-size arrays containing
        ```
            \\mathcal{I}_{order,t}^{(ij)}
        ```
        for all i,j.
        Rem. Output only has valid entries, ie free of NaNs
        """
        def integrate(p):
            return np.diff(
                self.parent.model.retspec.price_integral_until(
                    self.parent.prev_price,
                    self.parent.prev_variance,
                    self._roots,
                    order=p),
                axis=2)
        return [integrate(p) for p in [0, 1, 2]]

    @lazy_property
    def _delta(self):
        """
        Returns a prev_size-by-size+1 array containing
        ```
            \\delta_{t}^{(ij\\pm)}
        ```
        for all i,j.
        Rem. Output only has valid entries (ie free of NaNs or inf)
        In particular when voronoi==+np.inf, delta is zero
        """
        # Delta(0)
        dX = self.parent.model.retspec.root_price_derivative(
            self.broad_voronoi,
            self.parent.prev_variance)
        delta0 = np.zeros_like(dX)
        delta0[...,1:] = 0.5 * norm.pdf(self._roots[...,1:]) * dX[...,1:]

        # Delta(1)
        delta1 = np.zeros_like(delta0)
        delta1[..., :-1] = delta0[..., :-1] * self.broad_voronoi[..., :-1]

        return (delta0, delta1)

    @lazy_property
    def _roots(self):
        returns = np.log(self.broad_voronoi/self.parent.prev_price)
        return self.parent.model.retspec.one_step_filter(
            returns, self.parent.prev_variance)


class _PriceFactory(pymaat.quantutil.AbstractFactory1D):

    target = _PriceQuantizer

    def __init__(self, model, size, unc_flag, *,
            prev_proba, prev_variance, prev_price):
        return_scale = np.sqrt(np.amax(prev_variance))
        min_value = np.amin(prev_price) * np.exp(-10.*return_scale)
        max_value = np.amax(prev_price) * np.exp(10.*return_scale)
        scale_value = max_value-min_value

        super().__init__(model, size, unc_flag,
                         voronoi_bounds=[0, np.inf],
                         prev_proba=prev_proba,
                         min_value=min_value,
                         scale_value=scale_value)

        self.prev_variance = prev_variance[:, :, np.newaxis]
        self.prev_price = prev_price[:, np.newaxis, np.newaxis]
