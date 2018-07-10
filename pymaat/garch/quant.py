import os
import numpy as np
from multiprocessing import Pool
from functools import partial

import pymaat.quantutil as qutil
import pymaat.garch.varquant
import pymaat.garch.pricequant
from pymaat.garch.format import variance_formatter, base_price

import pymaat.testing as pt
from pymaat.nputil import printoptions
import time

def plot(model, quantization, ax, cax=None, freq='daily', lim=None,
        plim=None, show_quantizer=True):
    from matplotlib.patches import Polygon
    f = [base_price, variance_formatter(freq)]
    quantization.plot2d(ax, cax, formatters=f, lim=lim, plim=plim,
            show_quantizer=show_quantizer)

class AbstractCore(qutil.AbstractCore):

    def __init__(self,
                model,
                size,
                nper=None,
                freq='daily',
                first_variance=None,
                first_price=1.,
                first_probability=1.,
                verbose=False
                ):
        self.model = model
        shapes = self._get_shapes(size, nper)
        if first_variance is None:
            first_variance = self._get_default_first_variance(freq)
        first_quant = self._make_first_quant(
            first_price, first_variance, first_probability)
        super().__init__(shapes, first_quant, verbose)
        self._variance_formatter = variance_formatter(freq)
        self._price_formatter = base_price
        # TODO used frequency to set weight
        self._weight = np.array([1e1, 1e4])

    def _get_default_first_variance(self, freq):
            if freq.lower() == 'daily':
                return (0.18**2.)/252.
            elif freq.lower() == 'weekly':
                return (0.18**2.)/52.
            elif freq.lower() == 'monthly':
                return (0.18**2.)/12.
            elif freq.lower() == 'yearly':
                return 0.18**2.
            else:
                raise ValueError("Unexpected frequency")

    def _get_shapes(self, size, nper):
        raise NotImplementedError

    def _make_first_quant(self, price, variance, probability):
        raise NotImplementedError

class AbstractVoronoiCore(AbstractCore):

    _bounds = np.array([[0.,0.],[np.inf,np.inf]])

    def __init__(self, *args, nsim=1000000, **kwargs):
        self.nsim = nsim
        super().__init__(*args, **kwargs)

    def _get_shapes(self, size, nper):
        if nper is not None:
            return np.broadcast_to(size, (nper,))
        else:
            return np.ravel(size)

    def _make_first_quant(self, price, variance, probability):
        quant = qutil.VoronoiQuantization(
                np.column_stack((price, variance)),
                self._bounds,
                self._weight
                )
        quant.set_probability(probability)
        self._first_idx = np.random.choice(
                quant.size,
                self.nsim,
                p=quant.probability
                )
        return quant

    def _initialize_hook(self):
        self._previous_idx = self._first_idx

    def _one_step_optimize(self, t, shape, previous):
        simul = self._get_simulation(t, previous, self._previous_idx)
        if simul.shape[0] <= shape:
            raise ValueError("Number of simulations must be greater"
                    "than desired quantizer size")
        else:
            starting_at = simul[-shape:,:]
            assert starting_at.shape[0] == shape
        quant = qutil.VoronoiQuantization.stochastic_optimization(
                    starting_at,
                    simul,
                    nclvq=shape,
                    nlloyd=10,
                    weight=self._weight,
                    bounds=self._bounds
                    )
        idx = quant._digitize(simul)
        # Make sure all states are observed at least once
        unobserved = np.logical_not(
                np.isin(np.arange(quant.size), idx)
                )
        if np.any(unobserved):
            raise UnobservedState
        # Set quantization properties (probabilities, distortion, etc.)
        quant.set_previous(previous)
        quant._estimate_and_set_all_probabilities(idx, previous_idx)
        quant._estimate_and_set_distortion(simul, idx)
        self._previous_idx = idx
        return quant

    def _get_simulation(self, t, previous, previous_idx):
        raise NotImplementedError

class Marginal(AbstractVoronoiCore):

    def _initialize_hook(self):
        super()._initialize_hook()
        first = self._first_quantization.quantizer[self._first_idx,:]
        self._simul = self._do_all_simulation(first, len(self._shapes))

    def _get_simulation(self, t, previous, previous_idx):
        return self._simul[t]

    @staticmethod
    def _do_all_simulation(first, until):
        p0 = first[:,0]
        h0 = first[:,1]
        nsim = first.shape[0]
        innovations = np.random.normal(size=(until, nsim))
        variances, returns = model.timeseries_generate(
                innovations, h0[np.newaxis,:])
        returns = np.vstack((np.zeros((1,nsim)),returns))
        prices = p0[np.newaxis,:]*np.exp(np.cumsum(returns, axis=0))
        # Format...
        values = []
        for (p,v) in zip(prices, variances):
            values.append(np.column_stack((p,v)))
        assert len(values)==until+1
        assert np.all(values[0][:,0]==p0)
        assert np.all(prices[0][:,1]==h0)
        return values

class Markov(AbstractVoronoiCore):

    def _get_simulation(self, t, previous, previous_idx):
        return self._do_one_step_simulation(previous, previous_idx)

    @staticmethod
    def do_one_step_simulation(quant, idx):
        assert quant.probability is not None
        nsim = idx.size
        p0 = quant.quantizer[idx,0]
        h0 = quant.quantizer[idx,1]
        innovations = np.random.normal(size=nsim)
        variances, returns = model.one_step_generate(
                innovations, h0[np.newaxis,:])
        prices = p0[np.newaxis,:] * np.exp(returns)
        return np.column_stack((prices,variances))

class AbstractGridCore(AbstractCore):

    _price_bounds = _variance_bounds = np.array([0., np.inf])

    def _get_shapes(self, size, nper)
        if nper is not None:
            price_size = np.broadcast_to(size[0], (nper,))
            variance_size = np.broadcast_to(size[1], (nper,))
        else:
            price_size = np.ravel(size[0])
            variance_size = np.ravel(size[1])
        return zip(price_size, variance_size)

class Product(AbstractGridCore):

    def _make_first_quant(self, price, variance, probability):
        quant = qutil.GridQuantization2D(
                price,
                variance,
                self._price_bounds,
                self._variance_bounds,
                self._weight
                )
        quant.set_probability(probability)
        return quant

    def _one_step_optimize(self, t, shape, previous):
        price_size = shape[0]
        variance_size = shape[1]
        # Extract arrays from previous quantization
        prev_proba = np.ravel(previous.probability)
        prev_quantizer = previous._ravel()
        prev_price = prev_quantizer[:,0]  # Assumes first dim are prices...
        prev_variance = prev_quantizer[:,1]  # and second dim are variances
        # Perform marginal price quantization
        price_quant = self._get_price_quantizer(
                price_size, prev_proba, prev_price, prev_variance)
        # Perform marginal variance quantizations
        variance_quant = self._get_variance_quantizer(
                variance_size, previous)
        # Merge results
        out = qutil.GridQuantization2D(
             price_quant.value, variance_quant.value,
             [0., np.inf], [0., np.inf], self._weight
             )
        out.set_previous(previous)
        trans = np.empty((previous.size, price_size, variance_size))
        for idx in range(price_size):
            low_z = price_quant._roots[:,idx]
            high_z = price_quant._roots[:,idx+1]
            tmp_quant = pymaat.garch.varquant.Quantizer(
                    np.ravel(variance_quant.value),
                    prev_proba,
                    self.model,
                    prev_variance,
                    low_z=low_z,
                    high_z=high_z)
            trans[:, idx, :] = tmp_quant.transition_probability
        proba = np.einsum('i,ijk', prev_proba, trans)
        # Unravel previous dimension
        trans.shape = previous.shape + (price_size, variance_size)
        # Set probabilities
        out.set_probability(proba, strict=False)
        out.set_transition_probability(trans)
        return out

    def _get_price_quantizer(
            self, shape, prev_proba, prev_price, prev_variance):
        if self.verbose:
            print("\nMarginal Price Quantizer")
        # Do actual optimization
        factory = pymaat.garch.pricequant.Factory(
            self.model, prev_proba, prev_price, prev_variance)
        optimizer = qutil.Optimizer1D(
            factory,
            shape,
            verbose=self.verbose,
            print_formatter=self._price_formatter
            )
        search_bounds = factory.get_search_bounds(10.)
        return optimizer.optimize(search_bounds)

    def _get_variance_quantizer(self, shape, previous):
        if self.verbose:
            print("\nMarginal Variance Quantizer")
        # Merge same variances
        prev_proba = np.sum(previous.probability, axis=0)
        prev_variance = previous.value2
        # Do actual optimization...
        factory = pymaat.garch.varquant.Factory(
                self.model, prev_proba, prev_variance)
        search_bounds = factory.get_search_bounds(10.)
        optimizer = qutil.Optimizer1D(
            factory,
            shape,
            verbose=self.verbose,
            print_formatter=self._variance_formatter)
        return optimizer.optimize(search_bounds)

class Conditional(AbstractGridCore):

    def __init__(self, *args, parallel=False, **kwargs):
        self.parallel = parallel
        super().__init__(*args, **kwargs)

    def _make_first_quant(self, price, variance, probability):
        quant = qutil.ConditionalQuantization2D(
                price,
                variance,
                self._price_bounds,
                self._variance_bounds,
                self._weight)
        quant.set_probability(probability)
        return quant

    def _one_step_optimize(self, t, shape, previous):
        price_size = shape[0]
        variance_size = shape[1]

        # Extract arrays from previous quantization
        prev_proba = np.ravel(previous.probability)
        prev_quantizer = previous._ravel()
        prev_price = prev_quantizer[:,0]  # Assumes first dim are prices...
        prev_variance = prev_quantizer[:,1]  # and second dim are variances

        # Perform marginal price quantization
        price_quant = self._get_price_quantizer(
                price_size, prev_proba, prev_price, prev_variance)
        roots = price_quant._roots

        # Perform all conditional variance quantizations
        #   [Embarassingly parallel]
        f = partial(self._get_variance_quantizer_at,
                variance_size,
                prev_proba,
                prev_variance,
                roots)

        if self.parallel:
            p = Pool()
            variance_quant = p.map(f, range(price_size))
            p.close()
        else:
            variance_quant = []
            for idx in range(price_size):
                variance_quant.append(f(idx))

        # Merge results
        price = np.ravel(price_quant.value)
        variance = np.empty((price_size, variance_size))
        proba = np.empty((price_size, variance_size))
        trans = np.empty((previous.size, price_size, variance_size))
        for (idx, q) in enumerate(variance_quant):
            variance[idx, :] = np.ravel(q.value)
            proba[idx, :] = q.probability
            trans[:, idx, :] = q.transition_probability
        out = qutil.ConditionalQuantization2D(
            price, variance, [0., np.inf], [0., np.inf], self._weight)
        out.set_previous(previous)
        out.set_probability(proba, strict=False)  # Allow null probability!
        trans.shape = previous.shape + proba.shape
        out.set_transition_probability(trans)

        return out

    def _get_price_quantizer(
            self, shape, prev_proba, prev_price, prev_variance):
        if self.verbose:
            print("\nMarginal Price Quantizer")
        # Do actual optimization
        factory = pymaat.garch.pricequant.Factory(
            self.model, prev_proba, prev_price, prev_variance)
        optimizer = qutil.Optimizer1D(
            factory,
            shape,
            verbose=self.verbose,
            print_formatter=self._price_formatter
            )
        search_bounds = factory.get_search_bounds(10.)
        return optimizer.optimize(search_bounds)

    def _get_variance_quantizer_at(
            self, shape, prev_proba, prev_variance, roots, idx):
        if self.verbose:
            print("\n[idx={0}] Conditional Variance Quantizer".format(idx))
        factory = pymaat.garch.varquant.Factory(
                self.model,
                prev_proba,
                prev_variance,
                roots[:,idx],
                roots[:,idx+1])
        optimizer = qutil.Optimizer1D(
            factory,
            shape,
            verbose=self.verbose,
            print_formatter=self._variance_formatter)
        search_bounds = factory.get_search_bounds(10.)
        return optimizer.optimize(search_bounds)
