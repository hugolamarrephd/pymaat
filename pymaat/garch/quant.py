import os
from multiprocessing import Pool
from functools import partial, wraps
import warnings

import numpy as np
from zignor import randn

from pymaat.garch.spec.hngarch import HestonNandiGarch
import pymaat.quantutil as qutil
import pymaat.testing as pt
from pymaat.nputil import printoptions, ravel
from pymaat.util import method_decorator
from pymaat.garch.format import variance_formatter, log_price_formatter

class GarchCoreMixin:

    _logprice_bounds = np.array([-np.inf, np.inf])
    _variance_bounds = np.array([0., np.inf])
    _bounds = np.array([[-np.inf, 0.], [np.inf, np.inf]])

    def __init__(self, *,
                model=None,
                first_variance=None,
                first_logprice=0.,
                first_probability=1.,
                freq='daily',
                **kwargs
                ):
        if model is None:
            model = HestonNandiGarch(
                        mu=2.04,
                        omega=3.28e-7,
                        alpha=4.5e-6,
                        beta=0.8,
                        gamma=190
                        )
        if first_variance is None:
            first_variance = self._get_default_first_variance(freq)
        # Set properties
        self.model = model
        self.first_logprice = first_logprice
        self.first_variance = first_variance
        self.first_probability = first_probability
        # Some internals (for plotting)
        self._variance_formatter = variance_formatter(freq)
        self._logprice_formatter = log_price_formatter
        super().__init__(**kwargs)

    def _get_default_first_variance(self, freq):
        if freq.lower() == 'daily':
            return (0.18**2.)/252.
        elif freq.lower() == 'weekly':
            return (0.18**2.)/52.
        elif freq.lower() == 'monthly':
            return (0.18**2.)/12.
        elif freq.lower() == 'yearly':
            return (0.18**2.)/1.
        else:
            raise ValueError("Unexpected frequency")

    def get_weights(self):
        h0 = np.dot(self.first_variance, self.first_probability)
        _, varlp = self.model.termstruct_logprice(h0, self.nper)
        _, varh = self.model.termstruct_variance(h0, self.nper)
        weights = np.column_stack((varlp, varh[1:]))
        weights = np.sqrt(weights)
        weights **= -1.
        return weights

    def plot_at(self, t, ax, colorbar_ax=None, **kwargs):
        self.quantizations[t].plot2d(
                ax, colorbar_ax, **kwargs,
                formatter1 = self._logprice_formatter,
                formatter2 = self._variance_formatter)


    @method_decorator(ravel)
    def european_option(self, normstrike, *, T=None, put=False):
        if T is None:
            T = self.nper
        if not np.isscalar(T):
            raise ValueError('Time-to-expiry must be scalar')
        if T > self.nper:
            raise ValueError('Time-to-expiry longer than quantization')

        if put:
            payoff = lambda logp, k: np.maximum(k-np.exp(logp), 0)
        else:
            payoff = lambda logp, k: np.maximum(np.exp(logp)-k, 0)
        all_quantizations = self.quantizations[:T]

        def get_price(k):
            quant = all_quantizations[-1]  # Use last only!
            return np.sum(
                    quant.probability
                    * payoff(quant.quantizer[...,0], k)
                    )

        result = np.empty_like(normstrike)
        for (n,k) in enumerate(normstrike):
            result[n] = get_price(k)

        return result

    @method_decorator(ravel)
    def american_option(self, normstrike, rate, *, T=None, put=False):
        if T is None:
            T = self.nper
        if not np.isscalar(T):
            raise ValueError('Time-to-expiry must be scalar')
        if T > self.nper:
            raise ValueError('Time-to-expiry longer than quantization')

        discount_factor = np.exp(-rate)

        if put:
            payoff = lambda logp, k: np.maximum(k-np.exp(logp), 0)
        else:
            payoff = lambda logp, k: np.maximum(np.exp(logp)-k, 0)
        all_quantizations = self.quantizations[:T]


        def get_price(k):
            # Initialization
            quant = all_quantizations[-1]
            spotlogprice = quant.quantizer[...,0] + T*rate
            value = payoff(spotlogprice, k)
            # Backward recursion
            for (tau, quant) in enumerate(reversed(all_quantizations)):
                if quant.transition_probability is None:
                    raise ValueError(
                            'Transition probabilities must be set '
                            'prior to computing option prices'
                    )
                continuation = np.tensordot(
                        quant.transition_probability,
                        value,
                        axes=(
                            tuple(range(-value.ndim, 0)), # Last ndim dimensions
                            tuple(range(0, value.ndim))  # First ndim (all) dimensions
                            )
                        )
                continuation *= discount_factor
                logprice = quant.previous.quantizer[...,0]
                spotlogprice = logprice + (T-1-tau)*rate
                exercise = payoff(spotlogprice, k)
                value = np.maximum(continuation, exercise)
            return value

        result = np.empty_like(normstrike)
        for (n,k) in enumerate(normstrike):
            result[n] = get_price(k)

        return result

    def optimal_hedging_surface(self, strike, T=21, put=False):
        assert np.isscalar(strike)
        if put:
            payoff = lambda lp: np.maximum(strike-np.exp(lp), 0)
        else:
            payoff = lambda lp: np.maximum(np.exp(lp)-strike, 0)
        all_quantizations = self.quantizations[:T]
        # Initialization
        quant = all_quantizations[-1]
        logprice = quant.quantizer[...,0]
        c = payoff(logprice)
        gamma = np.ones_like(c)
        # Backward recursion
        surface = []
        for quant in reversed(all_quantizations):
            if quant.transition_probability is None:
                raise ValueError(
                        'Transition probabilities must be set '
                        'prior to computing option prices'
                )
            current = quant.quantizer[...,0]
            previous = quant.previous.quantizer[...,0]
            price_changes = np.exp(
                    np.reshape(current, (1,)*previous.ndim + current.shape)
                    - np.reshape(previous, previous.shape + (1,)*current.ndim)
                    ) - 1.
            current_dims = tuple(range(-current.ndim, 0))
            previous_dims = tuple(range(0, previous.ndim))
            tensor_axes = (current_dims, previous_dims)
            m0 = np.tensordot(
                    quant.transition_probability,
                    gamma,
                    axes=tensor_axes
                    )
            m1 = np.tensordot(
                    quant.transition_probability*price_changes,
                    gamma,
                    axes=tensor_axes
                    )
            m2 = np.tensordot(
                    quant.transition_probability*price_changes**2.,
                    gamma,
                    axes=tensor_axes
                    )
            q0 = np.tensordot(
                    quant.transition_probability,
                    c * gamma,
                    axes=tensor_axes
                    )
            q1 = np.tensordot(
                    quant.transition_probability*price_changes,
                    c * gamma,
                    axes=tensor_axes
                    )
            # Preparing next step...
            gamma = m0 - m1**2./m2  # Must be updated first!
            c = (q0 - m1*q1/m2)/gamma
            # Registering result in surface
            surface.append(
                {'quant':quant.previous, 'm1':m1, 'm2':m2, 'q1':q1})
        def get_hedge(prices, strikes, variances, portfolio, tau=T, put=None):
            assert np.all(strikes==strike)
            # `surface` is in closure
            if not (portfolio.size == prices.size == variances.size):
                raise ValueError("Size mismatch")
            # Extract surface and associated quantization
            s = surface[tau-1]
            quant = s['quant']
            # Quantize
            logprices = np.log(prices)
            values = np.column_stack((logprices,variances))
            idx = quant._digitize(values)
            m1 = s['m1'][idx]
            m2 = s['m2'][idx]
            q1 = s['q1'][idx]
            price = np.exp(quant.quantizer[...,0])[idx]
            # Compute deltas
            return (q1-portfolio*m1)/(m2*price)
        return c, get_hedge


    def simulate(self, nsim):
        first = self.first_quantization.simulate(nsim)
        first_logprice = first[:,0]
        first_variance = first[:,1]
        innovations = randn(int(self.nper), int(nsim))  # Fast normal
        variances, logprices = self.model.timeseries_generate(
                innovations, first_variance, first_logprice)
        # Formatting
        simul = []
        for (lp,v) in zip(logprices, variances):
            simul.append(np.column_stack((lp,v)))
        return simul[0], simul[1:]

    def one_step_simulate(self, previous, nsim):
        prev = previous.simulate(nsim)
        prev_logprice = prev[:,0]
        prev_variance = prev[:,1]
        innovations = randn(int(nsim))
        variances, returns = self.model.one_step_generate(
                innovations, prev_variance)
        logprices = prev_logprice + returns
        # Formatting
        prev_simul = np.column_stack((prev_logprice,prev_variance))
        simul = np.column_stack((logprices,variances))
        return prev_simul, simul

class GarchVoronoiCoreMixin(GarchCoreMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_first_quantization(self):
        quantizer = np.column_stack(
                (self.first_logprice, self.first_variance))
        weight = np.ones((2,))
        quant = qutil.VoronoiQuantization(quantizer, self._bounds, weight)
        quant.set_probability(self.first_probability)
        return quant

class Marginal(
        GarchVoronoiCoreMixin,
        qutil.AbstractMarginalVoronoiCore,
        ):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Markov(
        GarchVoronoiCoreMixin,
        qutil.AbstractMarkovianVoronoiCore,
        ):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class GarchGridCoreMixin(GarchCoreMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _optimize_logprice_quantizer(
            self, weight, shape, prev_proba, prev_logprice, prev_variance):
        if self.verbose:
            print("\n Optimizing Price Quantizer...")
        # Do actual optimization
        factory = self.model.retspec.get_quant_factory(
            prev_proba, prev_logprice, prev_variance)
        optimizer = qutil.Optimizer1D(
            factory,
            shape,
            verbose=self.verbose,
            weight=weight[0],
            formatter=self._logprice_formatter
            )
        search_bounds = factory.get_search_bounds(10.)
        return optimizer.optimize(search_bounds)

class Product(
        GarchGridCoreMixin,
        qutil.AbstractMarkovianCore
        ):

    def get_first_quantization(self):
        quant = qutil.GridQuantization2D(
                self.first_logprice,
                self.first_variance,
                self._logprice_bounds,
                self._variance_bounds,
                )
        quant.set_probability(self.first_probability)
        return quant

    def one_step_optimize(self, shape, previous, weight):
        price_size = shape[0]
        variance_size = shape[1]
        # Extract arrays from previous quantization
        prev_proba = np.ravel(previous.probability)
        prev_quantizer = previous._ravel()
        prev_logprice = prev_quantizer[:,0]  # Assumes first dim are prices...
        prev_variance = prev_quantizer[:,1]  # and second dim are variances
        # Perform marginal price quantization
        logprice_quant = self._optimize_logprice_quantizer(
                weight, price_size, prev_proba, prev_logprice, prev_variance)
        roots = logprice_quant.get_roots()[0]
        # Perform marginal variance quantizations
        variance_quant = self._optimize_variance_quantizer(
                weight, variance_size, previous)
        # Merge results
        out = qutil.GridQuantization2D(
             logprice_quant.value,
             variance_quant.value,
             self._logprice_bounds,
             self._variance_bounds,
             weight
             )
        # TODO: use sparse matrix here
        trans = np.empty((previous.size, price_size, variance_size))
        for idx in range(price_size):
            f = self.model.get_quant_factory(
                    prev_proba,
                    prev_variance,
                    roots[:,idx],
                    roots[:,idx+1]
                    )
            tmp = f.make(variance_quant.value)
            trans[:, idx, :] = tmp.transition_probability
        proba = np.einsum('i,ijk', prev_proba, trans)
        # Unravel previous dimension
        trans.shape = previous.shape + (price_size, variance_size)
        # Set quantization
        out.set_previous(previous)
        out.set_probability(proba, strict=False)
        out.set_transition_probability(trans)
        return out

    def _optimize_variance_quantizer(self, weight, shape, previous):
        if self.verbose:
            print("\aVariance Quantizer")
        # Merge same variances
        prev_proba = np.sum(previous.probability, axis=0)
        prev_variance = previous.value2
        # Do actual optimization...
        factory = self.model.get_quant_factory(prev_proba, prev_variance)
        search_bounds = factory.get_search_bounds(10.)
        optimizer = qutil.Optimizer1D(
            factory,
            shape,
            weight=weight[1],
            verbose=self.verbose,
            formatter=self._variance_formatter
            )
        return optimizer.optimize(search_bounds)

class Conditional(
        GarchGridCoreMixin,
        qutil.AbstractMarkovianCore
        ):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_first_quantization(self):
        quant = qutil.ConditionalQuantization2D(
                self.first_logprice,
                self.first_variance,
                self._logprice_bounds,
                self._variance_bounds,
                )
        quant.set_probability(self.first_probability)
        return quant

    def one_step_optimize(self, shape, previous, weight):
        price_size = shape[0]
        variance_size = shape[1]

        if previous.size == 1:
            # Catch degenerated case
            price_size = price_size * variance_size
            variance_size = 1

        # Extract arrays from previous quantization
        prev_proba = np.ravel(previous.probability)
        prev_quantizer = previous._ravel()
        prev_logprice = prev_quantizer[:,0]  # Assumes first dim are prices...
        prev_variance = prev_quantizer[:,1]  # and second dim are variances

        # Perform marginal price quantization
        logprice_quant = self._optimize_logprice_quantizer(
                weight, price_size, prev_proba, prev_logprice, prev_variance)
        roots = logprice_quant.get_roots()[0]

        # Perform all conditional variance quantizations
        #   [Embarassingly parallel]
        f = partial(self._optimize_variance_quantizer_at,
                weight,
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
        price = np.ravel(logprice_quant.value)
        variance = np.empty((price_size, variance_size))
        proba = np.empty((price_size, variance_size))
        #TODO: use sparse matrix here
        trans = np.empty((previous.size, price_size, variance_size))
        for (idx, tmp) in enumerate(variance_quant):
            variance[idx, :] = np.ravel(tmp.value)
            proba[idx, :] = tmp.probability
            trans[:, idx, :] = tmp.transition_probability
        trans.shape = previous.shape + proba.shape

        # Build Quantization
        out = qutil.ConditionalQuantization2D(
            price,
            variance,
            self._logprice_bounds,
            self._variance_bounds,
            weight
            )
        out.set_previous(previous)
        out.set_probability(proba, strict=False)
        out.set_transition_probability(trans)
        return out

    def _optimize_variance_quantizer_at(
            self, weight, shape, prev_proba, prev_variance, roots, idx):
        if self.verbose:
            print("\n[idx={0}] Conditional Variance Quantizer".format(idx))
        factory = self.model.get_quant_factory(
                prev_proba,
                prev_variance,
                roots[:,idx],
                roots[:,idx+1])
        optimizer = qutil.Optimizer1D(
            factory,
            shape,
            weight=weight[1],
            verbose=self.verbose,
            formatter=self._variance_formatter
            )
        search_bounds = factory.get_search_bounds(10.)
        return optimizer.optimize(search_bounds)
