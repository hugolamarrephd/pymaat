from abc import ABC, abstractmethod

import numpy as np
import pymaat.quantutil as qutil
from pymaat.mathutil import normcdf

from pymaat.nputil import ravel, elbyel
from pymaat.util import method_decorator


class AbstractOneLagReturn(ABC):

    @method_decorator(elbyel)
    def one_step_generate(self, innovations, variances):
        volatilities = np.sqrt(variances)
        return self._one_step_generate(innovations, variances, volatilities)

    @method_decorator(elbyel)
    def one_step_filter(self, returns, variances):
        volatilities = np.sqrt(variances)
        return self._one_step_filter(returns, variances, volatilities)


    @abstractmethod
    def _one_step_generate(self, innovations, variances, volatilities):
        pass

    @abstractmethod
    def _one_step_filter(self, returns, variances, volatilities):
        pass

    # Quantization

    @method_decorator(elbyel)
    def roots(self, prev_logprice, prev_variance, logprice):
        prev_vol = np.sqrt(prev_variance)
        return self._roots(
                prev_logprice, prev_variance, logprice, prev_vol)

    def _quantized_integral(
            self, prev_logprice, prev_variance, voronoi, I, d):
        raise NotImplementedError

    def _roots(
            self, prev_logprice, prev_variance, logprice,
            prev_volatility):
        raise NotImplementedError

    ################
    # Quantization #
    ################

    @method_decorator(ravel)
    def quantized_integral(self, prev_logprice, prev_variance, voronoi):
        # Memory allocation
        shape = (prev_logprice.size, voronoi.size-1)
        I = (
                np.empty(shape),
                np.empty(shape),
                np.empty(shape)
                )
        shape = (prev_logprice.size, voronoi.size)
        d = (
                np.empty(shape),
                np.empty(shape)
                )
        # Do computations
        self._quantized_integral(
                prev_logprice, prev_variance, voronoi, I, d)
        return I, d


    @method_decorator(ravel)
    def get_quant_factory(self, prev_proba, prev_logprice, prev_variance):
        if not (
                prev_proba.size
                == prev_logprice.size
                == prev_variance.size
                ):
            raise ValueError("Previous quantization size mismatch")

        retspec = self  # used in Factory's closure

        class Factory:

            @method_decorator(ravel)
            def make(self, value):

                class Quantizer(qutil.AbstractQuantizer1D):

                    bounds = [-np.inf, np.inf]

                    def __init__(self, value):
                        super().__init__(value, prev_proba)
                        self._integral, self._delta = \
                                retspec.quantized_integral(
                                prev_logprice,
                                prev_variance,
                                self.voronoi
                                )

                    def get_roots(self):
                        return retspec.roots(
                                prev_logprice[:,np.newaxis],
                                prev_variance[:,np.newaxis],
                                self.voronoi
                                )

                return Quantizer(value)

            def get_search_bounds(self, crop):
                plb = retspec.one_step_generate(-crop, prev_variance)
                pub = retspec.one_step_generate(crop, prev_variance)
                return np.array([
                        np.amin(prev_logprice + plb),
                        np.amax(prev_logprice + pub)
                        ])

        return Factory()



class AbstractOneLagGarch(ABC):

    def __init__(self, retspec):
        self.retspec = retspec

    @abstractmethod
    def _equation(self, innovations, variances, volatilities):
        pass

    @method_decorator(elbyel)
    def one_step_bounds(self, variances, zlb=-np.inf, zub=np.inf):
        return self._one_step_bounds(variances, zlb, zub)

    def _one_step_bounds(self, variances, zlb, zub):
        raise NotImplementedError

    @method_decorator(elbyel)
    def real_roots(self, variances, next_variances):
        """
        Computes and returns innovations such that
        ```
            next_variances[i] = _equation(innovations[i], variances[i])
        ```
        when such an innovation exists.
        Otherwise, returns zero.
        Rem. Unsigned derivative may be zero at singularity
            even if theoretical value is inf
        Also returns the corresponding unsigned derivative
        return (left, right), uder
        """
        return self._real_roots(variances, next_variances)

    def _real_roots(self, variances, next_variances):
        raise NotImplementedError


    def _quantized_integral(self, prev_variances, zlb, zub, voronoi, I, d):
        """
        I[order][i,j] is the following expectation:
            ```
            E[
                ind(equation(z, prev_variances[i]) in voronoi[j,j+1])
                * ind(z in (lb[i], ub[i]))
                *_equation(z, prev_variances[i])**order
            ]
            ```
        where ind(.) is an indicator function
        d[order] is the corresponding derivative wrt bounds
        returns ((I0,I1,I2), (d0,d1))
        """
        raise NotImplementedError

    def termstruct_variance(self, first_variances, until):
        raise NotImplementedError

    def termstruct_logprice(self, first_variances, until):
        raise NotImplementedError

    ##############
    # Timeseries #
    ##############

    def timeseries_filter(self, returns, first_variances):
        self._raise_value_error_if_any_invalid_variance(first_variances)
        single_path = False
        if returns.ndim == 1:
            single_path = True
            returns = returns[:,np.newaxis]

        # Initialize outputs
        variances = self._init_like(returns, first_variances)
        innovations = np.empty_like(returns)

        # Do computations and return
        variances, innovations = self._timeseries_filter(
                returns, variances, innovations)

        if single_path:
            return np.ravel(variances), np.ravel(innovations)
        else:
            return variances, innovations

    def _timeseries_filter(self, returns, variances, innovations):
        """
        Performs filtering of returns from
            pre-allocated output arrays (variances, innovations).
        Rem. This is provided as a convenience only.
            Implementing classes are strongly encouraged to override
            (eg using cython) for efficiency.
        """
        for (t,(r,h)) in enumerate(zip(returns, variances)):
            (variances[t+1], innovations[t]) = self.one_step_filter(r,h)

    def timeseries_generate(
            self, innovations, first_variances, first_logprice=None):
        """
        When passing a first_logprice,
            this function returns (variance, logprice)
            where first row corresponds to initial values,
            i.e.\ (first_variance, first_logprice)
        """
        self._raise_value_error_if_any_invalid_variance(first_variances)
        single_path = False
        if innovations.ndim == 1:
            single_path = True
            innovations = innovations[:,np.newaxis]
        variances = self._init_like(innovations, first_variances)
        if first_logprice is None:  # Return mode!
            out = np.empty_like(innovations)
            self._timeseries_generate_returns(innovations, variances, out)
        else:
            out = self._init_like(innovations, first_logprice)
            self._timeseries_generate_logprices(innovations, variances, out)
        if single_path:
            return np.ravel(variances), np.ravel(out)
        else:
            return variances, out

    def _timeseries_generate_returns(self, innovations, variances, returns):
        """
        Generates time-series from innovations and
            pre-allocated output arrays (variances, returns).
        Rem. This is provided as a convenience only.
            Implementing classes are strongly encouraged to override
            (eg using cython) for efficiency.
        """
        for (t,(z,h)) in enumerate(zip(innovations, variances)):
            (variances[t+1], returns[t]) = self.one_step_generate(z,h)

    def _timeseries_generate_logprices(
            self, innovations, variances, logprices):
        for (t,(z,h)) in enumerate(zip(innovations, variances)):
            (variances[t+1], returns) = self.one_step_generate(z,h)
            logprices[t+1] = logprices[t] + returns

    ############
    # One-step #
    ############

    @method_decorator(elbyel)
    def one_step_filter(self, returns, variances):
        self._raise_value_error_if_any_invalid_variance(variances)
        volatilities = np.sqrt(variances)
        innovations = self.retspec._one_step_filter(
                returns, variances, volatilities)
        next_variances = self._equation(
                innovations, variances, volatilities)
        return (next_variances, innovations)

    @method_decorator(elbyel)
    def one_step_generate(self, innovations, variances):
        self._raise_value_error_if_any_invalid_variance(variances)
        volatilities = np.sqrt(variances)
        returns = self.retspec._one_step_generate(
                innovations, variances, volatilities)
        next_variances = self._equation(
                innovations, variances, volatilities)
        return (next_variances, returns)


    ################
    # Quantization #
    ################

    @method_decorator(ravel)
    def get_quant_factory(
            self, prev_proba, prev_value, zlb=None, zub=None):
        # Default bounds
        if zlb is None or zub is None:
            shape = prev_proba.shape
            zlb = np.full(shape, -np.inf)
            zub = np.full(shape, np.inf)
        # Input checks
        if not (prev_proba.size == prev_value.size == zlb.size == zub.size):
            raise ValueError( "Previous quantization size mismatch")
        if np.any(zub < zlb):
            raise ValueError( "Invalid innovation bounds")

        # Compute transition probability and normalization factor
        trans = normcdf(zub) - normcdf(zlb)
        norm = np.sum(prev_proba*trans)
        if norm <= 0.:
            raise qutil.UnobservedState

        model = self  # used in closure of factory!

        class _VarianceQuantizerFactory:

            @method_decorator(ravel)
            def make(self, value):

                class _Quantizer(qutil.AbstractQuantizer1D):

                    bounds = [0., np.inf]

                    @method_decorator(ravel)
                    def __init__(self, value):
                        super().__init__(value, prev_proba, norm=norm)
                        self._integral, self._delta = \
                                model.quantized_integral(
                                prev_value, zlb, zub,
                                self.voronoi
                                )

                return _Quantizer(value)

            def get_search_bounds(self, crop=10.):
                _zlb = np.clip(zlb, -crop, crop)
                _zub = np.clip(zub, -crop, crop)
                valid = _zlb < _zub
                if not np.any(valid):
                    if crop < 10.:  # Fail-safe
                        return self.get_search_bounds()
                    else:
                        raise ValueError("Could not get search bounds")
                hlb, hub = model.one_step_bounds(
                        prev_value[valid],
                        _zlb[valid],
                        _zub[valid]
                        )
                return np.array([np.amin(hlb), np.amax(hub)])

        return _VarianceQuantizerFactory()

    @method_decorator(ravel)
    def quantized_integral(self, prev_variances, zlb, zub, voronoi):
        shape = (prev_variances.size, voronoi.size-1)
        I = (
                np.empty(shape),
                np.empty(shape),
                np.empty(shape)
                )
        shape = (prev_variances.size, voronoi.size)
        d = (
                np.empty(shape),
                np.empty(shape)
                )
        self._quantized_integral(prev_variances, zlb, zub, voronoi, I, d)
        return I, d

    ##################
    # Option Pricing #
    ##################

    def american_option_lsm(self,
            innovations, normstrike, variance, rate, *,  put=False):
        T = innovations.shape[0]
        if put:
            payoff = lambda p, k: np.maximum(k-p, 0)
            in_the_money = lambda p, k: k>p
        else:
            payoff = lambda p, k: np.maximum(p-k, 0)
            in_the_money = lambda p, k: k<p
        variance, logforward = self.timeseries_generate(
                innovations, variance, first_logprice=0)
        spot = np.exp(logforward + (np.arange(T+1)*rate)[:,np.newaxis])
        def get_price(k):
            # Initialize cash flows
            cashflow = payoff(spot[-1], k)
            flag = np.empty_like(cashflow, bool)
            value = np.empty_like(cashflow)
            for t in range(T-1,-1,-1):  # Backward recursion
                # Currently ITM?
                itm = in_the_money(spot[t], k)
                # Discount cash flows for one step
                cashflow *= np.exp(-rate)
                if np.any(itm):
                    # Regressors
                    p = spot[t][itm]
                    h = variance[t][itm]
                    X = np.column_stack(
                            (np.ones_like(p), p, h, p**2., h**2., p*h))
                    # Do regression
                    betas, _, _, _= np.linalg.lstsq(X, cashflow[itm])
                    continuation = X.dot(betas)
                    exercise = payoff(p, k)
                    # Update cash-flow vector
                    flag[:] = False
                    flag[itm] = exercise>continuation
                    value[itm] = exercise
                    cashflow[flag] = value[flag]
            return np.mean(cashflow)

        result = np.empty_like(normstrike)
        for (n,k) in enumerate(normstrike):
            result[n] = get_price(k)
        return result

    ##################
    # Option Hedging #
    ##################

    def option_hedging_backtest(
            self, risk_neutral, innovations, *,
            first_variance=0.18**2./252,
            strike=1.,
            put=False,
            v0=None,
            get_hedge=None):
        assert np.isscalar(first_variance)
        assert np.isscalar(strike)
        assert np.isscalar(put)
        T = innovations.shape[0]
        nsim = innovations.shape[1]
        variance, logprice = self.timeseries_generate(
                innovations, first_variance, first_logprice=0)
        price = np.exp(logprice)
        # Sale price
        c0 = risk_neutral.option_price(
                price[0][0], strike, variance[0][0], T=T, put=put)
        # Initialize portfolio
        v = np.empty((nsim,))
        if v0 is None:
            v0 = c0
        v[:] = v0
        # Warm-up hedging function
        if get_hedge is None:
            get_hedge = risk_neutral.delta_hedge
        # Do backtest
        delta = np.empty_like(innovations)
        for t in range(T):
            print(t)
            delta[t] = get_hedge(
                    price[t],
                    strike*np.ones((nsim,)),
                    variance[t],
                    v,
                    tau=T-t,
                    put=put,
                    )
            v += delta[t] * (price[t+1] - price[t])
        liability = np.maximum(price[-1]-strike,0.)
        if put:
            liability += -price[-1] + strike # Put-call parity
        print(c0)
        print(v0)
        pnl = c0 + (v-v0) - liability

        # import matplotlib.pyplot as plt
        # plt.scatter(np.ravel(price[0:-1]), np.ravel(delta))
        # plt.show()
        # plt.scatter(price[-1], pnl)
        # plt.show()

        return pnl, delta, price[-1]


    #############
    # Utilities #
    #############

    @staticmethod
    def _raise_value_error_if_any_invalid_variance(var):
        if not np.all(np.isfinite(var)) or np.any(var<=0):
            raise ValueError("Invalid variance detected.")

    @staticmethod
    def _init_like(like, first):
        shape = (like.shape[0]+1,) + like.shape[1:]
        out = np.empty(shape)
        out[0] = first  # may broadcast here
        return out
