from functools import partial

import numpy as np
import scipy.optimize
from scipy.stats import norm

class Garch():
    def __init__(self, mu, omega, alpha, beta, gamma):
        self.mu = mu
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if (1 - self.beta - self.alpha*self.gamma**2 <= 0):
            raise ValueError

    def timeseries_filter(self, return_ts, first_var):
        self._raise_value_error_if_invalid_variance(first_var)

        innov_ts = np.empty_like(return_ts)
        var_ts = self._init_var_ts(return_ts, first_var)

        for (t,(r,h)) in enumerate(zip(return_ts, var_ts)):
            (var_ts[t+1], innov_ts[t]) = self.one_step_filter(r,h)

        return (var_ts, innov_ts)

    def timeseries_simulate(self, innov_ts, first_var):
        self._raise_value_error_if_invalid_variance(first_var)

        return_ts = np.empty_like(innov_ts)
        var_ts = self._init_var_ts(innov_ts, first_var)

        for (t,(z,h)) in enumerate(zip(innov_ts, var_ts)):
            (var_ts[t+1], return_ts[t]) = self.one_step_simulate(z,h)

        return (var_ts, return_ts)

    def one_step_filter(self, ret, var):
        vol = np.sqrt(var)
        innov = (ret - (self.mu-0.5) * var)/vol
        next_var = ( self.omega + self.beta * var + self.alpha *
            np.power(innov - self.gamma * vol,2) )
        return (next_var, innov)

    def one_step_simulate(self, innov, var):
        vol = np.sqrt(var)
        ret = (self.mu-0.5) * var + vol * innov
        next_var = ( self.omega + self.beta * var + self.alpha *
            np.power(innov-self.gamma * vol,2) )
        return (next_var, ret)

    def one_step_roots(self, var, next_var):
        d = np.sqrt((next_var - self.omega - self.beta*var) / self.alpha)
        a = self.gamma * np.sqrt(var)
        innov_left = a-d
        innov_right = a+d
        return (innov_left, innov_right)

    def one_step_integrate_until_innov(self, innov, var):
        # Formatting input
        innov = np.asarray(innov)
        var = np.asarray(var)
        scalar_input = False
        if innov.ndim == 0 and var.ndim == 0:
            innov = innov[np.newaxis]  # Makes x 1D
            var = var[np.newaxis]  # Makes x 1D
            scalar_input = True
        # Compute factors
        cdf_factor = (self.omega + self.alpha
                + (self.beta + self.alpha * self.gamma ** 2) * var)
        pdf_factor = self.alpha * (2 * self.gamma * np.sqrt(var) - innov)
        # Treat limit cases (innov = +- infinity)
        limit_cases = np.isinf(innov)
        limit_cases = np.logical_and(limit_cases,
                np.ones_like(var, dtype=bool)) # Broadcast
        pdf_factor[limit_cases] = 0
        # Compute integral
        out = cdf_factor * norm.cdf(innov) + pdf_factor * norm.pdf(innov)
        # Formatting output
        if scalar_input:
            return np.squeeze(out)
        return out

    def one_step_has_roots(self, var, next_var):
        return next_var<=self._get_lowest_one_step_variance(var)

    def _get_lowest_one_step_variance(self, var):
        return self.omega + self.beta * var

    # TODO: Send to estimator
    # def negative_log_likelihood(self, innov, var):
    #     return 0.5 * (np.power(innov, 2) + np.log(var))

    @staticmethod
    def _raise_value_error_if_invalid_variance(var):
        if np.any(var<=0):
            raise ValueError

    @staticmethod
    def _init_var_ts(like, first_var):
        var_shape = (like.shape[0]+1,) + like.shape[1:]
        var_ts = np.empty(var_shape)
        var_ts[0] = first_var
        return var_ts

class Quantizer():
    def __init__(self, model, init_innov, first_var):
        self.model = model

        if init_innov.ndim == 2 and init_innov.shape[1]>1:
            self.n_per = init_innov.shape[0]+1
            self.n_quant = init_innov.shape[1]
        else:
            raise ValueError

        # Initialization of quantizer (incl. probabilities)
        (self.gamma,*_) = self.model.timeseries_simulate(init_innov, first_var)
        self.gamma.sort(axis=-1)
        self.proba = np.zeros_like(self.gamma)
        self.proba[:,0] = 1

    def quantize(self):
        self._one_step_quantize(self.gamma[0],self.proba[0], self.gamma[1])

    def _one_step_quantize(self, prev_gamma, prev_proba, init_gamma):
        # Optimize quantizer
        func_to_optimize = partial(self._one_step_gradient,
                prev_gamma,
                prev_proba)
        opt = scipy.optimize.root(func_to_optimize, init_gamma)
        # Compute transition probabilities
        #...
        print(opt.x)
        print(opt.success)
        print(opt.message)

    def _one_step_gradient(self, prev_gamma, prev_proba, gamma):
        print('In:')
        print(gamma)
        # Keep gamma in increasing order
        sort_id = np.argsort(gamma)
        gamma = gamma[sort_id]
        g = np.empty_like(gamma) # Initialize output
        # Warm-up for broadcasting
        gamma = gamma[np.newaxis,:]
        prev_gamma = prev_gamma[:,np.newaxis]
        # Compute integrals
        voronoi = self._get_voronoi(gamma)
        i = self._one_step_integrate(prev_gamma, voronoi)
        prev_proba = prev_proba[np.newaxis,:]
        # Compute gradient and put back in initial order
        g[sort_id] = -2 * prev_proba.dot(i).squeeze()
        print('Out:')
        print(g)
        assert(not np.any(np.isnan(g)))
        return g

    def _one_step_integrate(self, prev_gamma, voronoi):
        z = self.model.one_step_roots(prev_gamma, voronoi)
        cdf = self._get_cdf_factor(prev_gamma, voronoi)
        pdf = self._get_pdf_factor(prev_gamma, z)
        (i_left, i_right) = self._get_raw_integral(cdf, pdf, z)
        # Special bounds (voronoi[0] = 0, voronoi[-1] = +infty)
        left_pad = np.full((self.n_quant,1), np.nan)
        right_pad = np.zeros((self.n_quant,1))
        i_left = np.hstack((left_pad, i_left, right_pad))
        i_right = np.hstack((left_pad, i_right, cdf[:,-1,np.newaxis]))
        i = i_right - i_left
        # Evaluate integral over intervals (voronoi[i], voronoi[i+1])
        i[np.isnan(i)] = 0
        return np.diff(i, n=1, axis=-1)

    @staticmethod
    def _get_voronoi(gamma):
        return gamma[:,:-1] + 0.5 * np.diff(gamma, n=1, axis=-1)
