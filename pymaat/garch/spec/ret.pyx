#cython: linetrace=True

import numpy as np
from scipy.stats import norm

from pymaat.garch.model import AbstractOneLagReturn

class CentralRiskPremium(AbstractOneLagReturn):

    def __init__(self, mu):
        self.mu = mu

    def _one_step_generate(self, innovations, variances, volatilities):
        """
            `r_t = (mu - 0.5)*h_{t} + sqrt(h_{t}) * z_{t}`
        """
        return (self.mu-0.5)*variances + volatilities*innovations

    def _one_step_filter(self, returns, variances, volatilities):
        return (returns-(self.mu-0.5)*variances)/volatilities

    def _root_price_derivative(self, prices, variances, volatilities):
        return (prices * volatilities)**-1.

    def _first_order_integral(
            self, prices, variances, innovations, volatilities):
        return (prices
                * np.exp(variances*self.mu)
                * norm.cdf(innovations-volatilities))

    def _second_order_integral(
            self, prices, variances, innovations, volatilities):
        return (prices*prices
                * np.exp(variances*(2.*self.mu+1.))
                * norm.cdf(innovations-2.*volatilities))
