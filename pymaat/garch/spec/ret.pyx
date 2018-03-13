#cython: linetrace=True

import numpy as np

from pymaat.garch.model import AbstractOneLagReturn
from pymaat.nputil import flat_view

class CentralRiskPremium(AbstractOneLagReturn):

    def __init__(self, mu):
        self.mu = mu

    def one_step_generate(self, innovations, variances, volatilities):
        """
            `r_t = (mu - 0.5)*h_{t} + sqrt(h_{t}) * z_{t}`
        """
        return (self.mu-0.5)*variances + volatilities*innovations

    def one_step_filter(self, returns, variances, volatilities):
        return (returns-(self.mu-0.5)*variances)/volatilities
