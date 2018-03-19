import os

import pytest
import numpy as np
import matplotlib.pyplot as plt

import pymaat.testing as pt
from pymaat.garch.spec.hngarch import HestonNandiGarch
from pymaat.garch.varquant import MarginalVariance

def test_marginal_variance_quantization():
    # The user instantiates a GARCH model
    model = HestonNandiGarch(
        mu=2.01,
        omega=9.75e-20,
        alpha=4.54e-6,
        beta=0.79,
        gamma=196.21)
    # Instantiate the quantization from a model
    quantization = MarginalVariance(model, 0.18**2./252.,
            size=25, nper=21)
    quantization.optimize(verbose=True) # Do computations
    # Visualize the quantization
    quantization.plot_distortion() # Plot result
    plt.savefig(
            os.path.expanduser('~/marg_var_dist.eps'),
            format='eps', dpi=1000)
    quantization.plot_values() # Plot result
    plt.savefig(
            os.path.expanduser('~/marg_var_val.eps'),
            format='eps', dpi=1000)
    quantization.plot_transition_at(t=15)
    plt.savefig(
            os.path.expanduser('~/marg_var_trans.eps'),
            format='eps', dpi=1000)
