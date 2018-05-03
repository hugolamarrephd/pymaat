import os

import pytest
import numpy as np
import matplotlib.pyplot as plt

import pymaat.testing as pt
from pymaat.garch.spec.hngarch import HestonNandiGarch
import pymaat.garch.varquant
import pymaat.garch.quant

@pytest.mark.skip
def test_marginal_variance_quantization():
    # The user instantiates a GARCH model
    model = HestonNandiGarch(
        mu=2.01,
        omega=9.75e-20,
        alpha=4.54e-6,
        beta=0.79,
        gamma=196.21)
    # Instantiate the quantization from a model
    quantization = pymaat.garch.varquant.Core(model, 0.18**2./252.,
            size=25, nper=21)
    quantization.optimize(verbose=True, fast=True) # Do computations
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

def test_garch_quantization():
    # The user instantiates a GARCH model
    model = HestonNandiGarch(
        mu=2.01,
        omega=9.75e-20,
        alpha=4.54e-6,
        beta=0.79,
        gamma=196.21)
    # Instantiate the quantization from a model
    quantization = pymaat.garch.quant.Core(model, 0.18**2./252.,
            price_size=25, variance_size=10, nper=9)
    quantization.optimize(verbose=True, fast=False) # Do computations
    quantization.plot_values_3_by_3() # Plot result
    plt.savefig(
            os.path.expanduser('~/quant_val.eps'),
            format='eps', dpi=1000)
    assert False
