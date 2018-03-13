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
    size = np.concatenate([
            10*np.ones((21,), dtype=np.int_),
            ])
    quantization = MarginalVariance(model, 0.18**2./252., size=size)
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
    # fig.set_size_inches(4, 3)
    # quantization.visualize_transition_at(t=15)


# class TestMarginalVariance:

#     @pytest.fixture(scope='class')
#     def shape(self):
#         return (10,1)

#     @pytest.fixture(scope='class')
#     def nper(self):
#         return 21

#     @pytest.fixture(scope='class')
#     def quantization(self, model, variance_scale, shape, nper):
#         return MarginalVariance(model, variance_scale, shape[0], nper=nper)

#     @pytest.fixture(scope='class')
#     def quantizers(self, quantization):
#         quantization.optimize()
#         return quantization.all_quantizers[1:]

#     def test_distortion(self, quantizers):
#         for q in quantizers:
#             print(q.distortion)
# pytest.main(__file__)
        # print(
        #         np.sqrt(
        #             np.concatenate(
        #                 last1[:,np.newaixs],last2[:,np.newaxis],axis=1
        #                 )
        #             *252.)
        #         )
