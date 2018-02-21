import pytest
import numpy as np

import pymaat.testing as pt
from pymaat.garch.spec.hngarch import HestonNandiGarch
from pymaat.garch.varquant import MarginalVariance

@pytest.fixture
def model():
    return HestonNandiGarch(
        mu=2.01,
        omega=9.75e-20,
        alpha=4.54e-6,
        beta=0.79,
        gamma=196.21)

@pytest.fixture
def variance_scale():
    return 0.18**2./252.


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
