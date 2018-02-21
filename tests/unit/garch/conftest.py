import pytest

from pymaat.garch.spec.hngarch import HestonNandiGarch

ALL_MODELS = [
            HestonNandiGarch(
                mu=2.01,
                omega=9.75e-20,
                alpha=4.54e-6,
                beta=0.79,
                gamma=196.21),
             ]

ALL_MODEL_IDS = ['HN-GARCH',]

@pytest.fixture(scope='module')
def variance_scale():
    return 0.18**2/252.

@pytest.fixture(params=ALL_MODELS, ids=ALL_MODEL_IDS, scope='module')
def model(request):
    return request.param
