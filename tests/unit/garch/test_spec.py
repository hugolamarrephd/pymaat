import pytest

import pymaat.testing as pt
from pymaat.garch.spec.hngarch import HestonNandiGarch


class TestHestonNandiGarch:

    @pytest.fixture
    def model(self):
        return HestonNandiGarch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)

    def test_one_step_filter_at_few_values(self, model):
        next_var, innov = model.one_step_filter(0, 1)
        pt.assert_almost_equal(innov, 0.5-model.retspec.mu)
        pt.assert_almost_equal(next_var,
                               model.omega
                               + model.beta
                               + model.alpha * (innov - model.gamma) ** 2)

    def test_one_step_generate_at_few_values(self, model):
        next_var, ret = model.one_step_generate(0, 0)
        pt.assert_almost_equal(ret, 0)
        pt.assert_almost_equal(next_var, model.omega)
        next_var, ret = model.one_step_generate(0, 1)
        pt.assert_almost_equal(ret, model.retspec.mu-0.5)
        pt.assert_almost_equal(next_var,
                               model.omega
                               + model.beta
                               + model.alpha * model.gamma ** 2)

    def test_invalid_param_raise_exception(self):
        with pytest.raises(ValueError):
            HestonNandiGarch(
                mu=0,
                omega=0,
                alpha=0.5,
                beta=0.5,
                gamma=1)
