import pytest

import pymaat.testing as pt
import pymaat.garch


class TestGarch:

    model = pymaat.garch.Garch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)

    def test_one_step_filter_at_few_values(self):
        next_var, innov = self.model.one_step_filter(0, 1)
        pt.assert_almost_equal(innov, 0.5-self.model.mu)
        pt.assert_almost_equal(next_var,
                  self.model.omega
                + self.model.beta
                + self.model.alpha * (innov - self.model.gamma) ** 2)

    def test_one_step_generate_at_few_values(self):
        next_var, ret = self.model.one_step_generate(0, 0)
        pt.assert_almost_equal(ret, 0)
        pt.assert_almost_equal(next_var, self.model.omega)
        next_var, ret = self.model.one_step_generate(0, 1)
        pt.assert_almost_equal(ret, self.model.mu-0.5)
        pt.assert_almost_equal(next_var,
                  self.model.omega
                + self.model.beta
                + self.model.alpha * self.model.gamma ** 2)

    def test_invalid_param_raise_exception(self):
        with pytest.raises(ValueError):
            pymaat.garch.Garch(
                mu=0,
                omega=0,
                alpha=0.5,
                beta=0.5,
                gamma=1)
