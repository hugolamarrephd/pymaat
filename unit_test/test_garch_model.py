from functools import partial

import pytest
import numpy as np
from scipy.stats import norm

import pymaat.testing as pt
import pymaat.garch

ALL_MODELS = [
            pymaat.garch.Garch(
                mu=2.01,
                omega=9.75e-20,
                alpha=4.54e-6,
                beta=0.79,
                gamma=196.21),
             ]

ALL_MODEL_IDS = ['HN-GARCH',]

ALL_ONE_STEPS = [
                'one_step_filter',
                'one_step_generate',
                'one_step_has_roots',
                'one_step_roots',
                'one_step_root_from_return',
                'one_step_roots_unsigned_derivative',
                'one_step_expectation_until'
                ]

VAR_LEVEL = 0.18**2./252.
VOL_LEVEL = np.sqrt(VAR_LEVEL)
RET_EXP = 0.06/252.

@pytest.fixture(params=ALL_MODELS, ids=ALL_MODEL_IDS, scope='module')
def model(request):
    return request.param

@pytest.fixture(params=ALL_ONE_STEPS, scope='module')
def one_step_str(request):
    return request.param

class TestTimeseries():

    _innovations = np.array(
            [[[ 0.31559906, -1.80551443, -0.49468605, -1.04842416],
            [ 1.47092134, -0.18375408,  0.61439572,  0.64938352]],

           [[-1.62316804, -0.76478879, -0.75826613,  0.17750069],
            [ 0.82250737,  0.10701289, -1.15691368,  1.05573836]],

           [[-0.22774868,  0.74240245,  1.11521215,  1.10464703],
            [-0.56968483,  0.54138324, -0.69508153, -0.14254879]]])

    _returns = _innovations * VOL_LEVEL + RET_EXP

    _variances = VAR_LEVEL  * np.array(
            [[[ 3.40880883,  1.50217308,  1.11125856,  0.14462167],
            [ 0.08321133,  0.36507093,  1.52133949,  1.42666615]],

           [[ 0.78940018,  0.96257061,  0.3341445 ,  0.59402684],
            [ 1.67863795,  1.21765875,  1.50332767,  1.03224149]],

           [[ 0.78179542,  1.32126548,  1.64830936,  1.59253612],
            [ 0.21397881,  0.01596482,  2.48993567,  1.3034988 ]]])

    _first_variance = _variances[0]

    _shape = _innovations.shape

    def timeseries_filter(self, model, returns=_returns):
        return model.timeseries_filter(returns, self._first_variance)

    def timeseries_generate(self, model, innovations=_innovations):
        return model.timeseries_generate(innovations, self._first_variance)

    def test_generate_revertsto_filter(self, model):
        generated_variances, generated_returns = \
                self.timeseries_generate(model)
        filtered_variances, filtered_innovations = \
                self.timeseries_filter(model, returns=generated_returns)
        pt.assert_almost_equal(generated_variances, filtered_variances)
        pt.assert_almost_equal(self._innovations, filtered_innovations)

    def test_filter_initialize_variance(self, model):
        filtered_variance, _ = self.timeseries_filter(model)
        pt.assert_almost_equal(filtered_variance[0], self._first_variance)

    def test_filter_generate_variance(self, model):
        generated_variance, _ = self.timeseries_generate(model)
        pt.assert_almost_equal(generated_variance[0], self._first_variance)

    def test_filter_size(self, model):
        next_variances, innovations = self.timeseries_filter(model)
        expected_variance_shape = (self._shape[0]+1,) + self._shape[1:]
        assert next_variances.shape == expected_variance_shape
        assert innovations.shape == self._shape

    def test_filter_positive_variance(self, model):
        variances, _ = self.timeseries_filter(model)
        pt.assert_true(variances>0)

    def test_filter_throws_exception_when_non_positive_variance(self, model):
        with pytest.raises(ValueError):
            model.timeseries_filter(np.nan, -1)
        with pytest.raises(ValueError):
            model.timeseries_filter(np.nan, 0)

    def test_generate_size(self, model):
        variances, returns = self.timeseries_generate(model)
        expected_shape = (self._shape[0]+1,) + self._shape[1:]
        assert variances.shape == expected_shape
        assert returns.shape == self._shape

    def test_generate_positive_variance(self, model):
        variances,_ = self.timeseries_generate(model)
        pt.assert_true(variances>0)

    def test_generate_throws_exception_when_non_positive_variance(
            self, model):
        with pytest.raises(ValueError):
            model.timeseries_generate(np.nan, -1)
        with pytest.raises(ValueError):
            model.timeseries_generate(np.nan, 0)

class TestOneStep():

    _innovations = np.array(
            [[[ 0.36841639, -0.57224794,  0.25019277, -1.24721141],
            [ 1.35128646, -0.92096528, -0.92572966, -0.98178342],
            [-0.84366922, -1.28207108, -1.53467485, -0.22874378]],

            [[-1.05050049, -1.31158806, -1.17961533,  0.05363092],
            [-1.67960499,  0.02364286,  0.28870878,  0.01871441],
            [ 1.71135472,  0.7167683 , -0.93715744, -0.8482234 ]]])

    _returns = _innovations * VOL_LEVEL + RET_EXP

    _variances =  VAR_LEVEL * np.array(
            [[[ 0.3217341 ,  0.00868056,  1.22722262,  0.58144423],
            [ 0.54204103,  3.68888772,  0.77486123,  0.95989976],
            [ 0.90888372,  0.07370864,  2.91764007,  2.21766175]],

            [[ 0.7019393 ,  2.38621637,  1.24599797,  1.17731278],
            [ 0.81830769,  0.26109424,  0.2980641 ,  0.85817885],
            [ 0.93831248,  0.17303652,  0.45056502,  2.22328546]]])

    _shape = _innovations.shape

    _innovations_broadcast = np.array([[-0.70687441],
            [ 0.72613442],
            [-2.20412955],
            [-0.27571779],
            [ 1.38167603],
            [-0.96744496],
            [-0.05243877],
            [-0.41254891],
            [-1.27985463],
            [ 0.12262462]])

    _variances_broadcast =  VAR_LEVEL * np.array([[ 2.70588279,  0.96680575,
            0.09027765, 1.23017577,  1.14468471]])

    _shape_broadcast = (_innovations_broadcast.shape[0],
            _variances_broadcast.shape[1])

    # I/O


    def test_support_scalar(self, model, one_step_str):
        f = getattr(model, one_step_str)
        allowed_primitives = (bool, float)
        # Actual inputs do not matter here
        # Only testing shapes...
        out = f(np.nan, np.nan)
        if isinstance(out, tuple):
            for o in out:
                msg_ = 'Func was {}, Type was: {}'.format(f, type(o))
                pt.assert_true(type(o) in allowed_primitives, msg=msg_)
        else:
            msg_ = 'Func was {}, Type was: {}'.format(f, type(out))
            pt.assert_true(type(out) in allowed_primitives, msg=msg_)

    def test_support_array(self, model, one_step_str):
        f = getattr(model, one_step_str)
        # Actual inputs do not matter here
        # Only testing shapes...
        msg_ = 'Func was {}'.format(f)
        out = f(self._innovations, self._variances)
        if isinstance(out, tuple):
            for o in out:
                pt.assert_equal(o.shape, self._shape, msg=msg_)
        else:
            pt.assert_equal(out.shape, self._shape, msg=msg_)

    def test_support_broadcasting(self, model, one_step_str):
        f = getattr(model, one_step_str)
        # Actual inputs do not matter here
        # Only testing shapes...
        msg_ = 'Func was {}'.format(f)
        out = f(self._innovations_broadcast,
                self._variances_broadcast)
        if isinstance(out, tuple):
            for o in out:
                pt.assert_equal(o.shape, self._shape_broadcast,
                        msg=msg_)
        else:
            pt.assert_equal(out.shape, self._shape_broadcast, msg=msg_)

    # One-step filter

    def one_step_filter(self, model, returns=_returns):
        return model.one_step_filter(returns, self._variances)

    def one_step_generate(self, model, innovations=_innovations):
        return model.one_step_generate(innovations, self._variances)

    def test_one_step_generate_revertsto_filter(self, model):
        generated_variances, generated_returns = \
                self.one_step_generate(model)
        filtered_variances, filtered_innovations = \
                self.one_step_filter(model, returns=generated_returns)
        pt.assert_almost_equal(generated_variances, filtered_variances)
        pt.assert_almost_equal(self._innovations, filtered_innovations)

    def test_one_step_filter_positive_variance(self, model):
        next_variances, _ = self.one_step_filter(model)
        pt.assert_true(next_variances>0)

    def test_one_step_generate_positive_variance(self, model):
        next_variances, _ = self.one_step_generate(model)
        pt.assert_true(next_variances>0)

    # One-step innovation roots

    def get_lowest_one_step_variance(self, model):
        return model.get_lowest_one_step_variance(self._variances)

    def get_valid_one_step(self):
        next_variances = self._variances
        variances = np.min(next_variances)*0.9
        return (variances, next_variances)

    def get_invalid_one_step(self, model):
        variances = self._variances
        next_variances = \
                0.9*self.get_lowest_one_step_variance(model)
        return (variances, next_variances)

    def test_one_step_roots_when_valid(self, model):
        (variances, next_variances) = self.get_valid_one_step()
        left_roots, right_roots = model.one_step_roots(
                variances,
                next_variances)
        for (z,s) in zip((left_roots, right_roots), ('left', 'right')):
            next_variances_solved, _ = model.one_step_generate(
                z, variances)
            pt.assert_almost_equal(next_variances_solved, next_variances,
                    msg='Invalid ' + s + ' roots.')

    def test_one_step_roots_left_right_order(self, model):
        (variances, next_variances) = self.get_valid_one_step()
        left_roots, right_roots = model.one_step_roots(
                variances,
                next_variances)
        pt.assert_true(left_roots<right_roots)

    def test_roots_nan_when_invalid(self, model):
        (variances, next_variances) = self.get_invalid_one_step(model)
        left_roots, right_roots = model.one_step_roots(
                self._variances, next_variances)
        pt.assert_equal(left_roots, np.nan)
        pt.assert_equal(right_roots, np.nan)

    def test_same_roots_at_singularity(self, model):
        variances_at_singularity = self.get_lowest_one_step_variance(model)
        left_roots, right_roots = model.one_step_roots(
                self._variances, variances_at_singularity)
        pt.assert_almost_equal(left_roots, right_roots)

    def test_roots_at_inf_are_pm_inf(self, model):
        [left_root, right_root] = model.one_step_roots(
                self._variances, np.inf)
        pt.assert_equal(left_root, -np.inf)
        pt.assert_equal(right_root, np.inf)

    def test_roots_at_zero_are_nan(self, model):
        [left_root, right_root] = model.one_step_roots(self._variances, 0)
        pt.assert_equal(left_root, np.nan)
        pt.assert_equal(right_root, np.nan)

    def test_roots_derivatives_when_valid(self, model):
        (variances, at) = self.get_valid_one_step()
        def roots_func(next_variances):
            _, z = model.one_step_roots(
                    variances,
                    next_variances)
            return z
        def roots_der(next_variances):
            dz = model.one_step_roots_unsigned_derivative(
                    variances,
                    next_variances)
            return dz
        pt.assert_derivative_at(roots_der, roots_func, at,
                mode='forward', rtol=1e-4)

    def test_roots_derivative_is_positive(self, model):
        (variances, next_variances) = self.get_valid_one_step()
        der = model.one_step_roots_unsigned_derivative(
                variances,
                next_variances)
        pt.assert_true(der>0)

    def test_roots_derivatives_nan_when_invalid(self, model):
        (variances, next_variances) = self.get_invalid_one_step(model)
        der = model.one_step_roots_unsigned_derivative(
                variances, next_variances)
        pt.assert_equal(der, np.nan)

    def test_roots_derivatives_is_nan_at_singularity(self, model):
        variances_at_singularity = self.get_lowest_one_step_variance(model)
        der = model.one_step_roots_unsigned_derivative(
                self._variances, variances_at_singularity)
        pt.assert_equal(der, np.nan)

    def test_roots_derivatives_at_inf_are_null(self, model):
        der = model.one_step_roots_unsigned_derivative(
                self._variances, np.inf)
        pt.assert_equal(der, 0.)

    def test_roots_derivatives_at_zero_are_nan(self, model):
        der = model.one_step_roots_unsigned_derivative(
                self._variances, 0)
        pt.assert_equal(der, np.nan)

    def test_root_from_return(self, model):
        price = 100.
        next_prices = np.array([95.,100.,105.])
        returns = np.log(next_prices/price)
        z = model.one_step_root_from_return(returns, VAR_LEVEL)
        _, returns_solved = model.one_step_generate(z, VAR_LEVEL)
        pt.assert_almost_equal(returns_solved, returns,
                msg='Invalid root from price.')

    # One-step variance integration
    _test_expectation_values = np.array(
            [0.123,-0.542,-1.3421,1.452,5.234,-3.412,10])

    def test_one_step_expectation_until(self, model):
        def func_to_integrate(z):
            (h, _) = model.one_step_generate(z, VAR_LEVEL)
            return h * norm.pdf(z)
        integral = partial(model.one_step_expectation_until, VAR_LEVEL)
        pt.assert_integral_until(integral, func_to_integrate,
                self._test_expectation_values, lower_bound=-100)

    def test_one_step_expectation_squared_until(self, model):
        def func_to_integrate(z):
            (h, _) = model.one_step_generate(z, VAR_LEVEL)
            return h**2. * norm.pdf(z)
        integral = partial(model.one_step_expectation_until, VAR_LEVEL,
                order=2)
        pt.assert_integral_until(integral, func_to_integrate,
                self._test_expectation_values, lower_bound=-10)

    # TODO: send to estimator
    # def test_neg_log_like_at_few_values(self, model):
    #     nll = self.model.negative_log_likelihood(1, 1)
    #     pt.assert_almost_equal(nll, 0.5)
    #     nll = self.model.negative_log_likelihood(0, np.exp(1))
    #     pt.assert_almost_equal(nll, 0.5)
