from functools import partial

import pytest
import numpy as np
from scipy.stats import norm
import scipy.optimize

import pymaat.testing as pt

VAR_LEVEL = 0.18**2./252.
VOL_LEVEL = np.sqrt(VAR_LEVEL)
RET_EXP = 0.06/252.


@pytest.fixture
def innovations(random_normal):
    return random_normal


@pytest.fixture
def returns(innovations):
    return innovations * VOL_LEVEL + RET_EXP


@pytest.fixture
def variances(random_gamma):
    return VAR_LEVEL * random_gamma


class TestTimeseries():

    @pytest.fixture(params=[(10,), (10, 3), (10, 5, 6), ])
    def shape(self, request):
        """
        Shape of inputted arrays (variances, innovations & returns)
        The first dimension represents time
        """
        return request.param

    @pytest.fixture
    def first_variance(self, variances):
        return variances[4]

    def test_generate_reverts_to_filter(
            self, model, returns, first_variance, innovations):
        generated_variances, generated_returns = \
            model.timeseries_generate(innovations, first_variance)
        filtered_variances, filtered_innovations = \
            model.timeseries_filter(generated_returns, first_variance)
        pt.assert_almost_equal(generated_variances, filtered_variances)
        pt.assert_almost_equal(innovations, filtered_innovations)

    def test_filter_initialize_to_first_variance(
            self, model, returns, first_variance):
        filtered_variance, _ = model.timeseries_filter(
            returns, first_variance)
        pt.assert_almost_equal(filtered_variance[0], first_variance)

    def test_generate_initialize_to_first_variance(self, model,
                                                   innovations, first_variance):
        generated_variance, _ = model.timeseries_generate(
            innovations, first_variance)
        pt.assert_almost_equal(generated_variance[0], first_variance)

    def test_filter_shape(self, model, shape,
                          returns, first_variance):
        next_variances, innovations = model.timeseries_filter(
            returns, first_variance)
        expected_variance_shape = (shape[0]+1,) + shape[1:]
        assert next_variances.shape == expected_variance_shape
        assert innovations.shape == shape

    def test_generate_shape(self, model, shape,
                            innovations, first_variance):
        variances, returns = model.timeseries_generate(innovations,
                                                       first_variance)
        expected_shape = (shape[0]+1,) + shape[1:]
        assert variances.shape == expected_shape
        assert returns.shape == shape

    @pytest.fixture(params=[0.0, -0.00234, np.nan])
    def invalid_first_variance(self, request, first_variance):
        invalid = first_variance.copy()
        if invalid.ndim == 0:  # scalar case
            return np.float64(request.param)
        else:
            invalid[-1] = request.param
            return invalid

    def test_filter_value_error_when_non_positive_first_variance(
            self, model, returns, invalid_first_variance):
        with pytest.raises(ValueError):
            model.timeseries_filter(returns, invalid_first_variance)

    def test_generate_value_error_when_non_positive_first_variance(
            self, model, innovations, invalid_first_variance):
        with pytest.raises(ValueError):
            model.timeseries_generate(innovations, invalid_first_variance)

    def test_filter_positive_variance(self, model, returns, first_variance):
        variances, _ = model.timeseries_filter(returns, first_variance)
        pt.assert_greater(variances, 0.0, shape='broad')

    def test_generate_positive_variance(self, model,
                                        innovations, first_variance):
        variances, _ = model.timeseries_generate(innovations, first_variance)
        pt.assert_greater(variances, 0.0, shape='broad')


class TestOneStep():

    @pytest.fixture(params=[(1,), (2, 3), (4, 5, 6), ])
    def shape(self, request):
        return request.param

    @pytest.fixture
    def lowest_one_step(self, model, variances):
        return model.get_lowest_one_step_variance(variances)

    @pytest.fixture
    def valid_one_step(self, lowest_one_step, random_gamma):
        return lowest_one_step*(1.+random_gamma)

    @pytest.fixture
    def invalid_one_step(self, lowest_one_step, random_gamma):
        return lowest_one_step*np.exp(-random_gamma)

    # Run element-by-element test suite
    @pytest.mark.parametrize("funcname_nin_nout",
                             [('one_step_filter', 2, 2),
                              ('one_step_generate', 2, 2),
                              ('get_lowest_one_step_variance', 1, 1),
                              ('real_roots', 2, 2),
                              ('real_roots_unsigned_derivative', 2, 1),
                              ('variance_integral_until', 2, 1)],
                             ids=lambda val: val[0])
    def test_all_are_elbyel(self, model, funcname_nin_nout):
        funcname, *nin_nout = funcname_nin_nout
        f = getattr(model, funcname)  # As a bounded method
        pt.assert_elbyel(f, *(nin_nout))

    # One-step filter and generate

    def test_one_step_generate_revert_to_filter(self, model,
                                                innovations, variances):
        generated_variances, generated_returns = model.one_step_generate(
            innovations, variances)
        filtered_variances, filtered_innovations = \
            model.one_step_filter(generated_returns, variances)
        pt.assert_almost_equal(generated_variances, filtered_variances)
        pt.assert_almost_equal(innovations, filtered_innovations)

    def test_one_step_filter_positive_variance(self, model,
                                               returns, variances):
        next_variances, _ = model.one_step_filter(returns, variances)
        pt.assert_greater(next_variances, 0.0, shape='broad')

    def test_one_step_generate_positive_variance(self, model,
                                                 innovations, variances):
        next_variances, _ = model.one_step_generate(innovations, variances)
        pt.assert_greater(next_variances, 0.0, shape='broad')

    # One-step innovation roots

    def test_lowest_one_step_variance(self, model):
        value = model.get_lowest_one_step_variance(VAR_LEVEL)

        def func(z):
            next_variance, _ = model.one_step_generate(z, VAR_LEVEL)
            return next_variance
        res = scipy.optimize.brute(func, ranges=((-1., 1.),),
                                   full_output=True)
        expected_value = res[1]
        pt.assert_almost_equal(value, expected_value, rtol=1e-6)

    def test_valid_roots(self, model, variances, valid_one_step):
        roots = model.real_roots(variances, valid_one_step)
        for z in roots:
            # Check that it reverts to...
            solved, _ = model.one_step_generate(z, variances)
            pt.assert_almost_equal(solved, valid_one_step)

    def test_left_right_roots_order(self, model,
                                    variances, valid_one_step):
        left, right = model.real_roots(variances, valid_one_step)
        pt.assert_less(left, right)

    def test_invalid_roots(self, model, variances, invalid_one_step):
        roots = model.real_roots(variances, invalid_one_step)
        for z in roots:
            pt.assert_valid(z)  # Still returns valid values (real part)

    def test_same_roots_at_singularity(self, model,
                                       variances, lowest_one_step):
        left, right = model.real_roots(variances, lowest_one_step)
        pt.assert_almost_equal(left, right, rtol=1e-16)

    def test_root_at_inf_is_pm_inf(self, model, variances, inf):
        roots = model.real_roots(variances, inf)
        pt.assert_equal(roots[0], -np.inf, shape='broad')
        pt.assert_equal(roots[1], np.inf, shape='broad')

    def test_roots_derivatives_when_valid(self, model,
                                          variances, valid_one_step):
        # Test validity?
        der = model.real_roots_unsigned_derivative(
            variances, valid_one_step)
        # Derivative value?

        def func(next_variances):
            _, z = model.real_roots(variances, next_variances)
            return z
        pt.assert_derivative_at(der, valid_one_step,
                                function=func, rtol=1e-4)

    def test_roots_derivative_is_positive(self, model,
                                          variances, valid_one_step):
        der = model.real_roots_unsigned_derivative(
            variances, valid_one_step)
        pt.assert_greater(der, 0.0, shape='broad')

    def test_invalid_roots_derivatives(self, model, variances,
                                       invalid_one_step):
        der = model.real_roots_unsigned_derivative(
            variances, invalid_one_step)
        pt.assert_equal(der, 0., shape='broad')

    def test_roots_derivatives_is_zero_at_singularity(
            self, model, variances, lowest_one_step):
        der = model.real_roots_unsigned_derivative(
            variances, lowest_one_step)
        pt.assert_equal(der, 0., shape='broad')

    def test_roots_derivatives_at_inf_are_zero(
            self, model, variances, inf):
        der = model.real_roots_unsigned_derivative(variances, inf)
        pt.assert_equal(der, 0., shape='broad')

    # One-step variance integration
    @pytest.mark.parametrize("order", [0, 1, 2])
    @pytest.mark.parametrize("shape", (1,))
    def test_variance_integral_until(
            self, model, innovations, variances, order):
        integral = model.variance_integral_until(variances, innovations,
                                                 order=order)
        def func_to_integrate(z):
            (h, _) = model.one_step_generate(z, variances)
            return h**np.float64(order)*norm.pdf(z)
        pt.assert_integral_until(integral, func_to_integrate,
                                 innovations, lower_bound=-10., rtol=1e-4)

    @pytest.mark.parametrize("order", [0, 1, 2])
    @pytest.mark.parametrize("shape", (1,))
    def test_variance_integral_until_inf(
            self, inf, model, variances, order):
        # Value
        integral = model.variance_integral_until(
                variances, inf, order=order)
        # Expected value
        def func_to_integrate(z):
            (h, _) = model.one_step_generate(z, variances)
            return h**np.float64(order)*norm.pdf(z)
        pt.assert_integral_until(integral, func_to_integrate, inf,
                                 rtol=1e-2)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_variance_integral_until_minf(
            self, minf, model, variances, order):
        integral = model.variance_integral_until(
                variances, minf, order=order)
        pt.assert_equal(integral, 0.0, shape='broad')

    @pytest.mark.parametrize("invalid_order", ['abc', 3, -1, 2.123, None])
    def test_variance_integral_until_unexpected_order(
            self, model, variances, invalid_order):
        with pytest.raises(ValueError):
            model.variance_integral_until(variances, 0., order=invalid_order)


class TestRetSpec():

    @pytest.fixture(params=[(1,), (2, 3), (4, 5, 6), ])
    def shape(self, request):
        return request.param

    def test_generate_reverts_to_filter(
            self, model, innovations, variances):
        returns = model.retspec.one_step_generate(innovations, variances)
        expected_innovations = model.retspec.one_step_filter(
                returns, variances)
        pt.assert_almost_equal(innovations, expected_innovations)

    def test_root_price_derivative(
            self, model, returns, variances):
        prices = np.exp(returns)
        der = model.retspec.root_price_derivative(prices, variances)
        def func_to_derivate(prices):
            returns = np.log(prices)
            return model.retspec.one_step_filter(returns, variances)
        pt.assert_derivative_at(der, prices, function=func_to_derivate)

    def test_root_price_der_at_zero_is_inf(self, model, variances):
        der = model.retspec.root_price_derivative(0., variances)
        pt.assert_equal(der, np.inf, shape = 'broad')

    def test_root_price_der_at_inf_is_zero(self, model, variances):
        der = model.retspec.root_price_derivative(np.inf, variances)
        pt.assert_equal(der, 0.0, shape = 'broad')

    def test_root_price_der_at_negative(self, model, variances):
        der = model.retspec.root_price_derivative(-1., variances)
        pt.assert_invalid(der)

    @pytest.mark.parametrize("order", [0, 1, 2])
    @pytest.mark.parametrize("shape", (1,))
    def test_price_integral_until(
            self, model, returns, variances, innovations, order):
        prices = np.exp(returns)
        integral = model.retspec.price_integral_until(
                prices, variances, innovations, order=order)
        def func_to_integrate(z):
            next_prices = prices*np.exp(
                    model.retspec.one_step_generate(z, variances))
            return next_prices**np.float64(order)*norm.pdf(z)
        pt.assert_integral_until(integral, func_to_integrate,
                                 innovations, lower_bound=-10.)

    @pytest.mark.parametrize("order", [0, 1, 2])
    @pytest.mark.parametrize("shape", (1,))
    def test_price_integral_until_inf(
            self, model, returns, variances, order):
        prices = np.exp(returns)
        integral = model.retspec.price_integral_until(
                prices, variances, np.inf, order=order)
        def func_to_integrate(z):
            next_prices = prices*np.exp(
                    model.retspec.one_step_generate(z, variances))
            return next_prices**np.float64(order)*norm.pdf(z)
        pt.assert_integral_until(
                integral, func_to_integrate, np.inf)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_price_integral_until_minf(
            self, model, returns, variances, order):
        prices = np.exp(returns)
        integral = model.retspec.price_integral_until(
                prices, variances, -np.inf, order=order)
        pt.assert_equal(integral, 0., shape='broad')

    @pytest.mark.parametrize("invalid_order", ['abc', 3, -1, 2.123, None])
    def test_variance_integral_until_unexpected_order(
            self, model, returns, variances, invalid_order):
        prices = np.exp(returns)
        with pytest.raises(ValueError):
            model.retspec.price_integral_until(
                    prices, variances, 0., order=invalid_order)

        # TODO: send to estimator
        # def test_neg_log_like_at_few_values(self, model):
        #     nll = self.model.negative_log_likelihood(1, 1)
        #     pt.assert_almost_equal(nll, 0.5)
        #     nll = self.model.negative_log_likelihood(0, np.exp(1))
        #     pt.assert_almost_equal(nll, 0.5)
