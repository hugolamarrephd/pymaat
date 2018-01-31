from functools import partial

import pytest
import numpy as np
from scipy.stats import norm
import scipy.optimize

import pymaat.testing as pt
import pymaat.garch.model


VAR_LEVEL = 0.18**2./252.
VOL_LEVEL = np.sqrt(VAR_LEVEL)
RET_EXP = 0.06/252.

ALL_MODELS = [
            pymaat.garch.model.HestonNandiGarch(
                mu=2.01,
                omega=9.75e-20,
                alpha=4.54e-6,
                beta=0.79,
                gamma=196.21),
             ]

ALL_MODEL_IDS = ['HN-GARCH',]

@pytest.fixture(params=ALL_MODELS, ids=ALL_MODEL_IDS, scope='module')
def model(request):
    return request.param

@pytest.fixture(scope='class')
def innovations(shape):
    np.random.seed(1234566) # Ensures test consistency
    return np.random.normal(size=shape)

@pytest.fixture(scope='class')
def positive_rvs(shape):
    np.random.seed(1234566) # Ensures test consistency
    return np.random.gamma(1., size=shape)

@pytest.fixture(scope='class')
def returns(innovations):
    return innovations * VOL_LEVEL + RET_EXP

@pytest.fixture(scope='class')
def variances(positive_rvs):
    return VAR_LEVEL * positive_rvs

class TestTimeseries():


    @pytest.fixture(params=[(10,), (10,3), (10,5,6),], scope='class')
    def shape(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def first_variance(self, variances):
        return variances[4]

    @pytest.fixture(params=[0.0, -0.00234, np.nan], scope='class')
    def invalid_first_variance(self, request, first_variance):
        invalid = first_variance.copy()
        if invalid.ndim==0: # scalar case
            return np.float64(request.param)
        else:
            invalid[-1] = request.param
            return invalid

    def test_generate_revert_to_filter(self, model,
            returns, first_variance, innovations):
        generated_variances, generated_returns = \
                model.timeseries_generate(innovations, first_variance)
        filtered_variances, filtered_innovations = \
                model.timeseries_filter(generated_returns, first_variance)
        pt.assert_almost_equal(generated_variances, filtered_variances)
        pt.assert_almost_equal(innovations, filtered_innovations)

    def test_filter_initialize_to_first_variance(self, model,
            returns, first_variance):
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
        pt.assert_true(variances>0.0)

    def test_generate_positive_variance(self, model,
            innovations, first_variance):
        variances,_ = model.timeseries_generate(innovations, first_variance)
        pt.assert_true(variances>0.0)

class TestOneStep():


    @pytest.fixture(params=[(1,), (2,3), (4,5,6),], scope='class')
    def shape(self, request):
        return request.param

    @pytest.fixture(params=[True,False], scope='class')
    def masked(self, request):
        return request.param

    @pytest.fixture(scope = 'class')
    def lowest_one_step(self, masked, model, variances):
        singularity = model.get_lowest_one_step_variance(variances)
        if masked:
            return np.ma.array(singularity)
        else:
            return singularity

    @pytest.fixture(scope = 'class')
    def valid_one_step(self, masked, lowest_one_step, positive_rvs):
        above_singularity = lowest_one_step*(1.+positive_rvs)
        if masked:
            return np.ma.masked_array(above_singularity, copy=False)
        else:
            return above_singularity

    @pytest.fixture(scope = 'class')
    def invalid_one_step(self, masked, lowest_one_step, positive_rvs):
        below_singularity = lowest_one_step*np.exp(-positive_rvs)
        if masked:
            return np.ma.masked_array(below_singularity, copy=False)
        else:
            return below_singularity

    @pytest.fixture(scope = 'class')
    def inf(masked):
        if masked:
            return np.ma.array([np.inf])
        else:
            return np.array([np.inf])

    @pytest.fixture(scope = 'class')
    def minf(masked):
        if masked:
            return np.ma.array([-np.inf])
        else:
            return np.array([-np.inf])

    # Run element-by-element test suite
    @pytest.mark.parametrize("funcname_nin_nout",
            [('one_step_filter',2,2),
            ('one_step_generate',2,2),
            ('get_lowest_one_step_variance',1,1),
            ('one_step_roots',2,2),
            ('one_step_roots_unsigned_derivative',2,1),
            ('one_step_expectation_until',2,1)],
            ids=lambda val: val[0])
    def test_all_are_elbyel(self, model, funcname_nin_nout):
        funcname, *nin_nout = funcname_nin_nout
        f = getattr(model, funcname) # As a bounded method
        pt.assert_elbyel(f, *(nin_nout))

    # One-step filter and generate

    def test_one_step_generate_revertsto_filter(self, model,
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
        pt.assert_true(next_variances>0)

    def test_one_step_generate_positive_variance(self, model,
            innovations, variances):
        next_variances, _ = model.one_step_generate(innovations, variances)
        pt.assert_true(next_variances>0)

    # One-step innovation roots

    def test_lowest_one_step_variance(self, model):
        value = model.get_lowest_one_step_variance(VAR_LEVEL)
        def func(z):
            next_variance, _ = model.one_step_generate(z, VAR_LEVEL)
            return next_variance
        res = scipy.optimize.brute(func, ranges=((-1.,1.),),
                full_output=True)
        expected_value = res[1]
        pt.assert_almost_equal(value, expected_value, rtol=1e-6)

    def test_one_step_roots_when_valid(self, model, masked,
            variances, valid_one_step):
        roots = model.one_step_roots(variances, valid_one_step)
        for z in roots:
            # Valid output?
            if masked:
                pt.assert_false(z.mask)
                pt.assert_false(np.isnan(z.data))
            else:
                pt.assert_false(np.isnan(z))
            # Check that it reverts to...
            solved, _ = model.one_step_generate(z, variances)
            pt.assert_almost_equal(solved, valid_one_step)

    def test_one_step_roots_left_right_order(self, model,
            variances, valid_one_step):
        left, right = model.one_step_roots(variances, valid_one_step)
        pt.assert_true(left<right)

    def test_roots_nan_OR_masked_when_invalid(self, model, masked,
            variances, invalid_one_step):
        roots = model.one_step_roots(variances, invalid_one_step)
        for z in roots:
            if masked:
                pt.assert_true(z.mask)
                # Rem. z.data may contain valid floats (ie not NaNs)!
            else:
                pt.assert_true(np.isnan(z))

    def test_same_roots_at_singularity(self, model,
            variances, lowest_one_step):
        left, right = model.one_step_roots(variances, lowest_one_step)
        pt.assert_almost_equal(left, right, rtol=1e-16)

    def test_root_at_inf_is_pm_inf(self, model, masked, variances, inf):
        roots = model.one_step_roots(variances, inf)
        if masked:
            for z in roots:
                pt.assert_false(z.mask)
        if masked:
            pt.assert_equal(roots[0].data, -np.inf)
            pt.assert_equal(roots[1].data, np.inf)
        else:
            pt.assert_equal(roots[0], -np.inf)
            pt.assert_equal(roots[1], np.inf)

    def test_roots_derivatives_when_valid(self, model, masked,
            variances, valid_one_step):
        # Test validity?
        der = model.one_step_roots_unsigned_derivative(
                variances, valid_one_step)
        if masked:
            pt.assert_false(der.mask)
            pt.assert_false(np.isnan(der.data))
        else:
            pt.assert_false(np.isnan(der))

        # Derivative value?
        def func(next_variances):
            _, z = model.one_step_roots(variances, next_variances)
            return z
        pt.assert_derivative_at(der, func,  valid_one_step, rtol=1e-4)

    def test_roots_derivative_is_positive(self, model,
            variances, valid_one_step):
        der = model.one_step_roots_unsigned_derivative(
                variances, valid_one_step)
        pt.assert_true(der>0.)

    def test_roots_derivatives_nan_OR_masked_when_invalid(self,
            model, masked,
            variances, invalid_one_step):
        der = model.one_step_roots_unsigned_derivative(
                variances, invalid_one_step)
        if masked:
            pt.assert_true(der.mask)
            # Rem. der.data may contain valid floats (ie non NaNs)!
        else:
            pt.assert_true(np.isnan(der))

    def test_roots_derivatives_is_inf_at_singularity(self, model, masked,
            variances, lowest_one_step):
        der = model.one_step_roots_unsigned_derivative(
                variances, lowest_one_step)
        if masked:
            pt.assert_false(der.mask)
            pt.assert_equal(der.data, np.inf)
        else:
            pt.assert_equal(der, np.inf)

    def test_roots_derivatives_at_inf_are_zero(self, model, masked,
            variances, inf):
        der = model.one_step_roots_unsigned_derivative(variances, inf)
        if masked:
            pt.assert_false(der.mask)
            pt.assert_equal(der.data, 0.)
        else:
            pt.assert_equal(der, 0.)

    # One-step variance integration
    @pytest.mark.slow
    @pytest.mark.parametrize("order", [0,1,2])
    def test_one_step_expectation_until(self, model, innovations, order):
        def func_to_integrate(z):
            (h, _) = model.one_step_generate(z, VAR_LEVEL)
            return h**np.float64(order)*norm.pdf(z)
        integral = model.one_step_expectation_until(VAR_LEVEL, innovations,
                order=order)
        pt.assert_integral_until(integral, func_to_integrate,
                innovations, lower_bound=-10., rtol=1e-4)

    @pytest.mark.slow
    @pytest.mark.parametrize("order", [0,1,2])
    def test_one_step_expectation_until_inf(self, model, masked,
            order, inf):
        def func_to_integrate(z):
            (h, _) = model.one_step_generate(z, VAR_LEVEL)
            return h**np.float64(order)*norm.pdf(z)
        integral = model.one_step_expectation_until(VAR_LEVEL, inf,
                order=order)
        pt.assert_integral_until(integral, func_to_integrate, 10.,
                lower_bound=-10., rtol=1e-4)

    @pytest.mark.parametrize("order", [0,1,2])
    def test_one_step_expectation_until_minf(self, model, order, minf):
        integral = model.one_step_expectation_until(VAR_LEVEL, minf,
                order=order)
        assert integral[0] == 0.0


    @pytest.mark.parametrize("invalid_order", ['abc',3,-1,2.123])
    def test_one_step_expectation_until_unexpected_order(self, model,
            invalid_order):
        with pytest.raises(ValueError):
            model.one_step_expectation_until(VAR_LEVEL, 0.,
                    order=invalid_order)


    # TODO: send to estimator
    # def test_neg_log_like_at_few_values(self, model):
    #     nll = self.model.negative_log_likelihood(1, 1)
    #     pt.assert_almost_equal(nll, 0.5)
    #     nll = self.model.negative_log_likelihood(0, np.exp(1))
    #     pt.assert_almost_equal(nll, 0.5)
