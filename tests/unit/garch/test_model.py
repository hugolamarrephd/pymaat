import pytest
import numpy as np
import scipy.optimize
import scipy.integrate as integrate

import pymaat.testing as pt
from pymaat.mathutil import normcdf, normpdf
import pymaat.quantutil as qutil

VAR_LEVEL = 0.18**2./252.
VOL_LEVEL = np.sqrt(VAR_LEVEL)
RET_EXP = 0.06/252.


@pytest.fixture
def innovations(random_normal):
    return random_normal


@pytest.fixture
def returns(innovations, variance_scale):
    return innovations * np.sqrt(variance_scale) + RET_EXP


@pytest.fixture
def variances(random_gamma, variance_scale):
    return variance_scale * random_gamma


class TestTimeseries():

    @pytest.fixture(params=[(10,), (10, 3), (252, 1000)])
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


@pytest.mark.slow  # Simulation-based validation
class TestTermstruct():
    RTOL = 1e-2
    NSIM = 1000000

    @pytest.fixture(scope='class')
    def until(self):
        return 252

    @pytest.fixture(scope='class')
    def h0(self):
        return 0.14**2./252.

    @pytest.fixture(scope='class')
    def sim_variance_returns(self, model, h0, until):
        np.random.seed(123819023)
        return model.timeseries_generate(
            np.random.normal(size=(until, self.NSIM)), h0)

    @pytest.fixture
    def sim_variance(self, sim_variance_returns):
        return sim_variance_returns[0]

    @pytest.fixture
    def sim_logprice(self, sim_variance_returns):
        return np.cumsum(sim_variance_returns[1], axis=0)

    def test_variance_expect(self, model, h0, until, sim_variance):
        expect, _ = model.termstruct_variance(h0, until)
        approx_expect = np.mean(sim_variance, axis=1)
        pt.assert_almost_equal(approx_expect, expect, rtol=self.RTOL)

    def test_variance_variance(self, model, h0, until, sim_variance):
        _, var = model.termstruct_variance(h0, until)
        approx_var = np.var(sim_variance, axis=1)
        approx_var[0] = 0.
        pt.assert_almost_equal(approx_var, var, rtol=self.RTOL)

    def test_logprice_expect(self, model, h0, until, sim_logprice):
        # First values are difficult to estimate
        #   since sqrt(Var[log-price]) >> E[log-price]
        expect, var = model.termstruct_logprice(h0, until)
        to_test = np.sqrt(var)/expect < 10
        assert np.any(to_test)  # make sure at least testing one element
        approx_expect = np.mean(sim_logprice, axis=1)
        pt.assert_almost_equal(
            approx_expect[to_test], expect[to_test], rtol=self.RTOL)

    def test_logprice_variance(self, model, h0, until, sim_logprice):
        expect, var = model.termstruct_logprice(h0, until)
        approx_var = np.var(sim_logprice, axis=1)
        pt.assert_almost_equal(approx_var, var, rtol=self.RTOL)


class TestOneStep():

    @pytest.fixture(params=[(1,), (2, 3), (4, 5, 6), ])
    def shape(self, request):
        return request.param

    @pytest.fixture
    def lowest_one_step(self, model, variances):
        return model.one_step_bounds(variances)[0]

    @pytest.fixture
    def valid_one_step(self, lowest_one_step, random_gamma):
        return lowest_one_step*(1.+random_gamma)

    @pytest.fixture
    def invalid_one_step(self, lowest_one_step, random_gamma):
        return lowest_one_step*np.exp(-random_gamma)

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

    @pytest.mark.parametrize(
        "innovation_bounds",
        [(-1., 1.), (-10., 2.5), (3., 6.)]
    )
    @pytest.mark.parametrize("low", [True, False])
    def test_one_step_bounds(
            self, low, model, innovation_bounds, variance_scale):
        # Assumes gen(.) is convex... Might fail for some models
        zb = np.array(innovation_bounds)

        def gen(z):
            next_variance, _ = model.one_step_generate(z, variance_scale)
            return next_variance
        low, high = model.one_step_bounds(variance_scale, *zb)
        if low:
            value = low

            def to_min(z): return gen(z)/variance_scale
        else:
            value = high

            def to_min(z): return -gen(z)/variance_scale
        expected = scipy.optimize.minimize(to_min, 0.5, bounds=(tuple(zb),))
        pt.assert_almost_equal(value, gen(expected.x), rtol=1e-6)

    def test_roots_scalar(self, model, variance_scale):
        roots = model.real_roots(variance_scale, variance_scale*1.1)[0]
        for z in roots:
            solved, _ = model.one_step_generate(z, variance_scale)
            pt.assert_almost_equal(solved, variance_scale*1.1)

    def test_valid_roots(self, model, variances, valid_one_step):
        roots = model.real_roots(variances, valid_one_step)[0]
        for z in roots:
            solved, _ = model.one_step_generate(z, variances)
            pt.assert_almost_equal(solved, valid_one_step)

    def test_left_right_roots_order(
            self, model, variances, valid_one_step):
        left, right = model.real_roots(variances, valid_one_step)[0]
        pt.assert_less(left, right)

    def test_invalid_roots(self, model, variances, invalid_one_step):
        roots = model.real_roots(variances, invalid_one_step)[0]
        for z in roots:
            pt.assert_valid(z)  # Still returns valid values (real part)

    def test_same_roots_at_singularity(
            self, model, variances, lowest_one_step):
        left, right = model.real_roots(variances, lowest_one_step)[0]
        pt.assert_almost_equal(left, right, rtol=1e-16)

    def test_roots_at_inf_is_pm_inf(self, model, variances, inf):
        roots = model.real_roots(variances, inf)[0]
        pt.assert_equal(roots[0], -np.inf, shape='broad')
        pt.assert_equal(roots[1], np.inf, shape='broad')

    def test_roots_derivatives_when_valid(
            self, model, variances, valid_one_step):
        der = model.real_roots(variances, valid_one_step)[1]

        def func(next_variances):
            _, z = model.real_roots(variances, next_variances)[0]
            return z
        pt.assert_derivative_at(der, valid_one_step,
                                function=func, rtol=1e-4)

    def test_roots_derivative_is_positive(
            self, model, variances, valid_one_step):
        der = model.real_roots(variances, valid_one_step)[1]
        pt.assert_greater(der, 0.0, shape='broad')

    def test_invalid_roots_derivatives(
            self, model, variances, invalid_one_step):
        der = model.real_roots(variances, invalid_one_step)[1]
        pt.assert_equal(der, 0., shape='broad')

    def test_roots_derivatives_is_zero_at_singularity(
            self, model, variances, lowest_one_step):
        der = model.real_roots(variances, lowest_one_step)[1]
        pt.assert_equal(der, 0., shape='broad')

    def test_roots_derivatives_at_inf_are_zero(
            self, model, variances, inf):
        der = model.real_roots(variances, inf)[1]
        pt.assert_equal(der, 0., shape='broad')


class TestVarianceQuantization():

    @pytest.fixture(params=[1, 4, 3, 10],
                    ids=['prev_size(1)',
                         'prev_size(4)',
                         'prev_size(3)',
                         'prev_size(10)',
                         ])
    def prev_size(self, request):
        return request.param

    @pytest.fixture
    def prev_proba(self, prev_size):
        p = np.random.uniform(
            low=0.05,
            high=1,
            size=prev_size)
        p /= np.sum(p)  # Normalize
        return p

    @pytest.fixture
    def prev_value(self, prev_size, variance_scale):
        np.random.seed(1234567)
        return np.random.uniform(
            low=0.75*variance_scale,
            high=1.25*variance_scale,
            size=prev_size)

    @pytest.fixture
    def zlb(self, prev_size):
        np.random.seed(58934345)
        return 0.25 * np.random.normal(size=prev_size)

    @pytest.fixture
    def zub(self, prev_size, zlb):
        np.random.seed(9234795)
        return zlb + np.fabs(np.random.normal(size=prev_size))

    @pytest.fixture
    def factory(self, model, prev_proba, prev_value, zlb, zub):
        return model.get_quant_factory(prev_proba, prev_value, zlb, zub)

    def test_search_bounds(self, factory):
        b = factory.get_search_bounds(3.)
        assert b.size == 2
        pt.assert_finite(b)
        assert b[0] < b[1]

    def test_invalid_innov_bounds_raises_value_error(
            self, model, prev_proba, prev_value, zlb, zub):
        with pytest.raises(ValueError):
            # Swapping upper and lower bounds!
            model.get_quant_factory(prev_proba, prev_value, zub, zlb)

    def test_size_mismatch_raises_value_error(self, model):
        with pytest.raises(ValueError):
            model.get_quant_factory(np.array([1., 2.]),
                                    np.array(1.),
                                    np.array(0.),
                                    np.array(1.))
        with pytest.raises(ValueError):
            model.get_quant_factory(np.array([1., 2.]),
                                    np.array([1., 2.]),
                                    np.array(0.),
                                    np.array(1.))
        with pytest.raises(ValueError):
            model.get_quant_factory(np.array([1., 2.]),
                                    np.array([1., 2.]),
                                    np.array([1., 2.]),
                                    np.array(1.))

    def test_unattainable_state_raises_unobserved(
            self, prev_size, prev_proba, model, prev_value):
        zlb = zub = np.full((prev_size,), 1.)
        with pytest.raises(qutil.UnobservedState):
            f = model.get_quant_factory(
                prev_proba, prev_value, zlb, zub)

    @pytest.fixture(params=[1, 2, 3, 5, 10],
                    ids=['size(1)',
                         'size(2)',
                         'size(3)',
                         'size(5)',
                         'size(10)'])
    def size(self, request):
        return request.param

    @pytest.fixture
    def value(self, size, factory):
        np.random.seed(907831247)
        bounds = factory.get_search_bounds(3.)
        value = np.random.uniform(
            low=bounds[0], high=bounds[1], size=size)
        value.sort()
        return value

    @pytest.fixture
    def quantizer(self, value, factory):
        return factory.make(value)

    def test_norm(self, quantizer, prev_proba, zlb, zub):
        z_proba = normcdf(zub) - normcdf(zlb)
        expected = np.sum(prev_proba*z_proba)
        pt.assert_almost_equal(expected, quantizer.norm)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_integral(
            self, model, prev_size, size, prev_value, zlb, zub,
            factory, quantizer, value, order):
        expected = np.empty((prev_size, size))
        for (pv, lz, hz, out) in zip(prev_value, zlb, zub, expected):
            out[:] = self._quantized_integral(
                model,
                pv, lz, hz,
                value,
                order=order)
        pt.assert_almost_equal(
            quantizer._integral[order],
            expected,
            rtol=1e-8, atol=1e-32)

    @pytest.mark.parametrize("order", [0, 1])
    def test_delta(self, factory, value, order):
        qutil._assert_valid_delta_at(factory, value, order, rtol=1e-4)

    ########################
    ## Helper Function(s) #
    ########################

    def _quantized_integral(self,
                            model, prev_value, zlb, zub, value, *, order=None):
        if order == 0:
            def integrand(z, H): return normpdf(z)
        elif order == 1:
            def integrand(z, H): return H*normpdf(z)
        elif order == 2:
            def integrand(z, H): return H**2.*normpdf(z)
        else:
            assert False  # should never happen...

        def do_quantized_integration(low_h, high_h):
            def function_to_integrate(innov):
                next_variance, _ = model.one_step_generate(innov, prev_value)
                return integrand(innov, next_variance)
            # Identify integration intervals
            (b, c), _ = model.real_roots(prev_value, low_h)
            if high_h == np.inf:
                # Crop integral to improve numerical accuracy
                CROP = 10.
                a = min(-CROP, b)
                d = max(CROP, c)
            else:
                (a, d), _ = model.real_roots(prev_value, high_h)
            a = max(a, zlb)
            b = min(b, zub)
            b = max(a, b)
            c = max(c, zlb)
            d = min(d, zub)
            d = max(c, d)
         # Perform integration by quadrature
            return (integrate.quad(function_to_integrate, a, b)[0]
                    + integrate.quad(function_to_integrate, c, d)[0])
        # Perform integration for each value
        vor = qutil.voronoi_1d(np.ravel(value), lb=0.)
        out = np.empty((value.size,))
        for j in range(value.size):
            out[j] = do_quantized_integration(vor[j], vor[j+1])
        return out


class TestRetSpec():

    @pytest.fixture(params=[(1,), (2, 3), (4, 5, 6), ])
    def shape(self, request):
        return request.param

    @pytest.fixture
    def retspec(self, model):
        return model.retspec

    def test_generate_reverts_to_filter(
            self, retspec, innovations, variances):
        returns = retspec.one_step_generate(innovations, variances)
        expected_innovations = retspec.one_step_filter(
            returns, variances)
        pt.assert_almost_equal(innovations, expected_innovations)

    def test_root_logprice_derivative(self, retspec, variances, returns):
        prev_logprice = np.zeros_like(returns)
        logprice = returns
        prev_variance = variances
        #
        der = retspec.roots(prev_logprice, prev_variance, logprice)[1]

        def func_to_derivate(lp):
            return retspec.roots(prev_logprice, prev_variance, lp)[0]
        pt.assert_derivative_at(der, logprice, function=func_to_derivate)

    @pytest.fixture(params=[1, 4, 3, 10],
                    ids=[
        'prev_size(1)',
        'prev_size(4)',
        'prev_size(3)',
        'prev_size(10)',
    ])
    def prev_size(self, request):
        return request.param

    @pytest.fixture
    def prev_logprice(self, prev_size, variance_scale):
        np.random.seed(8923476)
        z = np.random.normal(size=(prev_size,))
        vol = np.sqrt(variance_scale)
        return z*vol

    @pytest.fixture
    def prev_variance(self, prev_size, variance_scale):
        np.random.seed(1234567)
        return np.random.uniform(
            low=0.75*variance_scale,
            high=1.25*variance_scale,
            size=(prev_size,))
        return v

    @pytest.fixture
    def prev_proba(self, prev_size):
        np.random.seed(8237123)
        p = np.random.uniform(
            low=0.05,
            high=1,
            size=prev_size)
        p /= np.sum(p)  # Normalize
        return p

    @pytest.fixture
    def factory(self, retspec, prev_proba, prev_logprice, prev_variance):
        return retspec.get_quant_factory(
            prev_proba, prev_logprice, prev_variance)

    @pytest.fixture(params=[1, 2, 3, 5],
                    ids=['size(1)',
                         'size(2)',
                         'size(3)',
                         'size(5)'])
    def size(self, request):
        return request.param

    @pytest.fixture
    def value(self, factory, size):
        np.random.seed(907831247)
        bounds = factory.get_search_bounds(3.)
        value = np.random.uniform(low=bounds[0], high=bounds[1], size=size)
        value.sort()
        return value

    @pytest.fixture
    def quantizer(self, factory, value):
        return factory.make(value)

    def test_size_mismatch_raises_value_error(self, retspec):
        with pytest.raises(ValueError):
            retspec.get_quant_factory(
                np.empty((5, 3)), np.empty((10, 1)), np.empty((1, 10)))

    def test_search_bounds(self, factory):
        b = factory.get_search_bounds(3.)
        assert b.size == 2
        pt.assert_finite(b)
        assert b[0] < b[1]

    # TODO: actually test search bounds (with simulations??)

    def test_roots(
            self, retspec, prev_variance, prev_logprice, quantizer):
        roots = quantizer.get_roots()[0]
        returns = retspec.one_step_generate(
            roots, prev_variance[:, np.newaxis])
        values = prev_logprice[:, np.newaxis] + returns
        pt.assert_almost_equal(values, quantizer.voronoi, shape='broad')

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_integral(self, model, prev_size, size,
                      prev_logprice, prev_variance, quantizer, order):
        # Compute expected integral for all previous values
        expected = np.empty((prev_size, size))
        for (plp, pv, out) in zip(prev_logprice, prev_variance, expected):
            out[:] = self._quantized_integral(
                model.retspec,
                plp,
                pv,
                quantizer.value,
                order=order)
        pt.assert_almost_equal(
            quantizer._integral[order],
            expected,
            rtol=1e-8, atol=1e-32)

    @pytest.mark.parametrize("order", [0, 1])
    def test_delta(self, factory, value, order):
        qutil._assert_valid_delta_at(factory, value, order, rtol=1e-5)

    #######################
    ## Helper Function(s) #
    #######################

    def _quantized_integral(
            self, retspec, prev_logprice, prev_variance, value,
            *, order=None):
        if order == 0:
            def integrand(z, X): return normpdf(z)
        elif order == 1:
            def integrand(z, X): return X*normpdf(z)
        elif order == 2:
            def integrand(z, X): return X**2.*normpdf(z)
        else:
            assert False  # should never happen...

        def do_quantized_integration(bounds):
            def function_to_integrate(innov):
                r = retspec.one_step_generate(innov, prev_variance)
                next_logprice = prev_logprice + r
                return integrand(innov, next_logprice)
            # Identify integration intervals
            ret = bounds-prev_logprice
            z_bounds = retspec.one_step_filter(ret, prev_variance)
            z_bounds = np.clip(z_bounds, -10., 10.)
            # Perform integration by quadrature
            return integrate.quad(function_to_integrate, *z_bounds)[0]
        # Perform integration for each value
        vor = qutil.voronoi_1d(np.ravel(value))
        out = np.empty((value.size,))
        for j in range(value.size):
            out[j] = do_quantized_integration(vor[j:j+2])
        return out
