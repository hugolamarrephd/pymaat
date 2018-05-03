import pytest
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing as pt
from pymaat.quantutil import voronoi_1d, _assert_valid_delta_at
from pymaat.nputil import diag_view

import pymaat.garch.varquant as vq


@pytest.fixture(params=[(1,), (4, 2), (3, 3), (1, 10)],
                ids=['prev_shape(1)',
                     'prev_shape(4,2)',
                     'prev_shape(3,3)',
                     'prev_shape(1,10)'
                     ])
def prev_shape(request):
    return request.param


@pytest.fixture
def prev_proba(prev_shape):
    p = np.random.uniform(
        low=0.05,
        high=1,
        size=prev_shape)
    p /= np.sum(p)  # Normalize
    return p


@pytest.fixture
def prev_value(prev_shape, variance_scale):
    np.random.seed(1234567)
    v = np.random.uniform(
        low=0.75*variance_scale,
        high=1.25*variance_scale,
        size=prev_shape)
    return v


@pytest.fixture
def low_z(prev_shape):
    np.random.seed(58934345)
    return 0.25 * np.random.normal(size=prev_shape)


@pytest.fixture
def high_z(prev_shape, low_z):
    np.random.seed(9234795)
    return low_z + np.fabs(np.random.normal(size=prev_shape))


@pytest.fixture
def factory(prev_proba, model, prev_value, low_z, high_z):
    return vq._Factory(prev_proba, model, prev_value, low_z, high_z)


class TestCore:

    @pytest.fixture
    def core(self, model, variance_scale):
        return vq.Core(model, variance_scale, size=10, nper=21, verbose=True)

    def test_optimize(self, core):
        core.optimize()


class TestFactory:

    def test_model(self, model, factory):
        assert model is factory.model

    def test_innov_bounds(self, low_z, high_z, factory):
        pt.assert_almost_equal(factory.low_z, low_z)
        pt.assert_almost_equal(factory.high_z, high_z)

    def test_prev_value(self, prev_value, factory):
        pt.assert_almost_equal(factory.prev_value, prev_value)

    def test_prev_proba_when_no_z_proba(
            self, factory, prev_proba, low_z, high_z):
        # Normalize previous probability...
        z_proba = norm.cdf(high_z) - norm.cdf(low_z)
        expected = prev_proba / np.sum(prev_proba*z_proba)
        pt.assert_almost_equal(expected, factory.prev_proba)

    def test_search_bounds(self, factory):
        b = factory.get_search_bounds(3.)
        assert b[0].size == 1
        assert b[1].size == 1
        pt.assert_finite(b[0])
        pt.assert_finite(b[1])
        pt.assert_less(b[0], b[1])

    def test_make_from_valid_value(self, factory):
        np.random.seed(907831247)
        bounds = factory.get_search_bounds(3.)
        value = np.random.uniform(
            low=bounds[0], high=bounds[1], size=10)
        value.sort()  # Must be sorted!
        quant = factory.make(value)
        assert isinstance(quant, vq._Quantizer)


    @pytest.mark.parametrize("wrong", [np.nan, np.inf, -1., 2.])
    def test_invalid_prev_proba_raises_value_error(
            self, prev_proba, model, prev_value, wrong):
        prev_proba[0] = wrong
        with pytest.raises(ValueError):
            vq._Factory(prev_proba, model, prev_value)

    @pytest.mark.parametrize("wrong", [np.nan, np.inf, -1.])
    def test_invalid_prev_value_raises_value_error(
            self, prev_proba, model, prev_value, wrong):
        prev_value[0] = wrong
        with pytest.raises(ValueError):
            vq._Factory(prev_proba, model, prev_value)

    def test_invalid_innov_bounds_raises_value_error(
            self, prev_shape, prev_proba, model, prev_value,
            low_z, high_z):
        with pytest.raises(ValueError):
            vq._Factory(prev_proba, model, prev_value, high_z, low_z)

    def test_prev_value_shape_mismatch_raises_value_error(self, model):
        with pytest.raises(ValueError):
            vq._Factory(np.array([1., 2.]), model, np.array(1.))

    def test_innov_bounds_shape_mismatch_raises_value_error(self, model):
        with pytest.raises(ValueError):
            vq._Factory(np.array([1., 2.]), model,
                            np.array([1., 2.]),
                            low_z=np.array(1.),
                            high_z=np.array(1.))
        with pytest.raises(ValueError):
            vq._Factory(np.array([1., 2.]), model,
                            np.array([1., 2.]),
                            low_z=np.array([1., 2.]),
                            high_z=np.array(1.))
        with pytest.raises(ValueError):
            vq._Factory(np.array([1., 2.]), model,
                            np.array([1., 2.]),
                            low_z=np.array(1.),
                            high_z=np.array([1., 2.]))

    def test_z_proba_shape_mismatch_raises_value_error(self, model):
        with pytest.raises(ValueError):
            vq._Factory(np.array([1., 2.]), model,
                            np.array([1., 2.]),
                            low_z=np.array([1., 2.]),
                            high_z=np.array([1., 2.]),
                            z_proba=np.array(1.))

    def test_default_when_no_innov_bounds(
            self, prev_proba, model, prev_value):
        f = vq._Factory(prev_proba, model, prev_value)
        pt.assert_all(f.low_z == -np.inf)
        pt.assert_all(f.high_z == np.inf)

    def test_unchanged_prev_proba_when_no_innov_bounds(
            self, prev_proba, model, prev_value):
        f = vq._Factory(prev_proba, model, prev_value)
        pt.assert_almost_equal(prev_proba, f.prev_proba)

    def test_prev_proba_when_z_proba(
            self, prev_proba, model, prev_value, low_z, high_z):
        # Generate some transition probabilities
        np.random.seed(1234567)
        z_proba = np.random.uniform(
            low=0.05,
            high=0.95,
            size=prev_proba.shape)
        # Instantiate factory with transition probabilities
        f = vq._Factory(
            prev_proba, model, prev_value, low_z, high_z, z_proba)
        # Expected previous probability
        expected = prev_proba / np.sum(prev_proba*z_proba)
        pt.assert_almost_equal(expected, f.prev_proba)

    def test_unattainable_state_raises_value_error(
            self, prev_shape, prev_proba, model, prev_value):
        low_z = high_z = np.full(prev_shape, 1.)
        with pytest.raises(ValueError):
            f = vq._Factory(prev_proba, model, prev_value, low_z, high_z)


class TestQuantizer:

    @pytest.fixture(params=[1, 2, 3, 5],
                    ids=['size(1)',
                         'size(2)',
                         'size(3)',
                         'size(5)'])
    def size(self, request):
        return request.param

    @pytest.fixture
    def quantizer(self, size, factory):
        np.random.seed(907831247)
        bounds = factory.get_search_bounds(3.)
        value = np.random.uniform(
            low=bounds[0], high=bounds[1], size=size)
        value.sort()
        return factory.make(value)

    def test_model(self, model, quantizer):
        assert model is quantizer.model

    def test_prev_value(self, prev_shape, prev_value, quantizer):
        assert quantizer.prev_value.shape == prev_shape + (1,)
        pt.assert_almost_equal(quantizer.prev_value[..., 0],
                               prev_value)

    def test_low_z(self, prev_shape, low_z, quantizer):
        assert quantizer.low_z.shape == prev_shape + (1,)
        pt.assert_almost_equal(quantizer.low_z[..., 0],
                               low_z)

    def test_high_z(self, prev_shape, high_z, quantizer):
        assert quantizer.high_z.shape == prev_shape + (1,)
        pt.assert_almost_equal(quantizer.high_z[..., 0],
                               high_z)

    @pytest.mark.parametrize("right", [False, True])
    def test_roots_shape(self, factory, quantizer, right):
        expected_shape = factory.prev_proba.shape + (quantizer.size+1,)
        assert quantizer._roots[right].shape == expected_shape

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_integral(self, factory, quantizer, order):
        # Compute expected integral for all previous values
        it = np.nditer(factory.prev_value, flags=['multi_index'])
        expected = np.empty(factory.prev_value.shape + (quantizer.size,))
        while not it.finished:
            expected[it.multi_index] = _quantized_integral(
                factory.model,
                factory.prev_value[it.multi_index],
                factory.low_z[it.multi_index],
                factory.high_z[it.multi_index],
                quantizer.value,
                order=order)
            it.iternext()
        pt.assert_almost_equal(
            quantizer._integral[order],
            expected,
            rtol=1e-8, atol=1e-32)

    @pytest.mark.parametrize("order", [0, 1])
    def test_delta(self, factory, size, order):
        np.random.seed(1234423)
        bounds = factory.get_search_bounds(3.)
        value = np.random.uniform(
            low=bounds[0], high=bounds[1], size=size)
        value.sort()
        _assert_valid_delta_at(factory, value, order)

########################
## Helper Function(s) #
########################

def _quantized_integral(
        model, prev_value, low_z, high_z, value, *, order=None):
    if order == 0:
        def integrand(z,H): return norm.pdf(z)
    elif order == 1:
        def integrand(z,H): return H*norm.pdf(z)
    elif order == 2:
        def integrand(z,H): return H**2.*norm.pdf(z)
    else:
        assert False  # should never happen...

    def do_quantized_integration(low_h, high_h):
        def function_to_integrate(innov):
            next_variance, _ = model.one_step_generate(innov, prev_value)
            return integrand(innov, next_variance)
        # Identify integration intervals
        (b, c) = model.real_roots(prev_value, low_h)
        if high_h == np.inf:
            # Crop integral to improve numerical accuracy
            CROP = 10.
            a = min(-CROP, b)
            d = max(CROP, c)
        else:
            (a, d) = model.real_roots(prev_value, high_h)
        a = max(a, low_z)
        b = min(b, high_z)
        b = max(a, b)
        c = max(c, low_z)
        d = min(d, high_z)
        d = max(c, d)
        # Perform integration by quadrature
        return (integrate.quad(function_to_integrate, a, b)[0]
                + integrate.quad(function_to_integrate, c, d)[0])
    # Perform integration for each value
    vor = voronoi_1d(np.ravel(value), lb=0.)
    out = np.empty((value.size,))
    for j in range(value.size):
        out[j] = do_quantized_integration(vor[j], vor[j+1])
    return out
