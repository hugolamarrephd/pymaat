import pytest
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing as pt
from pymaat.quantutil import voronoi_1d, _assert_valid_delta_at
from pymaat.nputil import diag_view

import pymaat.garch.varquant as vq
import pymaat.quantutil as qutil


@pytest.fixture(params=[1,4,3,10,1000],
                ids=['prev_size(1)',
                     'prev_size(4)',
                     'prev_size(3)',
                     'prev_size(10)',
                     'prev_size(1000)'
                     ])
def prev_size(request):
    return request.param


@pytest.fixture
def prev_proba(prev_size):
    p = np.random.uniform(
        low=0.05,
        high=1,
        size=prev_size)
    p /= np.sum(p)  # Normalize
    return p


@pytest.fixture
def prev_value(prev_size, variance_scale):
    np.random.seed(1234567)
    v = np.random.uniform(
        low=0.75*variance_scale,
        high=1.25*variance_scale,
        size=prev_size)
    return v


@pytest.fixture
def low_z(prev_size):
    np.random.seed(58934345)
    return 0.25 * np.random.normal(size=prev_size)


@pytest.fixture
def high_z(prev_size, low_z):
    np.random.seed(9234795)
    return low_z + np.fabs(np.random.normal(size=prev_size))


@pytest.fixture
def factory(model, prev_proba, prev_value, low_z, high_z):
    return vq.Factory(model, prev_proba, prev_value, low_z, high_z)


class TestFactory:

    def test_model(self, model, factory):
        assert model is factory.model

    def test_innov_bounds(self, low_z, high_z, factory):
        pt.assert_almost_equal(factory.low_z, low_z)
        pt.assert_almost_equal(factory.high_z, high_z)

    def test_prev_value(self, prev_value, factory):
        pt.assert_almost_equal(factory.prev_value, prev_value)

    def test_norm(self, factory, prev_proba, low_z, high_z):
        z_proba = norm.cdf(high_z) - norm.cdf(low_z)
        expected = np.sum(prev_proba*z_proba)
        pt.assert_almost_equal(expected, factory.norm)

    def test_search_bounds(self, factory):
        b = factory.get_search_bounds(3.)
        assert b.size==2
        pt.assert_finite(b)
        assert b[0] < b[1]

        # TODO actually test search bounds

    def test_make_from_valid_value(self, factory):
        np.random.seed(907831247)
        bounds = factory.get_search_bounds(3.)
        value = np.random.uniform(
            low=bounds[0], high=bounds[1], size=10)
        value.sort()  # Must be sorted!
        quant = factory.make(value)
        assert isinstance(quant, vq.Quantizer)

    def test_invalid_innov_bounds_raises_value_error(
            self, prev_proba, model, prev_value,
            low_z, high_z):
        with pytest.raises(ValueError):
            vq.Factory(model, prev_proba, prev_value, high_z, low_z)

    def test_innov_bounds_shape_mismatch_raises_value_error(self, model):
        with pytest.raises(ValueError):
            vq.Factory(model, np.array([1., 2.]),
                            np.array([1., 2.]),
                            low_z=np.array(1.),
                            high_z=np.array(1.))
        with pytest.raises(ValueError):
            vq.Factory(model, np.array([1., 2.]),
                            np.array([1., 2.]),
                            low_z=np.array([1., 2.]),
                            high_z=np.array(1.))
        with pytest.raises(ValueError):
            vq.Factory(model, np.array([1., 2.]),
                            np.array([1., 2.]),
                            low_z=np.array(1.),
                            high_z=np.array([1., 2.]))

    def test_z_proba_shape_mismatch_raises_value_error(self, model):
        with pytest.raises(ValueError):
            vq.Factory(model, np.array([1., 2.]),
                            np.array([1., 2.]),
                            low_z=np.array([1., 2.]),
                            high_z=np.array([1., 2.]),
                            z_proba=np.array(1.))

    def test_default_when_no_innov_bounds(
            self, prev_proba, model, prev_value):
        f = vq.Factory(model, prev_proba, prev_value)
        pt.assert_all(f.low_z == -np.inf)
        pt.assert_all(f.high_z == np.inf)

    def test_unchanged_prev_proba_when_no_innov_bounds(
            self, prev_proba, model, prev_value):
        f = vq.Factory(model, prev_proba, prev_value)
        pt.assert_almost_equal(prev_proba, f.prev_proba)

    def test_unattainable_state_raises_runtime(
            self, prev_size, prev_proba, model, prev_value):
        low_z = high_z = np.full((prev_size,), 1.)
        with pytest.raises(qutil.UnobservedState):
            f = vq.Factory(model, prev_proba, prev_value, low_z, high_z)


class TestQuantizer:

    @pytest.fixture(params=[1, 2, 3, 5, 10, 1000],
                    ids=['size(1)',
                         'size(2)',
                         'size(3)',
                         'size(5)',
                         'size(10)',
                         'size(1000)'])
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

    def test_model(self, model, quantizer):
        assert model is quantizer.model

    def test_prev_value(self, prev_size, prev_value, quantizer):
        assert quantizer.prev_value.shape ==  (prev_size,1)
        pt.assert_almost_equal(quantizer.prev_value[..., 0],
                               prev_value)

    def test_low_z(self, prev_size, low_z, quantizer):
        assert quantizer.low_z.shape ==  (prev_size,1)
        pt.assert_almost_equal(quantizer.low_z[..., 0],
                               low_z)

    def test_high_z(self, prev_size, high_z, quantizer):
        assert quantizer.high_z.shape ==  (prev_size,1)
        pt.assert_almost_equal(quantizer.high_z[..., 0],
                               high_z)

    @pytest.mark.parametrize("right", [False, True])
    def test_roots_shape(self, prev_size, size, factory, quantizer, right):
        expected_shape = (prev_size, size+1)
        assert quantizer._roots[right].shape == expected_shape


    def test_conditional_expectation_stays_sorted(self, quantizer):
        if np.all(quantizer.probability>1e-16):
            assert np.all(
                    np.diff(quantizer.conditional_expectation)>0.
                    )

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_integral(
            self, prev_size, size, prev_value, low_z, high_z,
            factory, quantizer, order):
        expected = np.empty((prev_size, size))
        for (pv, lz, hz, out) in zip(prev_value, low_z, high_z, expected):
            out[:] = _quantized_integral(
                factory.model,
                pv, lz, hz,
                quantizer.value,
                order=order)
        pt.assert_almost_equal(
            quantizer._integral[order],
            expected,
            rtol=1e-8, atol=1e-32)

    @pytest.mark.parametrize("order", [0, 1])
    def test_delta(self, factory, value, order):
        _assert_valid_delta_at(factory, value, order, rtol=1e-4)

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
