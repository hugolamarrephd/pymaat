import pytest
import numpy as np
from numpy.core.numeric import isclose
import scipy.optimize
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing as pt
from pymaat.quantutil import voronoi_1d, inv_voronoi_1d
from pymaat.quantutil import Optimizer
from pymaat.nputil import diag_view

from pymaat.garch.varquant import Quantization
from pymaat.garch.varquant import _QuantizerFactory
from pymaat.garch.varquant import _Quantizer
from pymaat.quantutil import _Unconstrained1D


@pytest.fixture(params=[1, 4],
                ids=['prev_size(1)',
                     'prev_size(4)'])
def prev_size(request):
    return request.param


@pytest.fixture(params=[2, 3],
                ids=['size(2)',
                     'size(3)'])
@pytest.fixture
def size(request):
    return request.param


@pytest.fixture
def previous(prev_size, variance_scale):
    from_ = 0.75 * variance_scale
    to_ = 1.25 * variance_scale
    value = np.linspace(from_, to_, num=prev_size)
    probability = np.ones((prev_size,))/prev_size
    return Quantization(
        value=value, probability=probability)


@pytest.fixture
def factory(model, size, previous):
    return _QuantizerFactory.make_from_prev(
            model, size, False, previous)


@pytest.fixture
def unc_factory(model, size, previous):
    return _QuantizerFactory.make_from_prev(
            model, size, True, previous)

@pytest.fixture
def shape(size):
    return (size,)


@pytest.fixture
def x(random_normal):
    return random_normal


@pytest.fixture
def value(factory, previous, size):
    from_ = max(factory.min_value, 0.75*np.min(previous.value))
    to_ = 1.25*np.max(previous.value)
    return np.linspace(from_, to_, num=size)


@pytest.fixture
def optimizer(unc_factory):
    return Optimizer(unc_factory)


@pytest.mark.slow
def test_optimize(optimizer):
    base_quant = optimizer.quick_optimize()
    robust_quant = optimizer.optimize()
    brute_quant = optimizer.brute_optimize()
    # Same distortion
    pt.assert_almost_equal(
        base_quant.distortion,
        brute_quant.distortion,
        rtol=1e-4)
    pt.assert_almost_equal(
        robust_quant.distortion,
        brute_quant.distortion,
        rtol=1e-4)
    # Same solution
    pt.assert_almost_equal(
        base_quant.value, brute_quant.value, rtol=1e-2)
    pt.assert_almost_equal(
        robust_quant.value, brute_quant.value, rtol=1e-2)

class TestQuantization:

    def test_make_stub_from_valid_value_and_proba(self, value):
        proba = np.ones_like(value)/value.size
        quant = Quantization(value, proba)
        pt.assert_almost_equal(quant.value, value)
        pt.assert_almost_equal(quant.probability, proba)
        assert quant.size == value.size
        assert quant.shape == value.shape

    @pytest.mark.parametrize("wrong", [np.nan, np.inf])
    def test_make_stub_from_invalid_value_raises_value_error(
            self, value, wrong):
        invalid_value = np.append(value, [wrong])
        proba = np.ones_like(invalid_value)/invalid_value.size
        with pytest.raises(ValueError):
            quant = Quantization(invalid_value, proba)

    def test_make_stub_from_negative_raises_value_error(self, value):
        invalid_value = np.append(value, [-1.])
        proba = np.ones_like(invalid_value)/invalid_value.size
        with pytest.raises(ValueError):
            quant = Quantization(invalid_value, proba)

    @pytest.mark.parametrize("wrong", [2., -1.])
    def test_make_stub_from_invalid_proba_raises_value_error(
            self, value, wrong):
        proba = np.ones_like(value)/value.size
        proba[0] = wrong
        with pytest.raises(ValueError):
            quant = Quantization(value, proba)

class TestFactory:

    def test__init__(self, model, factory, size, previous):
        assert factory.model is model
        assert factory.shape == (size,)
        assert factory.size == size

    def test_make_unconstrained(self, unc_factory, x):
        quant = unc_factory.make(x)
        quant = unc_factory.make(x)  # Query cache?
        assert isinstance(quant, _Unconstrained1D)
        pt.assert_almost_equal(quant.x, x, atol=1e-16)

    def test_make_from_valid_value(self, factory, value):
        quant = factory.make(value)
        quant = factory.make(value)  # Query cache?
        assert isinstance(quant, _Quantizer)
        assert not isinstance(quant, _Unconstrained1D)
        pt.assert_almost_equal(quant.value, value)

    def test_min_value(self, factory):
        expected = factory.model.get_lowest_one_step_variance(
            np.amin(factory.prev_value))
        pt.assert_almost_equal(factory.min_value, expected)


class TestQuantizer:

    DIST_RTOL = 1e-6
    GRADIENT_RTOL = 1e-6
    HESSIAN_RTOL = 1e-6

    @pytest.fixture
    def valid_quantizer(self, factory, value):
        return factory.make(value)

    # Special cases

    @pytest.fixture
    def singular_quantizer(self, factory, valid_quantizer):
        first = None
        if factory.singularities.size == 1:
            first = 0.5*factory.singularities[0]

        new_values = inv_voronoi_1d(
            factory.singularities,
            with_bounds=False,
            first_quantizer=first)

        until = min(factory.size, new_values.size)
        value = valid_quantizer.value.copy()
        value[:until] = new_values[:until]
        value.sort()

        quant = factory.make(value)

        # Make sure Voronoi was not rounded
        for i in range(quant.voronoi.size):
            id_ = isclose(
                quant.voronoi[i], factory.singularities, rtol=1e-15)
            if any(id_):
                quant.voronoi[i] = factory.singularities[id_]

        return quant

    @pytest.fixture
    def flat_quantizer(self, factory, valid_quantizer):
        value = valid_quantizer.value.copy()
        value[0] = 0.5 * factory.min_value
        if factory.size > 1:
            value[1] = factory.min_value
        return factory.make(value)

    # Tests

    def test__init__invalid_parent_raises_value_error(self, valid_quantizer):
        with pytest.raises(ValueError):
            _Quantizer('invalid_parent', valid_quantizer)

    def test__init__invalid_value_ndim_raises_value_error(self, factory):
        invalid_value = np.array([[0., 1.], [2., 3.]])
        with pytest.raises(ValueError):
            _Quantizer(factory, invalid_value)

    def test__init__invalid_value_shape_raises_value_error(
            self, factory, valid_quantizer):
        invalid_value = np.append(valid_quantizer.value, [1.])
        with pytest.raises(ValueError):
            _Quantizer(factory, invalid_value)

    def test__init__value_nan_raises_value_error(
            self, factory, valid_quantizer):
        invalid_value = valid_quantizer.value.copy()
        invalid_value[0] = np.nan
        with pytest.raises(ValueError):
            _Quantizer(factory, invalid_value)

    def test__init__value_infinite_raises_value_error(
            self, factory, valid_quantizer):
        invalid_value = valid_quantizer.value.copy()
        invalid_value[0] = np.inf
        with pytest.raises(ValueError):
            _Quantizer(factory, invalid_value)

    def test__init__non_increasing_value_raises_value_error(
            self, factory, valid_quantizer):
        if factory.size > 1:
            invalid_value = valid_quantizer.value.copy()
            invalid_value[[0, 1]] = valid_quantizer.value[[1, 0]]
            with pytest.raises(ValueError):
                _Quantizer(factory, invalid_value)

    def test_value_has_no_nan(self, valid_quantizer):
        pt.assert_valid(valid_quantizer.value)

    def test_probability_shape(self, valid_quantizer):
        expected = valid_quantizer.parent.shape
        value = valid_quantizer.probability.shape
        assert expected == value

    def test_probability_sums_to_one_and_strictly_positive(
            self, valid_quantizer):
        pt.assert_greater(valid_quantizer.probability, 0.0,
                          shape='broad')
        pt.assert_almost_equal(np.sum(valid_quantizer.probability),
                               1., rtol=1e-6, shape='broad')

    def test_probability(self, previous, valid_quantizer):
        value = valid_quantizer.probability
        expected = previous.probability.dot(
            valid_quantizer.transition_probability)
        pt.assert_almost_equal(value, expected, rtol=1e-6)

    def test_transition_probability_shape(self,
                                          previous, valid_quantizer):
        expected = (previous.value.size, valid_quantizer.parent.size)
        value = valid_quantizer.transition_probability.shape
        assert expected == value

    def test_transition_probability_sums_to_one_and_non_negative(
            self, valid_quantizer):
        # Non negative...
        pt.assert_greater_equal(valid_quantizer.transition_probability,
                                0.0, shape='broad')
        # Sums to one...
        marginal_probability = np.sum(
            valid_quantizer.transition_probability, axis=1)
        pt.assert_almost_equal(marginal_probability, 1.0,
                               shape='broad', rtol=1e-6)

    def test_transition_probability(self, previous, valid_quantizer):
        value = valid_quantizer.transition_probability
        expected_value = valid_quantizer._integral[0]
        pt.assert_almost_equal(value, expected_value, rtol=1e-12)

    def test_distortion(self, previous, valid_quantizer):
        # See below for _distortion_elements test
        distortion = np.sum(valid_quantizer._distortion_elements, axis=1)
        expected_value = previous.probability.dot(distortion)
        value = valid_quantizer.distortion
        pt.assert_almost_equal(expected_value, value, rtol=self.DIST_RTOL)

    def test_gradient(self, factory, valid_quantizer):
        pt.assert_gradient_at(valid_quantizer.gradient,
                              valid_quantizer.value,
                              function=lambda x: factory.make(x).distortion,
                              rtol=self.GRADIENT_RTOL)

    def test_gradient_at_flat(self, factory, flat_quantizer):
        if factory.size > 1:
            assert flat_quantizer.gradient[0] == 0.

    def test_hessian(self, factory, valid_quantizer):
        pt.assert_hessian_at(valid_quantizer.hessian,
                             valid_quantizer.value,
                             gradient=lambda x: factory.make(x).gradient,
                             rtol=self.HESSIAN_RTOL)

    def test_hessian_is_symmetric(self, valid_quantizer):
        pt.assert_equal(valid_quantizer.hessian,
                        np.transpose(valid_quantizer.hessian))

    # Testing distortion...

    def test_distortion_elements(self, model, previous, valid_quantizer):
        expected_value = numerical_quantized_integral(
            model, previous, valid_quantizer.value, distortion=True)
        value = valid_quantizer._distortion_elements
        pt.assert_almost_equal(value, expected_value, rtol=1e-6)

    def test_distortion_elements_at_flat(self, factory, flat_quantizer):
        if factory.size > 1:
            pt.assert_equal(
                flat_quantizer._distortion_elements[:, 0], 0.,
                shape='broad')

    # Testing quantized integrals...

    @pytest.mark.parametrize("order_rtol", [(0, 1e-8), (1, 1e-6), (2, 1e-3)])
    def test_integral(self, model, previous, valid_quantizer, order_rtol):
        order, rtol = order_rtol
        expected_value = numerical_quantized_integral(
            model, previous, valid_quantizer.value, order=order)
        value = valid_quantizer._integral[order]
        pt.assert_almost_equal(value, expected_value, rtol=rtol, atol=1e-32)

    def test_delta_at_singularity_is_zero(self, previous, factory,
                                          singular_quantizer):
        voronoi = singular_quantizer.voronoi
        delta = singular_quantizer._delta
        for (ii, sing) in enumerate(factory.singularities):
            id_ = isclose(sing, voronoi)
            if np.any(id_):
                assert np.sum(id_) == 1
                pt.assert_all(delta[0][ii, id_] == 0.)
                pt.assert_all(delta[1][ii, id_] == 0.)

    @pytest.mark.parametrize("right", [False, True])
    def test_roots_shape(self, valid_quantizer, right):
        expected_shape = (valid_quantizer.parent.prev_size,
                          valid_quantizer.parent.size+1)
        assert valid_quantizer._roots[right].shape == expected_shape

    # Voronoi

    def test_voronoi(self, valid_quantizer):
        v = valid_quantizer.voronoi
        assert v[0] == 0.
        pt.assert_almost_equal(v, voronoi_1d(valid_quantizer.value, lb=0))


class TestUnconstrained():

    RTOL_GRADIENT = 1e-6
    RTOL_HESSIAN = 1e-6

    @pytest.fixture
    def unc(self, unc_factory, x):
        return unc_factory.make(x)

    def test_value_is_in_space(self, unc):
        pt.assert_finite(unc.quantizer.value)
        # strictly increasing
        pt.assert_greater(np.diff(unc.quantizer.value), 0.0, shape='broad')
        pt.assert_greater(
            unc.quantizer.value, unc.parent.min_value, shape='broad')

    def test_gradient(self, unc_factory, x, unc):
        pt.assert_gradient_at(
            unc.gradient, x, rtol=self.RTOL_GRADIENT,
            function=lambda x_: unc_factory.make(x_).distortion
        )

    def test_hessian(self, unc_factory, x, unc):
        pt.assert_hessian_at(
            unc.hessian, x, rtol=self.RTOL_HESSIAN,
            gradient=lambda x_: unc_factory.make(x_).gradient
        )

    def test_jacobian(self, unc_factory, x, unc):
        pt.assert_jacobian_at(
            unc._jacobian,
            x, rtol=1e-8,
            function=lambda x_: unc_factory.make(x_).quantizer.value
        )

    def test_scaled_exponential_of_x(self, unc_factory, unc):
        expected_value = np.exp(unc.x) * unc_factory.min_value / (unc.size+1)
        pt.assert_almost_equal(unc._scaled_expx, expected_value, rtol=1e-12)

    def test_changed_variable(self, unc_factory, unc):
        expected_value = unc_factory.min_value + np.cumsum(unc._scaled_expx)
        pt.assert_almost_equal(unc.quantizer.value,
                expected_value, rtol=1e-12)

    def test_inv_change_variable_reverts(self, unc_factory, unc):
        x = unc_factory.inv_change_of_variable(unc.quantizer.value)
        pt.assert_almost_equal(x, unc.x, rtol=1e-6)


#######################
# Helper Function(s) #
#######################


def numerical_quantized_integral(model, previous, value, *,
                                 order=None,
                                 distortion=False):
    """
    This helper function numerically estimates (via quadrature)
    an integral (specified by order or distortion)
    over quantized tiles
    """
    # Setup integrand
    if order is not None:
        if order == 0:
            def integrand(z, H, h): return norm.pdf(z)
        elif order == 1:
            def integrand(z, H, h): return H * norm.pdf(z)
        elif order == 2:
            def integrand(z, H, h): return H**2. * norm.pdf(z)
        else:
            assert False  # Should never happen...
    elif distortion:
        def integrand(z, H, h): return (H-h)**2. * norm.pdf(z)
    else:
        assert False  # Should never happen...

    def do_quantized_integration(lb, ub, prev_variance, value):
        assert prev_variance.size == 1
        assert value.size == 1
        # Define function to integrate

        def function_to_integrate(innov):
            assert isinstance(innov, float)
            # 0 if innovation is between tile bounds else integrate
            next_variance, _ = model.one_step_generate(
                innov, prev_variance)
            # assert (next_variance >= lb and next_variance <= ub)
            return integrand(innov, next_variance, value)

        # Identify integration intervals
        (b, c) = model.real_roots(prev_variance, lb)
        if ub == np.inf:
            # Crop integral to improve numerical accuracy
            CROP = 10.
            a = min(-CROP, b)
            d = max(CROP, c)
        else:
            (a, d) = model.real_roots(prev_variance, ub)
        # Perform integration by quadrature
        return (integrate.quad(function_to_integrate, a, b)[0]
                + integrate.quad(function_to_integrate, c, d)[0])

    # Perform Integration
    voronoi = voronoi_1d(value, lb=0.)
    I = np.empty((previous.value.size, value.size))
    for (i, prev_h) in enumerate(previous.value):
        for (j, (lb, ub, h)) in enumerate(
                zip(voronoi[:-1], voronoi[1:], value)):
            I[i, j] = do_quantized_integration(lb, ub, prev_h, h)
    return I
