import pytest
import numpy as np
from numpy.core.numeric import isclose
import scipy.optimize
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing as pt
from pymaat.mathutil import voronoi_1d, inv_voronoi_1d
from pymaat.nputil import diag_view

from pymaat.garch.varquant import _QuantizerFactory, \
    _Quantizer, _Unconstrained, MarginalVariance


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
    return _QuantizerFactory.make_stub(
        value=value, probability=probability)


@pytest.fixture
def factory(model, size, previous):
    return _QuantizerFactory(model, size, previous)


@pytest.fixture
def shape(size):
    return (size,)


@pytest.fixture
def x(random_normal):
    return random_normal


@pytest.fixture
def value(factory, previous, size):
    from_ = max(factory.min_variance, 0.75*np.min(previous.value))
    to_ = 1.25*np.max(previous.value)
    return np.linspace(from_, to_, num=size)


class TestFactory:

    @pytest.fixture(params=[
        (100, 100, 1.81753634548327519244801912973536e-13),
        (1000, 1000, 1.82758229081871336437695944554397e-15),
        ],
        ids=[
            "100x100",
            ])
    def setup(self, request):
        return request.param

    @pytest.fixture
    def prev_size(self, setup):
        return setup[0]

    @pytest.fixture
    def size(self, setup):
        return setup[1]

    @pytest.fixture
    def optimal_distortion(self, setup):
        return setup[2]

    @pytest.fixture
    def shape(self, size):
        return (size,)

    @pytest.fixture
    def previous(self, prev_size, variance_scale):
        from_ = 0.75 * variance_scale
        to_ = 1.25 * variance_scale
        value = np.linspace(from_, to_, num=prev_size)
        probability = np.ones((prev_size,))/prev_size
        return _QuantizerFactory.make_stub(
            value=value, probability=probability)

    @pytest.fixture
    def factory(self, model, size, previous):
        return _QuantizerFactory(model, size, previous)

    @pytest.mark.slow
    def test_optimal_distortion(self, factory, optimal_distortion):
        quant = factory.optimize()
        pt.assert_almost_equal(quant.distortion,
                optimal_distortion,
                rtol=1e-4)

    @pytest.mark.slow
    def test_far_inits_converge_to_same_result(
            self, factory, shape, random_normal):
        SEED = 987563
        # x
        init = 2.*random_normal
        quant = factory.optimize(init=init, seed=SEED)
        x1 = quant.x.copy()
        # Minus x
        init = -init
        quant = factory.optimize(init=init, seed=SEED)
        x2 = quant.x.copy()

        pt.assert_almost_equal(x1, x2, rtol=1e-2)


class TestFactoryWithBruteForce:

    def test__init__(self, model, factory, size, previous):
        assert factory.model is model
        assert factory.shape == (size,)
        assert factory.size == size
        assert factory.previous is previous

    def test_make_unconstrained(self, factory, x):
        quant = factory.make_unconstrained(x)
        quant = factory.make_unconstrained(x)  # Query cache?
        assert isinstance(quant, _Unconstrained)
        pt.assert_almost_equal(quant.x, x, atol=1e-16)

    def test_make_from_valid_value(self, factory, value):
        quant = factory.make(value)
        quant = factory.make(value)  # Query cache?
        assert isinstance(quant, _Quantizer)
        assert not isinstance(quant, _Unconstrained)
        pt.assert_almost_equal(quant.value, value)

    def test_make_stub_from_valid_value_and_proba(self, value):
        proba = np.ones_like(value)/value.size
        quant = _QuantizerFactory.make_stub(value, proba)
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
            quant = _QuantizerFactory.make_stub(invalid_value, proba)

    def test_make_stub_from_negative_raises_value_error(self, value):
        invalid_value = np.append(value, [-1.])
        proba = np.ones_like(invalid_value)/invalid_value.size
        with pytest.raises(ValueError):
            quant = _QuantizerFactory.make_stub(invalid_value, proba)

    @pytest.mark.parametrize("wrong", [2., -1.])
    def test_make_stub_from_invalid_proba_raises_value_error(
            self, value, wrong):
        proba = np.ones_like(value)/value.size
        proba[0] = wrong
        with pytest.raises(ValueError):
            quant = _QuantizerFactory.make_stub(value, proba)

    def test_min_variance(self, factory):
        expected = factory.model.get_lowest_one_step_variance(
            np.amin(factory.previous.value))
        pt.assert_almost_equal(factory.min_variance, expected)

    @pytest.mark.slow
    def test_optimize_with_brute_force(self, factory, size):
        quant = factory.optimize()
        width = 1.5*np.max(np.absolute(quant.x))
        ranges = (-width, width)
        ranges = (ranges,)*size
        expected = scipy.optimize.brute(
            lambda x: factory.make_unconstrained(x).distortion,
            ranges, full_output=True, disp=True
        )
        brute_x = expected[0]
        # Same optimal value
        pt.assert_almost_equal(quant.x, brute_x, rtol=1e-3)
        # Decreased distortion by less or equal
        brute_distortion = expected[1]
        assert quant.distortion <= brute_distortion


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
        value[0] = 0.5 * factory.min_variance
        if factory.size > 1:
            value[1] = factory.min_variance
        return factory.make(value)

    # Tests

    def test__init__invalid_parent_raises_value_error(self, valid_quantizer):
        with pytest.raises(ValueError):
            _Quantizer('invalid_parent', valid_quantizer)

    def test__init__parent_references(self, factory, valid_quantizer):
        assert valid_quantizer.model is factory.model
        assert valid_quantizer.size == factory.size
        assert valid_quantizer.shape == factory.shape
        assert valid_quantizer.previous is factory.previous

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
        expected = valid_quantizer.shape
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
        expected = (previous.value.size, valid_quantizer.size)
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

    def test_hessian_diag_at_singularity(self, factory, singular_quantizer):
        hessian_diag = diag_view(singular_quantizer.hessian)
        for ii in range(factory.size):
            # Does this Voronoi tile has singularity?
            has_singularity = np.any(isclose(
                singular_quantizer.voronoi[ii:ii+2, np.newaxis],
                factory.singularities[np.newaxis, :]))
            if has_singularity:
                assert hessian_diag[ii] == -np.inf
            else:
                assert np.isfinite(hessian_diag[ii])

    def test_hessian_off_diag_at_singularity(
            self, factory, singular_quantizer):
        if factory.size > 1:
            hessian_offd = diag_view(singular_quantizer.hessian, k=-1)
            for ii in range(factory.size-1):
                # Does the left bound of Voronoi tile is singularity?
                has_singularity = np.any(
                    singular_quantizer.voronoi[ii+1, np.newaxis]
                    == factory.singularities[np.newaxis, :])
                if has_singularity:
                    assert hessian_offd[ii] == -np.inf
                else:
                    assert np.isfinite(hessian_offd[ii])

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
        order, rtol=order_rtol
        expected_value=numerical_quantized_integral(
            model, previous, valid_quantizer.value, order=order)
        value=valid_quantizer._integral[order]
        pt.assert_almost_equal(value, expected_value, rtol=rtol, atol=1e-32)

    def test_delta_at_singularity_is_inf(self, previous, factory,
                                         singular_quantizer):
        voronoi = singular_quantizer.voronoi
        delta = singular_quantizer._delta
        for (ii, sing) in enumerate(factory.singularities):
            id_ = isclose(sing, voronoi)
            if np.any(id_):
                assert np.sum(id_) == 1
                pt.assert_all(delta[ii, id_] == np.inf)
                pt.assert_finite(delta[ii, ~id_])

    def test_delta_at_singularity_at_most_one_inf(self, singular_quantizer):
        delta = singular_quantizer._delta
        pt.assert_less_equal(np.sum(delta == np.inf, axis=0), 1,
                             shape='broad')
        pt.assert_less_equal(np.sum(delta == np.inf, axis=1), 1,
                             shape='broad')

    # Testing roots (possibly contains NaNs)

    @pytest.mark.parametrize("right", [False, True])
    def test_pdf(self, valid_quantizer, right):
        root = valid_quantizer._roots[right]
        with np.errstate(invalid='ignore'):
            expected_pdf = norm.pdf(root)
        expected_pdf[:, -1] = 0.0  # Limit at pm inf
        pdf = valid_quantizer._pdf[right]
        pt.assert_almost_equal(pdf, expected_pdf,
                               rtol=1e-12, invalid='allow')

    @pytest.mark.parametrize("right", [False, True])
    def test_cdf(self, valid_quantizer, right):
        root = valid_quantizer._roots[right]
        with np.errstate(invalid='ignore'):
            expected_cdf = norm.cdf(root)
        if right:
            expected_cdf[:, -1] = 1.0  # Limit at plus inf
        else:
            expected_cdf[:, -1] = 0.0  # Limit at minus inf
        value = valid_quantizer._cdf[right]
        pt.assert_almost_equal(value, expected_cdf,
                               rtol=1e-12, invalid='allow')

    @pytest.mark.parametrize("right", [False, True])
    def test_root_reverts_to(self, previous, valid_quantizer, right):
        root = valid_quantizer._roots[right]
        expected, _=valid_quantizer.model.one_step_generate(
            root, previous.value[:, np.newaxis])
        value=np.broadcast_to(
            valid_quantizer.voronoi[np.newaxis, :], expected.shape)
        pt.assert_almost_equal(value, expected,
                               rtol=1e-12,
                               invalid=valid_quantizer._no_roots)

    def test_roots_left_right_ordering(self, valid_quantizer):
        left = valid_quantizer._roots[False]
        right = valid_quantizer._roots[True]
        pt.assert_less(left, right,
                       invalid=valid_quantizer._no_roots)

    @pytest.mark.parametrize("right", [False, True])
    def test_roots_shape(self, valid_quantizer, right):
        expected_shape = (valid_quantizer.previous.size,
                          valid_quantizer.size+1)
        assert valid_quantizer._roots[right].shape == expected_shape

    def test_no_roots(self, model, previous, valid_quantizer):
        value = valid_quantizer._no_roots
        lowest = model.get_lowest_one_step_variance(previous.value)
        expected_value = (lowest[:, np.newaxis]
                          > valid_quantizer.voronoi[np.newaxis, :])
        pt.assert_all(value == expected_value)

    # Voronoi

    def test_voronoi(self, valid_quantizer):
        v = valid_quantizer.voronoi
        assert v[0] == 0.
        pt.assert_almost_equal(v, voronoi_1d(valid_quantizer.value, lb=0))


class TestUnconstrained():

    RTOL_GRADIENT = 1e-6
    RTOL_HESSIAN = 1e-6

    @pytest.fixture
    def quantizer(self, factory, x):
        return factory.make_unconstrained(x)

    def test__init__parent_references(self, factory, quantizer):
        assert quantizer.min_variance is factory.min_variance

    def test_value_is_in_space(self, quantizer):
        pt.assert_finite(quantizer.value)
        # strictly increasing
        pt.assert_greater(np.diff(quantizer.value), 0.0, shape='broad')
        pt.assert_greater(
            quantizer.value, quantizer.min_variance, shape='broad')

    def test_gradient(self, factory, x, quantizer):
        pt.assert_gradient_at(
            quantizer.gradient, x, rtol=self.RTOL_GRADIENT,
            function=lambda x_: factory.make_unconstrained(x_).distortion
        )

    def test_hessian(self, factory, x, quantizer):
        pt.assert_hessian_at(
            quantizer.hessian, x, rtol=self.RTOL_HESSIAN,
            gradient=lambda x_: factory.make_unconstrained(x_).gradient
        )

    def test_jacobian(self, factory, x, quantizer):
        pt.assert_jacobian_at(
            quantizer._jacobian,
            x, rtol=1e-8,
            function=lambda x_: factory.make_unconstrained(x_).value
        )

    def test_changed_variable(self, quantizer):
        expected_value = (quantizer.min_variance
                          + np.cumsum(quantizer._scaled_exp_x))
        pt.assert_almost_equal(quantizer.value, expected_value, rtol=1e-12)

    def test_inv_change_variable_reverts(self, factory, quantizer):
        x = quantizer._inv_change_of_variable(factory, quantizer.value)
        expected_x = quantizer.x
        pt.assert_almost_equal(x, expected_x, rtol=1e-6)

    def test_scaled_exponential_of_x(self, quantizer):
        expected_value = (np.exp(quantizer.x) * quantizer.min_variance
                          / (quantizer.size+1))
        pt.assert_almost_equal(quantizer._scaled_exp_x,
                               expected_value, rtol=1e-12)

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
            next_variance, _=model.one_step_generate(
                innov, prev_variance)
            assert (next_variance >= lb and next_variance <= ub)
            return integrand(innov, next_variance, value)

        # Identify integration intervals
        pts=(model.one_step_roots(prev_variance, lb)
               + model.one_step_roots(prev_variance, ub))
        pts=np.array(pts)
        pts=pts[~np.isnan(pts)]
        pts.sort()

        if pts.size == 0:
            out = 0.
        else:
            # Crop integral for acheiving better accuracy
            CROP = 10.
            if pts[0] == -np.inf and pts[-1] == np.inf:
                pts[0] = -CROP
                pts[-1] = CROP
            # Perform integration by quadrature
            if pts.size == 2:
                out = integrate.quad(function_to_integrate,
                                     pts[0], pts[1])[0]
            elif pts.size == 4:
                out = (integrate.quad(function_to_integrate,
                                      pts[0], pts[1])[0]
                       + integrate.quad(function_to_integrate,
                                        pts[2], pts[3])[0])
            else:
                assert False  # Should never happen
        return out

    # Perform Integration
    voronoi = voronoi_1d(value, lb=0.)
    I = np.empty((previous.value.size, value.size))
    for (i, prev_h) in enumerate(previous.value):
        for (j, (lb, ub, h)) in enumerate(
                zip(voronoi[:-1], voronoi[1:], value)):
            I[i, j] = do_quantized_integration(lb, ub, prev_h, h)
    return I
