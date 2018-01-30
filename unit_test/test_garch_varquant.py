from functools import partial

import pytest
import numpy as np
import scipy.optimize
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing as pt
from pymaat.garch import Garch
import pymaat.garch.varquant as q
from pymaat.mathutil import voronoi_1d

MODEL = Garch(mu=2.01,
        omega=9.75e-20,
        alpha=4.54e-6,
        beta=0.79,
        gamma=196.21)

VARIANCE_SCALE = 0.18**2./252.

@pytest.fixture(params=[1,2,4],
        ids=['prev_scalar','prev_len2','prev_len4_unsorted'],
        scope='module')
def previous(request):
    if request.param == 1:
        v = np.array([1.])
    elif request.param == 2:
        v = np.array([0.95,1.05])
    elif request.param == 4:
        v = np.array([1.3,0.55,1.,0.75])
    else:
        assert False # Should never happen
    p = np.ones_like(v)/v.size
    return q._QuantizerFactory.make_stub(
            value=VARIANCE_SCALE*v, probability=p)

@pytest.fixture(params=[1,2,4],
        ids=['scalar','len2','len4'],
        scope='module')
def size(request):
    return request.param

@pytest.fixture(scope='module')
def factory(previous, size):
    return q._QuantizerFactory(MODEL, size, previous)

@pytest.fixture(scope='module')
def value(factory, size):
    if size == 1:
        perc = np.array([.5])
    elif size == 2:
        perc = np.array([.5,1.])
    elif size == 4:
        perc = np.array([.25,.5,1.,1.5])
    else:
        assert False # Should never happen
    perc.sort()
    return factory.min_variance*(1.+perc)

@pytest.fixture(scope='module')
def x(size):
    if size == 1:
        return np.array([0.25])
    elif size == 2:
        return np.array([-0.23,-0.12])
    elif size == 4:
        return np.array([-0.12,-0.24,0.23,0.31])
    elif size == 8:
        return np.array([0.47,-1.07,0.86,0.20,-0.12,-0.24,0.23,0.31])
    else:
        assert False # Should never happend

@pytest.fixture(scope='module')
def large_scale_size():
    return 10

@pytest.fixture(scope='module')
def large_scale_x(large_scale_size):
    np.random.seed(1234567) # Ensure test consistency
    return np.random.normal(size=(large_scale_size,))

@pytest.fixture(scope='module')
def large_scale_factory(large_scale_size, previous):
    return q._QuantizerFactory(MODEL, large_scale_size, previous)

class TestFactoryLargeScale:

    def test_far_inits_converge_to_same_result(self,
            large_scale_factory,
            large_scale_x):
        # x
        init = large_scale_x
        success, x1 = large_scale_factory._perform_optimization(init=init)
        assert success
        # Minus x
        init = -init
        success, x2 = large_scale_factory._perform_optimization(init=init)
        assert success
        pt.assert_almost_equal(x1,x2,rtol=1e-4)

class TestFactory:

    def test__init__(self, factory, size, previous):
        assert factory.model is MODEL
        assert factory.shape == (size,)
        assert factory.size == size
        assert factory.previous is previous

    def test_make_unconstrained(self, factory, x):
        quant = factory.make_unconstrained(x)
        quant = factory.make_unconstrained(x) # Query cache?
        assert isinstance(quant, q._Unconstrained)
        pt.assert_almost_equal(quant.x, x, atol=1e-16)

    def test_make_from_valid_value(self, factory, value):
        quant = factory.make(value)
        quant = factory.make(value) # Query cache?
        assert isinstance(quant, q._Quantizer)
        assert not isinstance(quant, q._Unconstrained)
        pt.assert_almost_equal(quant.value, value)

    def test_make_stub_from_valid_value_and_proba(self, value):
        proba = np.ones_like(value)/value.size
        quant = q._QuantizerFactory.make_stub(value, proba)
        pt.assert_almost_equal(quant.value, value)
        pt.assert_almost_equal(quant.probability, proba)
        assert quant.size == value.size
        assert quant.shape == value.shape

    @pytest.mark.parametrize("wrong", [np.nan, np.inf])
    def test_make_stub_from_invalid_value_raises_value_error(self,
            value, wrong):
        invalid_value = np.append(value,[wrong])
        proba = np.ones_like(invalid_value)/invalid_value.size
        msg = 'Invalid value: NaNs or infinite'
        with pytest.raises(ValueError):
            quant = q._QuantizerFactory.make_stub(invalid_value, proba)

    def test_make_stub_from_negative_raises_value_error(self, value):
        invalid_value = np.append(value,[-1.])
        proba = np.ones_like(invalid_value)/invalid_value.size
        msg = 'Invalid value: variance must be strictly positive'
        with pytest.raises(ValueError):
            quant = q._QuantizerFactory.make_stub(invalid_value, proba)

    @pytest.mark.parametrize("wrong", [2.,-1.])
    def test_make_stub_from_invalid_proba_raises_value_error(self,
            value, wrong):
        proba = np.ones_like(value)/value.size
        proba[0] = wrong
        with pytest.raises(ValueError):
            quant = q._QuantizerFactory.make_stub(value, proba)

    def test_optimize_from_zero_converge_to_brute_force(self, factory, size):
        # Value
        success, x = factory._perform_optimization()
        assert success
        quant = factory.make_unconstrained(x)
        if size<3: # Brute force is too slow when dim is greater than 2D
            ranges = ((-7.,3.),)*size
            func = lambda x: factory.make_unconstrained(x).distortion
            expected = scipy.optimize.brute(func, ranges, full_output=True)
            pt.assert_almost_equal(x, expected[0], rtol=1e-2)
            assert quant.distortion<=expected[1]

    def test_min_variance(self, factory):
        expected = factory.model.get_lowest_one_step_variance(
                np.amin(factory.previous.value))
        pt.assert_almost_equal(factory.min_variance, expected)

class TestQuantizer:

    @pytest.fixture(scope='class')
    def quantizer(self, factory, value):
        return factory.make(value)

    def test__init__invalid_parent_raises_value_error(self, value):
        msg = 'Quantizers must be instantiated from valid factory'
        with pytest.raises(ValueError):
            q._Quantizer('invalid_parent', value)

    def test__init__parent_references(self, factory, quantizer):
        assert quantizer.model is factory.model
        assert quantizer.size == factory.size
        assert quantizer.shape == factory.shape
        assert quantizer.previous is factory.previous
        assert quantizer.min_variance is factory.min_variance

    def test__init__invalid_value_ndim_raises_value_error(self,
            factory, value):
        expected_msg = 'Unexpected quantizer shape'
        invalid_value = np.array([[0.,1.],[2.,3.]])
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)

    def test__init__invalid_value_shape_raises_value_error(self,
            factory, value):
        expected_msg = 'Unexpected quantizer shape'
        invalid_value = np.append(value, [1.])
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)

    def test__init__value_nan_raises_value_error(
            self, factory, value):
        msg = 'Invalid quantizer'
        invalid_value = value.copy()
        invalid_value[0] = np.nan
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)

    def test__init__value_infinite_raises_value_error(
            self, factory, value):
        msg = 'Invalid quantizer'
        invalid_value = value.copy()
        invalid_value[0] = np.inf
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)

    def test__init__value_below_min_variance_raises_value_error(
            self, factory, value):
        msg = ('Unexpected quantizer value(s): '
        'must be strictly above minimum variance')
        invalid_value = value.copy()
        invalid_value[0] =  factory.min_variance * 0.5
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)

    def test__init__non_increasing_value_raises_value_error(self,
            size, factory, value):
        if size>1:
            msg = ('Unexpected quantizer value(s): '
                    'must be strictly increasing')
            invalid_value = value.copy()
            invalid_value[[0,1]] = value[[1,0]]
            with pytest.raises(ValueError):
                q._Quantizer(factory, invalid_value)

    def test_quantizer_has_no_nan(self, quantizer):
        pt.assert_false(np.isnan(quantizer.voronoi))
        pt.assert_false(np.isnan(quantizer.value))
        pt.assert_false(np.isnan(quantizer.distortion))
        pt.assert_false(np.isnan(quantizer.gradient))
        pt.assert_false(np.isnan(quantizer.hessian))
        pt.assert_false(np.isnan(quantizer.probability))
        pt.assert_false(np.isnan(quantizer.transition_probability))

    def test_probability_shape(self, quantizer):
        expected = quantizer.shape
        value = quantizer.probability.shape
        assert expected == value

    def test_probability_sums_to_one_and_strictly_positive(self, quantizer):
        proba = quantizer.probability
        pt.assert_true(proba>0.)
        pytest.approx(np.sum(proba),1.)

    def test_transition_probability_shape(self,
                previous, quantizer):
        expected = (previous.value.size, quantizer.size)
        value = quantizer.transition_probability.shape
        assert expected == value

    def test_transition_probability_sums_to_one_and_non_negative(
                self, quantizer):
        # Non negative...
        pt.assert_true(quantizer.transition_probability>=0.)
        # Sums to one...
        marginal_proba = np.sum(quantizer.transition_probability,axis=1)
        pytest.approx(marginal_proba, 1.)

    def test_transition_probability(self, previous, quantizer):
        value = quantizer.transition_probability
        expected_value = quantizer._integral[0]
        pt.assert_almost_equal(value, expected_value, rtol=1e-12)

    def test_distortion(self, previous, quantizer):
        # See below for _distortion_elements test
        distortion = np.sum(quantizer._distortion_elements, axis=1)
        expected_value = previous.probability.dot(distortion)
        value = quantizer.distortion
        pt.assert_almost_equal(expected_value, value,
                msg='Incorrect distortion', rtol=1e-6)

    def test_gradient(self, factory, quantizer):
        def func(value_):
            return factory.make(value_).distortion
        pt.assert_gradient_at(quantizer.gradient,
                func, quantizer.value, rtol=1e-6)

    def test_hessian(self, factory, quantizer):
        def func(value_):
            return factory.make(value_).gradient
        pt.assert_jacobian_at(quantizer.hessian, func,
                quantizer.value, rtol=1e-6)

    # Testing distortion...

    def test_valid_distortion_elements(self, quantizer):
        value = quantizer._distortion_elements
        # Any NaNs?
        assert not np.any(np.isnan(value))
        # Shape
        expected_shape = (quantizer.previous.size, quantizer.size)
        assert value.shape == expected_shape

    def test_distortion_elements(self, previous, quantizer):
        expected_value = numerical_quantized_integral(
                previous, quantizer.value, distortion=True)
        value = quantizer._distortion_elements
        pt.assert_almost_equal(value, expected_value, rtol=1e-6)

    # Testing quantized integrals...

    @pytest.mark.parametrize("order",[0,1,2])
    def test_valid_integral(self, quantizer, order):
        value = quantizer._integral[order]
        # Any NaNs?
        assert not np.any(np.isnan(value))
        # Shape?
        expected_shape = (quantizer.previous.size, quantizer.size)
        assert value.shape == expected_shape

    @pytest.mark.parametrize("order",[0,1,2])
    def test_integral(self, previous, quantizer, order):
        if order == 0:
            rtol = 1e-8
        elif order == 1:
            rtol = 1e-6
        elif order == 2:
            rtol = 1e-3
        expected_value = numerical_quantized_integral(
                previous,
                quantizer.value, order=order)
        value = quantizer._integral[order]
        pt.assert_almost_equal(value, expected_value, rtol=rtol, atol=1e-32,
                msg='Incorrect next variance integral')

    # Testing derivatives of integrals

    @pytest.mark.parametrize("order",[0,1])
    def test_valid_integral_derivative(self, quantizer, order):
        # Any NaNs?
        value = quantizer._integral_derivative[order]
        assert not np.any(np.isnan(value))
        # Shape?
        expected_shape = (quantizer.previous.size, quantizer.size)
        assert value.shape == expected_shape

    @pytest.mark.slow
    @pytest.mark.parametrize("order",[0,1])
    def test_integral_derivative(self, previous, quantizer, order):
        derivative = quantizer._integral_derivative[order]
        for j in range(quantizer.size):
            def func(h):
                new_value = quantizer.value.copy()
                new_value[j] = h
                I = numerical_quantized_integral(
                        previous,
                        new_value, order=order)
                return I[:,j]
            pt.assert_derivative_at(derivative[:,j], func,
                    quantizer.value[j], rtol=1e-3)

    @pytest.mark.parametrize("order",[0,1])
    def test_valid_integral_derivative_lagged(self, quantizer, order):
        # Any NaNs?
        value = quantizer._integral_derivative_lagged[order]
        assert not np.any(np.isnan(value))
        # Shape?
        expected_shape = (quantizer.previous.size, quantizer.size-1)
        assert value.shape == expected_shape

    @pytest.mark.slow
    @pytest.mark.parametrize("order",[0,1])
    def test_integral_derivative_lagged(self,
            previous, quantizer, order):
        derivative = quantizer._integral_derivative_lagged[order]
        for j in range(1, quantizer.size):
            def func(h):
                new_value = quantizer.value.copy()
                new_value[j-1] = h
                I = numerical_quantized_integral(
                        previous,
                        new_value,
                        order=order)
                return I[:,j]
            pt.assert_derivative_at(derivative[:,j-1], func,
                    quantizer.value[j-1], rtol=1e-3)

    # Testing roots

    @pytest.mark.parametrize("right",[False,True])
    def test_pdf_or_is_masked(self, quantizer, right):
        root = quantizer._roots[right]
        expected_pdf = norm.pdf(root)
        pdf = quantizer._pdf[right]
        pt.assert_almost_equal(pdf, expected_pdf, rtol=1e-12)

    @pytest.mark.parametrize("right",[False,True])
    def test_cdf_or_is_masked(self, quantizer, right):
        root = quantizer._roots[right]
        expected_cdf = norm.cdf(root)
        cdf = quantizer._cdf[right]
        pt.assert_almost_equal(cdf, expected_cdf, rtol=1e-12)

    @pytest.mark.parametrize("right",[False,True])
    def test_root_reverts_to_or_is_masked(self, previous, quantizer, right):
        root = quantizer._roots[right]
        expected, _ = quantizer.model.one_step_generate(root,
                previous.value[:,np.newaxis])
        value = np.broadcast_to(
                quantizer.voronoi[np.newaxis,:], expected.shape)
        pt.assert_almost_equal(value, expected, rtol=1e-12)

    def test_roots_same_mask(self, quantizer):
        pt.assert_equal(quantizer._roots[True].mask,
                quantizer._roots[False].mask)

    def test_roots_left_right_ordering_or_is_masked(self, quantizer):
        left = quantizer._roots[False]
        right = quantizer._roots[True]
        pt.assert_true(right>left)

    @pytest.mark.parametrize("right",[False,True])
    def test_roots_size(self, quantizer, right):
        expected_shape = (quantizer.previous.size, quantizer.size+1)
        assert quantizer._roots[right].shape == expected_shape

    # Voronoi

    def test_voronoi(self, quantizer):
        v = quantizer.voronoi
        assert not np.any(np.isnan(v))
        assert v[0] == 0.
        pt.assert_true(v[1:]>=quantizer.min_variance)
        pt.assert_almost_equal(v, voronoi_1d(quantizer.value, lb=0))


class TestUnconstrained():


    @pytest.fixture(scope='class')
    def quantizer(self, factory, x):
        return factory.make_unconstrained(x)

    def test_quantizer_is_valid(self, quantizer):
        pt.assert_false(np.isnan(quantizer.gradient))
        pt.assert_false(np.isnan(quantizer.hessian))
        pt.assert_false(np.isnan(quantizer._jacobian))

    def test_value_is_in_space(self, quantizer):
        pt.assert_true(np.isfinite(quantizer.value))
        pt.assert_true(quantizer.value>0.)
        pt.assert_true(np.diff(quantizer.value)>0., msg='is not sorted')
        pt.assert_true(quantizer.value>quantizer.min_variance)

    def test_gradient(self, factory, x, quantizer):
        def func(x_):
            return factory.make_unconstrained(x_).distortion
        pt.assert_gradient_at(quantizer.gradient, func, x, rtol=1e-6)

    def test_hessian(self, factory, x, quantizer):
        def func(x_):
            return factory.make_unconstrained(x_).gradient
        pt.assert_jacobian_at(quantizer.hessian, func, x, rtol=1e-6)

    def test_jacobian(self, factory, x, quantizer):
        def func(x_):
            return factory.make_unconstrained(x_).value
        pt.assert_jacobian_at(quantizer._jacobian, func,
                x, rtol=1e-8, atol=1e-32)

    def test_changed_variable(self, quantizer):
        expected_value = (quantizer.min_variance
                + np.cumsum(quantizer._scaled_exp_x))
        pt.assert_almost_equal(quantizer.value, expected_value, rtol=1e-12)

    def test_scaled_exponential_of_x(self, quantizer):
        expected_value = (np.exp(quantizer.x) * quantizer.min_variance
            / (quantizer.size+1))
        pt.assert_almost_equal(quantizer._scaled_exp_x,
                expected_value, rtol=1e-12)

######################
# Helper Function(s) #
######################

def numerical_quantized_integral(previous, value, *,
        order=None,
        distortion=False):
    """
    This helper function numerically estimates (via quadrature)
    an integral (specified by order or distortion)
    over quantized tiles
    """
    # Setup integrand
    if order is not None:
        if order==0:
            integrand = lambda z,H,h: norm.pdf(z)
        elif order==1:
            integrand = lambda z,H,h: H * norm.pdf(z)
        elif order==2:
            integrand = lambda z,H,h: H**2. * norm.pdf(z)
        else:
            assert False # Should never happen...
    elif distortion:
        integrand = lambda z,H,h: (H-h)**2. * norm.pdf(z)
    else:
        assert False # Should never happen...

    def do_quantized_integration(lb, ub, prev_variance, value):
        assert prev_variance.size==1
        assert value.size==1

        lb = np.atleast_1d(lb)
        ub = np.atleast_1d(ub)

        # Define function to integrate
        def function_to_integrate(innov):
            assert isinstance(innov, float)
            # 0 if innovation is between tile bounds else integrate
            next_variance, _ = MODEL.one_step_generate(
                    innov, prev_variance)
            assert (next_variance>=lb and next_variance<=ub)
            return integrand(innov, next_variance, value)

        # Identify integration intervals
        pts = (MODEL.one_step_roots(prev_variance, lb)
                + MODEL.one_step_roots(prev_variance, ub))
        pts = np.concatenate([p.compressed() for p in pts])
        pts.sort()

        if pts.size==0:
            out = 0.
        else:
            # Crop integral for acheiving better accuracy
            CROP=15
            if pts[0]==-np.inf and pts[-1]==np.inf:
                pts[0]=-CROP; pts[-1]=CROP
            # Perform integration by quadrature
            if pts.size==2:
                out = integrate.quad(function_to_integrate,
                        pts[0], pts[1])[0]
            elif pts.size==4:
                out = (integrate.quad(function_to_integrate,
                        pts[0], pts[1])[0]
                    + integrate.quad(function_to_integrate,
                        pts[2], pts[3])[0])
            else:
                assert False # Should never happen
        return out

    # Perform Integration
    voronoi = voronoi_1d(value, lb=0.)
    I = np.empty((previous.value.size,value.size))
    for (i,prev_h) in enumerate(previous.value):
        for (j,(lb,ub,h)) in enumerate(
                zip(voronoi[:-1], voronoi[1:], value)):
            I[i,j] = do_quantized_integration(lb, ub, prev_h, h)
    return I

# pytest.main(__file__)
