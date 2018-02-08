from functools import partial

import pytest
import numpy as np
import scipy.optimize
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing as pt
import pymaat.garch.varquant as q
from pymaat.mathutil import voronoi_1d, inv_voronoi_1d

VARIANCE_SCALE = 0.18**2./252.

@pytest.fixture(params=[
        np.array([1.]),
        np.array([0.95,1.05]),
        np.array([1.3,0.55,1.,0.75])],
        ids=['prev_scalar','prev_len2','prev_len4_unsorted'],
        scope='module')
def previous(request):
    v = request.param
    p = np.ones_like(v)/v.size
    return q._QuantizerFactory.make_stub(
            value=VARIANCE_SCALE*v, probability=p)


# @pytest.mark.xfail
# class TestFactoryLargeScale:

#     @pytest.fixture
#     def shape(self):
#         return (10,)

#     @pytest.fixture
#     def factory(model, size, previous):
#         return q._QuantizerFactory(model, large_scale_size, previous)

#     def test_far_inits_converge_to_same_result(self, factory, random_normal):
#         # x
#         init = random_normal
#         success, x1 = large_scale_factory._perform_optimization(init=init)
#         assert success
#         # Minus x
#         init = -random_normal
#         success, x2 = large_scale_factory._perform_optimization(init=init)
#         assert success
#         pt.assert_almost_equal(x1,x2,rtol=1e-4)

# @pytest.mark.xfail
# class TestFactory:

#     def test__init__(self, model, factory, size, previous):
#         assert factory.model is model
#         assert factory.shape == (size,)
#         assert factory.size == size
#         assert factory.previous is previous

#     def test_make_unconstrained(self, factory, x):
#         quant = factory.make_unconstrained(x)
#         quant = factory.make_unconstrained(x) # Query cache?
#         assert isinstance(quant, q._Unconstrained)
#         pt.assert_almost_equal(quant.x, x, atol=1e-16)

#     def test_make_from_valid_value(self, factory, value):
#         quant = factory.make(value)
#         quant = factory.make(value) # Query cache?
#         assert isinstance(quant, q._Quantizer)
#         assert not isinstance(quant, q._Unconstrained)
#         pt.assert_almost_equal(quant.value, value)

#     def test_make_stub_from_valid_value_and_proba(self, value):
#         proba = np.ones_like(value)/value.size
#         quant = q._QuantizerFactory.make_stub(value, proba)
#         pt.assert_almost_equal(quant.value, value)
#         pt.assert_almost_equal(quant.probability, proba)
#         assert quant.size == value.size
#         assert quant.shape == value.shape

#     @pytest.mark.parametrize("wrong", [np.nan, np.inf])
#     def test_make_stub_from_invalid_value_raises_value_error(self,
#             value, wrong):
#         invalid_value = np.append(value,[wrong])
#         proba = np.ones_like(invalid_value)/invalid_value.size
#         with pytest.raises(ValueError):
#             quant = q._QuantizerFactory.make_stub(invalid_value, proba)

#     def test_make_stub_from_negative_raises_value_error(self, value):
#         invalid_value = np.append(value,[-1.])
#         proba = np.ones_like(invalid_value)/invalid_value.size
#         with pytest.raises(ValueError):
#             quant = q._QuantizerFactory.make_stub(invalid_value, proba)

#     @pytest.mark.parametrize("wrong", [2.,-1.])
#     def test_make_stub_from_invalid_proba_raises_value_error(self,
#             value, wrong):
#         proba = np.ones_like(value)/value.size
#         proba[0] = wrong
#         with pytest.raises(ValueError):
#             quant = q._QuantizerFactory.make_stub(value, proba)

#     def test_optimize_from_zero_converge_to_brute_force(self, factory, size):
#         # Value
#         success, x = factory._perform_optimization()
#         assert success
#         quant = factory.make_unconstrained(x)
#         if size<3: # Brute force is too slow when dim is greater than 2D
#             ranges = ((-7.,3.),)*size
#             func = lambda x: factory.make_unconstrained(x).distortion
#             expected = scipy.optimize.brute(func, ranges, full_output=True)
#             pt.assert_almost_equal(x, expected[0], rtol=1e-2)
#             assert quant.distortion<=expected[1]

#     def test_min_variance(self, factory):
#         expected = factory.model.get_lowest_one_step_variance(
#                 np.amin(factory.previous.value))
#         pt.assert_almost_equal(factory.min_variance, expected)

class TestQuantizer:

    @pytest.fixture(params = [
            np.array([.5]),
            np.array([.5,1.]),
            np.array([.25,.5,1.,1.5])],
            ids = ['len1', 'len2', 'len4'],
            scope='class')
    def setup(self, request, model, previous):
        perc = request.param
        size = perc.size
        factory = q._QuantizerFactory(model, size, previous)
        perc.sort()
        value = factory.min_variance*(1.+perc)
        quantizer = factory.make(value)
        return (factory, value, quantizer)

    @pytest.fixture(scope='class')
    def quantizer_at_singularity(self, model, previous, factory, value):
        value = value.copy()
        prev = np.sort(previous.value)
        lowest = model.get_lowest_one_step_variance(prev)

        if lowest.size==1:
            new_values = inv_voronoi_1d(lowest, with_bounds=False,
                    first_quantizer=0.5*lowest)
        else:
            new_values = inv_voronoi_1d(lowest, with_bounds=False)

        until = min(factory.size, new_values.size)
        value[:until] = new_values[:until]
        value.sort()
        return factory.make(value)

    # Some fixtures for convenience only...
    @pytest.fixture(scope='class')
    def factory(self, setup):
        return setup[0]

    @pytest.fixture(scope='class')
    def size(self, factory):
        return factory.size

    @pytest.fixture(scope='class')
    def value(self, setup):
        return setup[1]

    @pytest.fixture(scope='class')
    def quantizer(self, setup):
        return setup[2]

    def test__init__invalid_parent_raises_value_error(self, value):
        with pytest.raises(ValueError):
            q._Quantizer('invalid_parent', value)

    def test__init__parent_references(self, factory, quantizer):
        assert quantizer.model is factory.model
        assert quantizer.size == factory.size
        assert quantizer.shape == factory.shape
        assert quantizer.previous is factory.previous

    def test__init__invalid_value_ndim_raises_value_error(self,
            factory, value):
        invalid_value = np.array([[0.,1.],[2.,3.]])
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)

    def test__init__invalid_value_shape_raises_value_error(self,
            factory, value):
        invalid_value = np.append(value, [1.])
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)

    def test__init__value_nan_raises_value_error(
            self, factory, value):
        invalid_value = value.copy()
        invalid_value[0] = np.nan
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)

    def test__init__value_infinite_raises_value_error(
            self, factory, value):
        invalid_value = value.copy()
        invalid_value[0] = np.inf
        with pytest.raises(ValueError):
            q._Quantizer(factory, invalid_value)


    def test__init__non_increasing_value_raises_value_error(self,
            size, factory, value):
        if size>1:
            invalid_value = value.copy()
            invalid_value[[0,1]] = value[[1,0]]
            with pytest.raises(ValueError):
                q._Quantizer(factory, invalid_value)

    def test_value_has_no_nan(self, quantizer):
        pt.assert_valid(quantizer.value)

    def test_probability_shape(self, quantizer):
        expected = quantizer.shape
        value = quantizer.probability.shape
        assert expected == value

    def test_probability_sums_to_one_and_strictly_positive(self, quantizer):
        proba = quantizer.probability
        pt.assert_less(0.0, proba)
        pt.assert_almost_equal(np.sum(proba), 1., rtol=1e-6)

    def test_probability(self, previous, quantizer):
        value = quantizer.probability
        expected = previous.probability.dot(quantizer.transition_probability)
        pt.assert_almost_equal(value, expected, rtol=1e-6)

    def test_transition_probability_shape(self,
                previous, quantizer):
        expected = (previous.value.size, quantizer.size)
        value = quantizer.transition_probability.shape
        assert expected == value

    def test_transition_probability_sums_to_one_and_non_negative(
                self, quantizer):
        # Non negative...
        pt.assert_less_equal(0.0, quantizer.transition_probability)
        # Sums to one...
        marginal_proba = np.sum(quantizer.transition_probability,axis=1)
        pt.assert_almost_equal(marginal_proba, 1., rtol=1e-6)

    def test_transition_probability(self, previous, quantizer):
        value = quantizer.transition_probability
        expected_value = quantizer._integral[0]
        pt.assert_almost_equal(value, expected_value, rtol=1e-12)

    def test_distortion(self, previous, quantizer):
        # See below for _distortion_elements test
        distortion = np.sum(quantizer._distortion_elements, axis=1)
        expected_value = previous.probability.dot(distortion)
        value = quantizer.distortion
        pt.assert_almost_equal(expected_value, value, rtol=1e-6)

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

    def test_distortion_elements(self, model, previous, quantizer):
        expected_value = numerical_quantized_integral(model,
                previous, quantizer.value, distortion=True)
        value = quantizer._distortion_elements
        pt.assert_almost_equal(value, expected_value, rtol=1e-6)

    # Testing quantized integrals...

    @pytest.mark.parametrize("order_rtol", [(0,1e-8),(1,1e-6),(2,1e-3)])
    def test_integral(self, model, previous, quantizer, order_rtol):
        order, rtol = order_rtol
        expected_value = numerical_quantized_integral(model, previous,
                quantizer.value, order=order)
        value = quantizer._integral[order]
        pt.assert_almost_equal(value, expected_value, rtol=rtol, atol=1e-32)

    # Testing derivatives of integrals

    @pytest.mark.slow
    @pytest.mark.parametrize("order",[0,1])
    def test_integral_derivative(self, model, previous, quantizer, order):
        derivative = quantizer._integral_derivative[order]
        # Test shape prior to iteration on columns
        expected_shape = (quantizer.previous.size, quantizer.size)
        assert derivative.shape == expected_shape
        # Value
        for j in range(quantizer.size):
            def func(h):
                new_value = quantizer.value.copy()
                new_value[j] = h
                I = numerical_quantized_integral(model, previous,
                        new_value, order=order)
                return I[:,j]
            pt.assert_derivative_at(derivative[:,j], func,
                    quantizer.value[j], rtol=1e-3)

    @pytest.mark.slow
    @pytest.mark.parametrize("order",[0,1])
    def test_integral_derivative_lagged(self, model, previous, quantizer,
            order):
        derivative = quantizer._integral_derivative_lagged[order]
        # Test shape prior to iteration on columns
        expected_shape = (quantizer.previous.size, quantizer.size-1)
        assert derivative.shape == expected_shape
        # Value
        for j in range(1, quantizer.size):
            def func(h):
                new_value = quantizer.value.copy()
                new_value[j-1] = h
                I = numerical_quantized_integral(model, previous,
                        new_value, order=order)
                return I[:,j]
            pt.assert_derivative_at(derivative[:,j-1], func,
                    quantizer.value[j-1], rtol=1e-3)

    def test_delta_at_singularity_is_inf(self, model, previous, size,
            quantizer_at_singularity):

        singularity = model.get_lowest_one_step_variance(
                previous.value)
        voronoi = quantizer_at_singularity.voronoi
        delta = quantizer_at_singularity._delta

        for i in range(previous.size):
            for j in range(size+1):
                for o in range(3):
                    # TODO more lenient here?
                    if voronoi[j] == singularity[i]:
                        assert delta[o][i,j] == np.inf
                    else:
                        assert np.isfinite(delta[o][i,j])

    def test_delta_at_singularity_at_most_one_inf(self,
            quantizer_at_singularity):
        delta = quantizer_at_singularity._delta
        for o in range(3):
            number_of_inf = np.sum(delta[o]==np.inf, axis=1)
            pt.assert_all(number_of_inf<=1)

    def test_hessian_at_singularity(self,
            quantizer_at_singularity):
        hessian = quantizer_at_singularity.hessian
        assert False

    # Testing roots (possibly contains NaNs)

    @pytest.mark.parametrize("right",[False,True])
    def test_pdf(self, quantizer, right):
        root = quantizer._roots[right]
        with np.errstate(invalid='ignore'):
            expected_pdf = norm.pdf(root)
        expected_pdf[:,-1] = 0.0 # Limit at pm inf
        pdf = quantizer._pdf[right]
        pt.assert_almost_equal(pdf, expected_pdf,
                rtol=1e-12, invalid='allow')

    @pytest.mark.parametrize("right",[False,True])
    def test_cdf(self, quantizer, right):
        root = quantizer._roots[right]
        with np.errstate(invalid='ignore'):
            expected_cdf = norm.cdf(root)
        if right:
            expected_cdf[:,-1] = 1.0 # Limit at plus inf
        else:
            expected_cdf[:,-1] = 0.0 # Limit at minus inf
        value = quantizer._cdf[right]
        pt.assert_almost_equal(value, expected_cdf,
                rtol=1e-12, invalid='allow')

    @pytest.mark.parametrize("right",[False,True])
    def test_root_reverts_to(self, previous, quantizer, right):
        root = quantizer._roots[right]
        expected, _ = quantizer.model.one_step_generate(root,
                previous.value[:,np.newaxis])
        value = np.broadcast_to(
                quantizer.voronoi[np.newaxis,:], expected.shape)
        pt.assert_almost_equal(value, expected,
                rtol=1e-12,
                invalid=quantizer._no_roots)

    def test_roots_left_right_ordering(self, quantizer):
        left = quantizer._roots[False]
        right = quantizer._roots[True]
        pt.assert_less(left, right,
                invalid=quantizer._no_roots)

    @pytest.mark.parametrize("right",[False,True])
    def test_roots_shape(self, quantizer, right):
        expected_shape = (quantizer.previous.size, quantizer.size+1)
        assert quantizer._roots[right].shape == expected_shape

    def test_no_roots(self, model, previous, quantizer):
        value = quantizer._no_roots
        lowest = model.get_lowest_one_step_variance(previous.value)
        expected_value = (lowest[:,np.newaxis]
                > quantizer.voronoi[np.newaxis,:])
        pt.assert_all(value==expected_value)

    # Voronoi

    def test_voronoi(self, quantizer):
        v = quantizer.voronoi
        assert v[0] == 0.
        pt.assert_almost_equal(v, voronoi_1d(quantizer.value, lb=0))


#class TestUnconstrained():

    # def test__init__value_below_min_variance_raises_value_error(
    #         self, factory, value):
    #     invalid_value = value.copy()
    #     invalid_value[0] =  factory.min_variance * 0.5
    #     with pytest.raises(ValueError):
    #         q._Quantizer(factory, invalid_value)
#    @pytest.fixture(scope='class')
#    def x(size):
#        if size == 1:
#            return np.array([0.25])
#        elif size == 2:
#            return np.array([-0.23,-0.12])
#        elif size == 4:
#            return np.array([-0.12,-0.24,0.23,0.31])
#        elif size == 8:
#            return np.array([0.47,-1.07,0.86,0.20,-0.12,-0.24,0.23,0.31])
#        else:
#            assert False # Should never happend

#    @pytest.fixture(scope='class')
#    def quantizer(self, factory, x):
#        return factory.make_unconstrained(x)

    # def test__init__parent_references(self, factory, quantizer):
    #     assert quantizer.min_variance is factory.min_variance

#    def test_value_is_in_space(self, quantizer):
#        pt.assert_all(np.isfinite(quantizer.value))
#        pt.assert_less(0.0, quantizer.value)
#        pt.assert_less(0.0, np.diff(quantizer.value))
#        pt.assert_less(quantizer.min_variance, quantizer.value)

#    def test_gradient(self, factory, x, quantizer):
#        def func(x_):
#            return factory.make_unconstrained(x_).distortion
#        pt.assert_gradient_at(quantizer.gradient, func, x, rtol=1e-6)

#    def test_hessian(self, factory, x, quantizer):
#        def func(x_):
#            return factory.make_unconstrained(x_).gradient
#        pt.assert_jacobian_at(quantizer.hessian, func, x, rtol=1e-6)

#    def test_jacobian(self, factory, x, quantizer):
#        def func(x_):
#            return factory.make_unconstrained(x_).value
#        pt.assert_jacobian_at(quantizer._jacobian, func,
#                x, rtol=1e-8, atol=1e-32)

#    def test_changed_variable(self, quantizer):
#        expected_value = (quantizer.min_variance
#                + np.cumsum(quantizer._scaled_exp_x))
#        pt.assert_almost_equal(quantizer.value, expected_value, rtol=1e-12)

#    def test_scaled_exponential_of_x(self, quantizer):
#        expected_value = (np.exp(quantizer.x) * quantizer.min_variance
#            / (quantizer.size+1))
#        pt.assert_almost_equal(quantizer._scaled_exp_x,
#                expected_value, rtol=1e-12)

#######################
## Helper Function(s) #
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
        # Define function to integrate
        def function_to_integrate(innov):
            assert isinstance(innov, float)
            # 0 if innovation is between tile bounds else integrate
            next_variance, _ = model.one_step_generate(
                    innov, prev_variance)
            assert (next_variance>=lb and next_variance<=ub)
            return integrand(innov, next_variance, value)

        # Identify integration intervals
        pts = (model.one_step_roots(prev_variance, lb)
                + model.one_step_roots(prev_variance, ub))
        pts = np.array(pts)
        pts = pts[~np.isnan(pts)]
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

 ##pytest.main(__file__)
