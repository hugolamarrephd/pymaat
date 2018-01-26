import pytest
from functools import partial

import pytest
import numpy as np
import scipy.optimize
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing as pt
import pymaat.garch
import pymaat.garch.quant
from pymaat.mathutil import voronoi_1d

MODEL = pymaat.garch.Garch(
        mu=2.01,
        omega=9.75e-20,
        alpha=4.54e-6,
        beta=0.79,
        gamma=196.21)

VARIANCE_SCALE = 0.18**2./252.

class TestMarginalVariance:

    ############
    # Fixtures #
    ############

    quant_fact = pymaat.garch.quant.MarginalVariance._QuantizerFactory

    @pytest.fixture(params=[
        quant_fact(
            value=VARIANCE_SCALE*np.array([1.]),
            probability=np.array([1.]),
            transition_probability=None),
        quant_fact(
            value=VARIANCE_SCALE*np.array([0.95,1.]),
            probability=np.array([0.5,0.5]),
            transition_probability=None),
        quant_fact(
            value=VARIANCE_SCALE*np.array([0.55,0.75,1.,1.43]),
            probability=np.array([0.2,0.25,0.45,0.10]),
            transition_probability=None)],
        ids=['prev_scalar','prev_len2','prev_len4'],
        scope='class')
    def previous_quantizer(self, request):
        return request.param

    @pytest.fixture(params=[
            np.array([0.25]),
            np.array([-0.23,-0.12]),
            np.array([-0.12,-0.24,0.23,0.31]),
            np.array([0.47,-1.07,0.86,0.20,-0.12,-0.24,0.23,0.31]),
            ],
            ids = [
                'x_scalar',
                'x_len2',
                'x_len4',
                'x_len8'],
            scope='class')
    def x(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def optimizer(self, previous_quantizer, x):
        return pymaat.garch.quant._MarginalVarianceOptimizer(
                MODEL, x.size, previous_quantizer)

    @pytest.fixture(scope='class')
    def state(self, optimizer, x):
        return optimizer.eval_at_x(x)

    ###################
    # Optimizer Tests #
    ###################

    def test_optimizer_size(self, x, optimizer):
        assert optimizer.size == x.size

    def test_optimizer_shape(self, x, optimizer):
        assert optimizer.shape == (x.size,)

    def test_optimizer_returns_optimum_from_zero(self, optimizer):
        value = optimizer()
        func = lambda x: optimizer.eval_at_x(x).distortion
        expected = scipy.optimize.basinhopping(
                func, np.zeros(optimizer.shape))
        assert value.distortion<=expected.fun

    def test_far_inits_converge_to_same_result(self, optimizer, x):
        sucess, x1 = optimizer._perform_optimization(init=x)
        sucess, x2 = optimizer._perform_optimization(init=-x)
        pt.assert_almost_equal(x1,x2,rtol=1e-6)

    def test_minimal_variance(self, previous_quantizer, optimizer):
        pt.assert_true(optimizer.min_variance>0.,
                msg='minimum variance must be positive')
        pt.assert_true(optimizer.min_variance<previous_quantizer.value[0],
                msg='minimum variance is strictly decreasing in time')

    ###################################
    # Top Level Optimizer State Tests #
    ###################################

    def test_state_is_valid(self, state):
        assert not np.any(np.isnan(state.value))
        assert not np.any(np.isnan(state.distortion))
        assert not np.any(np.isnan(state.transformed_gradient))
        assert not np.any(np.isnan(state.transformed_hessian))
        assert not np.any(np.isnan(state.probability))
        assert not np.any(np.isnan(state.transition_probability))

    def test_value_is_in_space(self, state, x):
        pt.assert_true(np.isfinite(state.value))
        pt.assert_true(state.value>0.)
        pt.assert_true(np.diff(state.value)>0., msg='is not sorted')
        pt.assert_true(state.value>state.parent.min_variance)

    def test_value(self, state):
        pt.assert_almost_equal(state.value,
                state.get_changed_variable(),
                rtol=1e-12)

    def test_probability_shape(self, state):
        pt.assert_equal(state.probability.shape, state.parent.shape)

    def test_probability_sums_to_one_and_strictly_positive(self, state):
        pt.assert_true(state.probability>0.)
        assert np.sum(state.probability)==1.

    def test_transition_probability_shape(self, previous_quantizer, state):
        pt.assert_equal(state.transition_probability.shape,
                (previous_quantizer.value.size, state.parent.size))

    def test_transition_probability_sums_to_one_and_non_negative(self, state):
        pt.assert_true(state.transition_probability>=0.)
        pt.assert_almost_equal(np.sum(state.transition_probability,axis=1),
                1.)

    def test_transition_probability(self, previous_quantizer, state):
        value = state.transition_probability
        expected_value = state.get_integral(0)
        pt.assert_almost_equal(value, expected_value)

    def test_distortion(self, previous_quantizer, state):
        distortion = np.sum(state.get_distortion_elements(), axis=1)
        expected_value = previous_quantizer.probability.dot(distortion)
        value = state.distortion
        pt.assert_almost_equal(expected_value, value,
                msg='Incorrect distortion', rtol=1e-6)

    def test_transformed_gradient(self, optimizer, x, state):
        def func(x_):
            return optimizer.eval_at_x(x_).distortion
        pt.assert_gradient_at(state.transformed_gradient,
             func, x, rtol=1e-4)

    def test_transformed_hessian(self, optimizer, x, state):
        def func(x_):
            return optimizer.eval_at_x(x_).transformed_gradient
        pt.assert_jacobian_at(state.transformed_hessian,
                func, x, rtol=1e-4)

    ###################
    # Low Level Tests #
    ###################

    def test_valid_helpers(self, state):
        assert not np.any(np.isnan(state.get_distortion_elements()))
        assert not np.any(np.isnan(state.get_gradient()))
        assert not np.any(np.isnan(state.get_hessian()))
        assert not np.any(np.isnan(state.get_jacobian()))
        assert not np.any(np.isnan(state.get_first_thess_term()))
        assert not np.any(np.isnan(state.get_second_thess_term()))
        assert not np.any(np.isnan(state.get_changed_variable()))
        assert not np.any(np.isnan(state.get_scaled_exp_x()))

    def test_distortion_elements(self, previous_quantizer, state):
        expected_value = self._quantized_integral(
                previous_quantizer, state.value, distortion=True)
        value = state.get_distortion_elements()
        pt.assert_almost_equal(value, expected_value, rtol=1e-6)

    # Testing un-transformed Hessian and gradient...
    def test_gradient(self, optimizer, state):
        def func(value_):
            return optimizer.eval_at_value(value_).distortion
        grad = state.get_gradient()
        pt.assert_gradient_at(grad, func, state.value, rtol=1e-6)


    def test_hessian(self, optimizer, state):
        def func(value_):
            return optimizer.eval_at_value(value_).distortion
        hess = state.get_hessian()
        pt.assert_hessian_at(hess, func, state.value,
                rtol=1e-4, atol=1e-6)

    # Change of variable

    def test_jacobian(self, optimizer, x, state):
        def func(x_):
            return optimizer.eval_at_x(x_).get_changed_variable()
        pt.assert_jacobian_at(state.get_jacobian(), func,
                x, rtol=1e-8, atol=1e-32)

    @pytest.mark.slow
    def test_first_thess_term(self, state):
        expected = np.zeros((state.parent.size,state.parent.size))
        for i in range(state.parent.size):
            for j in range(state.parent.size):
                for xi in range(state.parent.size):
                    for nu in range(state.parent.size):
                        expected[i,j] += (state.get_jacobian()[xi,i]
                                * state.get_hessian()[xi,nu]
                                * state.get_jacobian()[nu,j])
        value = state.get_first_thess_term()
        pt.assert_almost_equal(value, expected, rtol=1e-12)

    def test_second_thess_term_diagonal(self, state):
        sth = state.get_second_thess_term()
        expected = np.empty(state.parent.shape)
        for i in range(state.parent.size):
            expected[i] = 0.
            for nu in range(state.parent.size):
                if i<=nu:
                    expected[i] += (state.get_gradient()[nu]
                    * state.get_scaled_exp_x()[i])
        pt.assert_almost_equal(np.diag(sth), expected, rtol=1e-12)

    def test_second_thess_term_is_diagonal(self, state):
        sth = state.get_second_thess_term()
        for i in range(state.parent.size):
            for j in range(state.parent.size):
                if i != j:
                    assert sth[i,j]==0.

    def test_changed_variable(self, state):
        expected_value = (state.parent.min_variance
                + np.cumsum(state.get_scaled_exp_x()))
        pt.assert_almost_equal(state.get_changed_variable(),
                expected_value, rtol=1e-12)

    def test_scaled_exponential_of_x(self, state):
        expected_value = (np.exp(state.x)
            * state.parent.min_variance
            / (state.parent.size+1))
        pt.assert_almost_equal(state.get_scaled_exp_x(),
                expected_value, rtol=1e-12)

    # Testing quantized integrals...

    @pytest.mark.parametrize("order",[0,1,2])
    def test_valid_integral(self, state, order):
        assert not np.any(np.isnan(state.get_integral(order)))

    @pytest.mark.parametrize("order",[0,1,2])
    def test_integral(self, previous_quantizer, state, order):
        """
        \mathcal{I}_{order,t}^{(ij)} for all i,j
        """
        if order == 0:
            rtol = 1e-8
        elif order == 1:
            rtol = 1e-6
        elif order == 2:
            rtol = 1e-3
        expected_value = self._quantized_integral(previous_quantizer,
                state.value, order=order)
        value = state.get_integral(order)
        pt.assert_almost_equal(value, expected_value, rtol=rtol, atol=1e-32,
                msg='Incorrect next variance integral')

    # Testing derivatives of integrals

    @pytest.mark.parametrize("order",[0,1])
    def test_valid_integral_derivative(self, state, order):
        assert not np.any(np.isnan(state.get_integral_derivative(
            order=order, lagged=False)))

    @pytest.mark.parametrize("order",[0,1])
    def test_integral_derivative(self, previous_quantizer, state, order):
        """
        d\mathcal{I}_{order,t}^{(ij)}/dh_t^{(j)} for all i,j
        """
        derivative = state.get_integral_derivative(
                order=order, lagged=False)
        for j in range(state.parent.size):
            def func(h):
                new_value = state.value.copy()
                new_value[j] = h
                I = self._quantized_integral(previous_quantizer,
                        new_value, order=order)
                return I[:,j]
            pt.assert_derivative_at(derivative[:,j], func,
                    state.value[j], rtol=1e-3)

    @pytest.mark.parametrize("order",[0,1])
    def test_valid_integral_derivative_lagged(self, state, order):
        value = state.get_integral_derivative(order=order, lagged=True)
        assert not np.any(np.isnan(value[:,1:]))

    @pytest.mark.parametrize("order",[0,1])
    def test_integral_derivative_lagged(self,
            previous_quantizer, state, order):
        """
        d\mathcal{I}_{order,t}^{(ij)}/dh_t^{(j-1)} for all i,j
        """
        derivative = state.get_integral_derivative(
                order=order, lagged=True)
        for j in range(1,state.parent.size):
            def func(h):
                new_value = state.value.copy()
                new_value[j-1] = h
                I = self._quantized_integral(previous_quantizer, new_value,
                        order=order)
                return I[:,j]
            pt.assert_derivative_at(derivative[:,j], func,
                    state.value[j-1], rtol=1e-3)
        # First column is NaN by convention
        pt.assert_equal(derivative[:,0], np.nan)

    # Testing roots

    @pytest.mark.parametrize("right",[0,1])
    def test_pdf_or_is_nan(self, state, right):
        r = state.get_roots(right)
        is_valid = ~np.isnan(r)
        expected_pdf = norm.pdf(r)
        pdf = state.get_pdf(right)
        pt.assert_almost_equal(pdf[is_valid], expected_pdf[is_valid],
                rtol=1e-12)
        pt.assert_true(np.isnan(pdf[~is_valid]))

    @pytest.mark.parametrize("right",[0,1])
    def test_cdf_or_is_nan(self, state, right):
        r = state.get_roots(right)
        is_valid = ~np.isnan(r)
        expected_cdf = norm.cdf(r)
        cdf = state.get_cdf(right)
        pt.assert_almost_equal(cdf[is_valid], expected_cdf[is_valid],
                rtol=1e-12)
        pt.assert_true(np.isnan(cdf[~is_valid]))

    @pytest.mark.parametrize("right",[0,1])
    def test_at_least_one_valid_root(self, state, right):
        r = state.get_roots(right)
        assert not np.any(np.isnan(r[0,1:]))

    @pytest.mark.parametrize("right",[0,1])
    def test_decreasing_number_of_invalid_roots(self,
            previous_quantizer, state, right):
        r = state.get_roots(right)
        prev_id_ = 0
        for j in range(state.parent.size):
            print(r[:,j])
            id_ = np.argmax(np.isnan(r[:,j]))
            assert id_==0 or id_>=prev_id_
            prev_id_ = id_

    @pytest.mark.parametrize("right",[0,1])
    def test_root_reverts_to_or_is_nan(self,
            previous_quantizer, state, right):
        r = state.get_roots(right)
        expected, _ = state.parent.model.one_step_generate(r,
                previous_quantizer.value[:,np.newaxis])
        value = np.broadcast_to(state.get_voronoi()[np.newaxis,:],
                expected.shape)
        is_valid = ~np.isnan(r)
        pt.assert_almost_equal(value[is_valid], expected[is_valid],
                rtol=1e-12)

    def test_roots_left_right_ordering_or_is_nan(self, state):
        left = state.get_roots(0)
        right = state.get_roots(1)
        is_valid = ~np.isnan(left)
        pt.assert_true(right[is_valid]>left[is_valid])

    # Voronoi

    def test_voronoi(self, state):
        v = state.get_voronoi()
        assert not np.any(np.isnan(v))
        assert v[0] == 0.
        pt.assert_true(v[1:]>=state.parent.min_variance)
        pt.assert_almost_equal(v, voronoi_1d(state.value, lb=0))

    #########################
    # Test Helper Functions #
    #########################

    @staticmethod
    def _quantized_integral(previous_quantizer, value, *,
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

        def do_quantized_integration(tile_bounds, prev_variance, value):
            assert len(tile_bounds)==2
            assert tile_bounds[0].size==1 and tile_bounds[1].size==1
            assert prev_variance.size==1
            assert value.size==1
            assert np.isfinite(tile_bounds[0])

            # Define function to integrate
            def function_to_integrate(innov):
                assert isinstance(innov, float)
                # 0 if innovation is between tile bounds else integrate
                next_variance, _ = MODEL.one_step_generate(
                        innov, prev_variance)
                assert (next_variance>=tile_bounds[0] and
                        next_variance<=tile_bounds[1])
                return integrand(innov, next_variance, value)

            # Identify integration intervals
            critical_pts = np.concatenate(
                    [MODEL.one_step_roots(prev_variance, b)
                    for b in tile_bounds])
            critical_pts = critical_pts[~np.isnan(critical_pts)]
            critical_pts.sort()

            if critical_pts.size==0:
                out = 0.
            else:
                # Crop integral for acheiving better accuracy
                CROP=15
                if critical_pts[0]==-np.inf and critical_pts[-1]==np.inf:
                    critical_pts[0]=-CROP; critical_pts[-1]=CROP
                # Perform integration by quadrature
                if critical_pts.size==2:
                    out = integrate.quad(function_to_integrate,
                            critical_pts[0], critical_pts[1])[0]
                elif critical_pts.size==4:
                    out = (integrate.quad(function_to_integrate,
                            critical_pts[0], critical_pts[1])[0]
                        + integrate.quad(function_to_integrate,
                            critical_pts[2], critical_pts[3])[0])
                else:
                    assert False # Should never happen
            return out

        # Perform Integration
        voronoi = voronoi_1d(value, lb=0.)
        I = np.empty((previous_quantizer.value.size,value.size))
        for (i,prev_h) in enumerate(previous_quantizer.value):
            for (j,(lb,ub,h)) in enumerate(
                    zip(voronoi[:-1], voronoi[1:], value)):
                I[i,j] = do_quantized_integration((lb,ub), prev_h, h)
        return I
