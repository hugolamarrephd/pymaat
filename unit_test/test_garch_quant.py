import pytest
from functools import partial

import pytest
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing as pt
import pymaat.garch
import pymaat.garch.quant

MODEL = pymaat.garch.Garch(
        mu=2.01,
        omega=9.75e-20,
        alpha=4.54e-6,
        beta=0.79,
        gamma=196.21)

VARIANCE_SCALE = 0.18**2./252.

# TODO: move to math utilities tests
class TestVoronoi1D:
    # TODO: test tensors

    def test_vector(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([1.,2.,3.]))
        pt.assert_almost_equal(v, np.array([-np.inf,1.5,2.5,np.inf]))

    def test_vector_positive(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([1.,2.,3.]), lb=0.)
        pt.assert_almost_equal(v, np.array([0.,1.5,2.5,np.inf]))

    def test_vector_negative(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([-3.,-2.,-1.]), ub=0.)
        pt.assert_almost_equal(v, np.array([-np.inf,-2.5,-1.5,0.]))

    def test_vector_when_N_is_one(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([1.]), lb=0., ub=2.)
        pt.assert_almost_equal(v, np.array([0.,2.]))

    def test_vector_when_N_is_zero(self):
        with pytest.raises(ValueError):
            pymaat.garch.quant.get_voronoi_1d(np.array([]))

    # Supports matrices
    def test_matrices_columns(self):
        a_matrix = np.array([[1.,3.],[2.,4.]])
        v = pymaat.garch.quant.get_voronoi_1d(a_matrix, axis=0)
        expected = np.array([[-np.inf,-np.inf],
                [1.5, 3.5],
                [np.inf, np.inf]])
        pt.assert_almost_equal(v, expected)
        pt.assert_equal(a_matrix, [[1.,3.],[2.,4.]])

    def test_matrices_rows(self):
        a_matrix = np.array([[1.,2.],[3.,4.]])
        v = pymaat.garch.quant.get_voronoi_1d(a_matrix, axis=1)
        expected = np.array([[-np.inf, 1.5, np.inf],
                [-np.inf, 3.5, np.inf]])
        pt.assert_almost_equal(v, expected)
        pt.assert_equal(a_matrix, [[1.,2.],[3.,4.]])

    def test_matrices_columns_when_N_is_one(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([[1.,1.5]]),
            lb=0., ub=2., axis=0)
        pt.assert_almost_equal(v, np.array([[0.,0.],[2.,2.]]))

    def test_matrices_rows_when_N_is_one(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([[1.],[1.5]]),
            lb=0., ub=2., axis=1)
        pt.assert_almost_equal(v, np.array([[0.,2.],[0.,2.]]))

    def test_matrices_when_N_is_zero(self):
        with pytest.raises(ValueError):
            pymaat.garch.quant.get_voronoi_1d(np.array([[]]), axis=0)
        with pytest.raises(ValueError):
            pymaat.garch.quant.get_voronoi_1d(np.array([[]]), axis=1)


class TestVarianceState:

    @pytest.fixture(params=[
        pymaat.garch.quant.MarginalVarianceQuantization.Quantizer(
            value=VARIANCE_SCALE*np.array([1.]),
            probability=np.array([1.]),
            transition_probability=None),
        pymaat.garch.quant.MarginalVarianceQuantization.Quantizer(
            value=VARIANCE_SCALE*np.array([0.95,1.,1.25]),
            probability=np.array([0.25,0.25,0.5]),
            transition_probability=None),
        pymaat.garch.quant.MarginalVarianceQuantization.Quantizer(
            value=VARIANCE_SCALE*np.array([0.55,0.75,1.,1.43]),
            probability=np.array([0.2,0.25,0.45,0.10]),
            transition_probability=None)],
        ids=['prev_scalar','prev_len3','prev_len4'])
    def previous_quantizer(self, request):
        return request.param

    @pytest.fixture(params=[
            np.array([0.25]),
            np.array([3.234,-6.3123,-5.123]),
            np.array([-0.213,0.135,0.542]),
            np.array([-5.123,-3.243,5.234,2.313]),
            np.array([0.47, -1.07,  0.86,  0.23]),
            ],
            ids = [
                'x_scalar',
                'x_len3_1',
                'x_len3_2',
                'x_len4_1',
                'x_len4_2'])
    def x(self, request):
        return request.param


    @pytest.fixture
    def state(self, previous_quantizer, x):
        optimizer = pymaat.garch.quant._MarginalVarianceOptimizer(
                MODEL, x.size, previous_quantizer)
        return optimizer.eval_at(x)

    @staticmethod
    def get_integrand(order=0):
        if order==0:
            integrand = lambda z,H,h: norm.pdf(z)
        elif order==1:
            integrand = lambda z,H,h: H * norm.pdf(z)
        elif order==2:
            integrand = lambda z,H,h: H**2. * norm.pdf(z)
        else:
            assert False
        return integrand

    def _quantized_integral(self,
            previous_quantizer, integrand, value):
        """
        This helper function (numerically) computes the integral of
        the integrand over quantized tiles
        """
        voronoi = pymaat.garch.quant.get_voronoi_1d(value)
        I = np.empty((previous_quantizer.value.size,value.size))
        for (i,prev_h) in enumerate(previous_quantizer.value):
            for (j,(lb,ub,h)) in enumerate(
                    zip(voronoi[:-1], voronoi[1:], value)):
                I[i,j] = self._do_quantized_integration(
                        integrand, (lb,ub), prev_h, h)
        return I

    def _do_quantized_integration(self, integrand,
            tile_bounds, prev_variance, constant=None):
        assert prev_variance.size==1

        # Help integration by computing discontinuities...
        critical_pts = np.concatenate(
                [MODEL.one_step_roots(prev_variance, b)
                for b in tile_bounds])
        critical_pts = critical_pts[np.isfinite(critical_pts)]

        # Neglect integral outside [-lim, lim]
        INNOV_LIM = np.array([10.])
        if critical_pts.size>0:
            lim = (np.amin(np.concatenate([-INNOV_LIM, critical_pts])),
                np.amax(np.concatenate([INNOV_LIM, critical_pts])))
        else:
            lim = (-INNOV_LIM, INNOV_LIM)

        # Define function to integrate
        def function_to_integrate(innov):
            assert isinstance(innov, float)
            # 0 if innovation is between tile bounds else integrate
            next_variance, _ = MODEL.one_step_generate(innov, prev_variance)
            if next_variance<tile_bounds[0] or next_variance>tile_bounds[1]:
                return 0.
            else:
                return integrand(innov, next_variance, constant)

        # Perform integration by quadrature
        out = integrate.quad(function_to_integrate, lim[0], lim[1],
                points=critical_pts)

        return out[0]

    def test_zeroth_order_integral(self, previous_quantizer, state):
        expected_value = self._quantized_integral(previous_quantizer,
                self.get_integrand(order=0),
                state.value)
        pt.assert_almost_equal(state.integrals[0],
                expected_value, rtol=1e-6, atol=1e-16,
                msg='Incorrect pdf integral (I0)')

    # @parametrize_x
    # @parametrize_previous_quantizer
    # def test_first_order_integral(self, previous_quantizer, state):
    #     expected_value = self.quantized_integral(
    #             self.get_integrand(order=1),
    #             self.state.value)
    #     pt.assert_almost_equal(self.state.integrals[1],
    #             expected_value, rtol=1e-4,
    #             msg='Incorrect MODEL integral (I1)')

    # def test_second_order_integral(self):
    #     expected_value = self.quantized_integral(
    #             self.get_integrand(order=2),
    #             self.state.value)
    #     pt.assert_almost_equal(self.state.integrals[2],
    #             expected_value, rtol=1e-3,
    #             msg='Incorrect MODEL squared integral (I2)')

    # def assert_integral_derivative(self, order=0, lag=0):
    #     if lag==0:
    #         dI = self.state.integral_derivatives[0]
    #     elif lag==-1:
    #         dI = self.state.integral_derivatives[1]
    #     else:
    #         assert False
    #     assert order==0 or order==1

    #     for j in range(NVAR-np.abs(lag)):

    #         def func(h):
    #             new_value = self.state.value.copy()
    #             new_value[j+(lag==1)] = h
    #             I = self.quantized_integral(
    #                 self.get_integrand(order=order),
    #                 new_value)
    #             return I[:,j+(lag==-1)]

    #         derivative = dI[order][:,j+(lag==-1)]

    #         pt.assert_derivative_at(derivative, func,
    #                 self.state.value[j+(lag==1)], rtol=1e-3)

    # def test_zeroth_order_integral_derivative_lag_m1(self):
    #     self.assert_integral_derivative(order=0, lag=-1)

    # def test_first_order_integral_derivative_lag_m1(self):
    #     self.assert_integral_derivative(order=1, lag=-1)

    # def test_zeroth_order_integral_derivative_lag_0(self):
    #     self.assert_integral_derivative(order=0, lag=0)

    # def test_first_order_integral_derivative_lag_0(self):
    #     self.assert_integral_derivative(order=1, lag=0)

    # def distortion(self, value):
    #     distortion = self.quantized_integral(
    #             lambda z,H,h: (H-h)**2. * norm.pdf(z), value)
    #     return self.prev_proba.dot(distortion).sum()

    # def test_distortion(self):
    #     pt.assert_almost_equal(self.state.distortion,
    #             self.distortion(self.state.value),
    #             msg='Incorrect distortion', rtol=1e-2)

    # def test_distortion_gradient(self):
    #     pt.assert_gradient_at(self.state.gradient, self.distortion,
    #             self.state.value, rtol=1e-2)

    # def test_distortion_hessian(self):
    #     pt.assert_hessian_at(self.state.hessian, self.distortion,
    #             self.state.value, rtol=1e-2, atol=1e-2)

    # def test_minimal_variance(self):
    #     pt.assert_true(self.state.h_min>0.,
    #             msg='minimum variance must be positive')
    #     pt.assert_true(self.state.h_min<self.prev_value[0],
    #             msg='minimum variance is strictly decreasing in time')

    # def assert_inv_trans_is_in_space_for(self, x):
    #     pt.assert_true(np.diff(self.state.value)>0, msg='is not sorted')
    #     # The first value point could be equal to minimum variance
    #     # => test second point ...
    #     pt.assert_true(self.state.value[1]>self.state.h_min)
    #     # The first voronoi point must be strictly greater than minimum
    #     # variance
    #     pt.assert_true(0.5*(self.state.value[0]+self.state.value[1])
    #             >self.state.h_min,
    #             msg='First voronoi tile has no area above minimum variance')

    # def test_inv_trans_is_in_space_at_few_values(self):
    #     self.assert_inv_trans_is_in_space_for(
    #             np.array([-0.213,0.432,0.135,0.542]))
    #     self.assert_inv_trans_is_in_space_for(
    #             np.array([-5.123,-3.243,5.234,2.313]))
    #     self.assert_inv_trans_is_in_space_for(
    #             np.array([3.234,-6.3123,-5.123,0.542]))

    # def transform(self, x):
    #     state = pymaat.garch.quant._MarginalVarianceState(
    #             MODEL, NVAR, self.previous, x)
    #     return state.value

    # def test_trans_jacobian(self):
    #     pt.assert_jacobian_at(self.state.jacobian, self.transform,
    #             self.x, rtol=1e-6, atol=1e-8)

    # # Test transformed distortion function

    # def distortion_from_x(self, x):
    #     value = self.transform(x)
    #     distortion = self.distortion(value)
    #     return distortion

    # def test_distortion_from_x_gradient(self):
    #     pt.assert_gradient_at(self.state.transformed_gradient,
    #          self.distortion_from_x, self.x, rtol=1e-2)

    # def test_distortion_from_x_hessian(self):
    #     pt.assert_hessian_at(self.state.transformed_hessian,
    #             self.distortion_from_x, self.x, rtol=1e-2)

    # # Test transition probabilities

    # def test_trans_proba_size(self):
    #     pt.assert_equal(self.state.transition_probability.shape,
    #             (NVAR, NVAR))

    # def test_trans_proba_sum_to_one_and_non_negative(self):
    #     trans = self.state.transition_probability
    #     pt.assert_true(trans>=0)
    #     pt.assert_almost_equal(np.sum(trans,axis=1),1)

    # def test_trans_proba(self):
    #     value = self.state.transition_probability
    #     expected_value = self.quantized_integral(self.get_integrand(order=0),
    #             self.state.value)
    #     pt.assert_almost_equal(value, expected_value)
