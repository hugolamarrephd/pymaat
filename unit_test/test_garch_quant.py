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

NVAR = 4
FIRST_VARIANCE = 1e-4

# Move to semantic utils
class simple_dot_dict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestVoronoi1D:

    def test_voronoi(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([1.,2.,3.]))
        pt.assert_almost_equal(v, np.array([-np.inf,1.5,2.5,np.inf]))

    def test_voronoi_positive(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([1.,2.,3.]), lb=0.)
        pt.assert_almost_equal(v, np.array([0.,1.5,2.5,np.inf]))

    def test_voronoi_negative(self):
        v = pymaat.garch.quant.get_voronoi_1d(np.array([-3.,-2.,-1.]), ub=0.)
        pt.assert_almost_equal(v, np.array([-np.inf,-2.5,-1.5,0.]))

    def test_voronoi_for_matrices_columns(self):
        a_matrix = np.array([[1.,3.],[2.,4.]])
        v = pymaat.garch.quant.get_voronoi_1d(a_matrix, axis=0)
        expected = np.array([[-np.inf,-np.inf],
                [1.5, 3.5],
                [np.inf, np.inf]])
        pt.assert_almost_equal(v, expected)
        pt.assert_equal(a_matrix, [[1.,3.],[2.,4.]])

    def test_voronoi_for_matrices_rows(self):
        a_matrix = np.array([[1.,2.],[3.,4.]])
        v = pymaat.garch.quant.get_voronoi_1d(a_matrix, axis=1)
        expected = np.array([[-np.inf, 1.5, np.inf],
                [-np.inf, 3.5, np.inf]])
        pt.assert_almost_equal(v, expected)
        pt.assert_equal(a_matrix, [[1.,2.],[3.,4.]])

        #TODO test for tensors
        #TODO test when N=1

class TestQuantizerState:
    pass

class TestOneStepVarianceOptimizer:
    pass

class TestFirstVarianceState:
    pass

class TestVarianceState:

    prev_value = 0.18**2./252. * np.array([0.55,0.75,1.,1.43])
    prev_proba = np.array([0.2,0.25,0.45,0.10])
    previous = simple_dot_dict({'value':prev_value, 'probability':prev_proba})

    x = np.array([ 0.47964318,  -1.07353684,  0.86325162,  0.23725981])
    quantizer = pymaat.garch.quant._MarginalVarianceState(
            MODEL, NVAR, previous, x)

    def get_integrand(self, order=0):
        if order==0:
            integrand = lambda z,H,h: norm.pdf(z)
        elif order==1:
            integrand = lambda z,H,h: H * norm.pdf(z)
        elif order==2:
            integrand = lambda z,H,h: H**2. * norm.pdf(z)
        else:
            assert False
        return integrand

    def quantized_integral(self, integrand, value):
        over = 10 # neglect range outside [-over,over]
        value = value.squeeze()
        def do_integration(bounds, prev_h, h):
            assert prev_h.size==1
            assert h.size==1
            assert h>bounds[0] and h<bounds[1]
            def function_to_integrate(z):
                # 0 if between bounds else integrate
                H, _ = MODEL.one_step_generate(z, prev_h)
                if H<bounds[0] or H>bounds[1]:
                    return 0.
                else:
                    return integrand(z,H,h)
            # Help integration by providing discontinuities
            critical_pts = np.hstack([
                MODEL.one_step_roots(prev_h, b) for b in bounds])
            critical_pts = critical_pts[np.isfinite(critical_pts)]
            # Can safely neglect when z outside [-over,over]
            out = integrate.quad(function_to_integrate, -over, over,
                    points=critical_pts)
            return out[0]
        # Do computations
        voronoi = pymaat.garch.quant.get_voronoi_1d(value)
        I = np.empty((NVAR,NVAR))
        for (i,prev_h) in enumerate(self.prev_value):
            for (j,(lb,ub,h)) in enumerate(
                    zip(voronoi[:-1], voronoi[1:], value)):
                I[i,j] = do_integration((lb,ub), prev_h, h)
        return I


    def test_zeroth_order_integral(self):
        expected_value = self.quantized_integral(
                self.get_integrand(order=0),
                self.quantizer.value)
        pt.assert_almost_equal(self.quantizer.integrals[0],
                expected_value, rtol=1e-6,
                msg='Incorrect pdf integral (I0)')

    def test_first_order_integral(self):
        expected_value = self.quantized_integral(
                self.get_integrand(order=1),
                self.quantizer.value)
        pt.assert_almost_equal(self.quantizer.integrals[1],
                expected_value, rtol=1e-4,
                msg='Incorrect MODEL integral (I1)')

    def test_second_order_integral(self):
        expected_value = self.quantized_integral(
                self.get_integrand(order=2),
                self.quantizer.value)
        pt.assert_almost_equal(self.quantizer.integrals[2],
                expected_value, rtol=1e-3,
                msg='Incorrect MODEL squared integral (I2)')

    def assert_integral_derivative(self, order=0, lag=0):
        if lag==0:
            dI = self.quantizer.integral_derivatives[0]
        elif lag==-1:
            dI = self.quantizer.integral_derivatives[1]
        else:
            assert False
        assert order==0 or order==1

        for j in range(NVAR-np.abs(lag)):

            def func(h):
                new_value = self.quantizer.value.copy()
                new_value[j+(lag==1)] = h
                I = self.quantized_integral(
                    self.get_integrand(order=order),
                    new_value)
                return I[:,j+(lag==-1)]

            derivative = dI[order][:,j+(lag==-1)]

            pt.assert_derivative_at(derivative, func,
                    self.quantizer.value[j+(lag==1)], rtol=1e-3)

    def test_zeroth_order_integral_derivative_lag_m1(self):
        self.assert_integral_derivative(order=0, lag=-1)

    def test_first_order_integral_derivative_lag_m1(self):
        self.assert_integral_derivative(order=1, lag=-1)

    def test_zeroth_order_integral_derivative_lag_0(self):
        self.assert_integral_derivative(order=0, lag=0)

    def test_first_order_integral_derivative_lag_0(self):
        self.assert_integral_derivative(order=1, lag=0)

    def distortion(self, value):
        distortion = self.quantized_integral(
                lambda z,H,h: (H-h)**2. * norm.pdf(z), value)
        return self.prev_proba.dot(distortion).sum()

    def test_distortion(self):
        pt.assert_almost_equal(self.quantizer.distortion,
                self.distortion(self.quantizer.value),
                msg='Incorrect distortion', rtol=1e-2)

    def test_distortion_gradient(self):
        pt.assert_gradient_at(self.quantizer.gradient, self.distortion,
                self.quantizer.value, rtol=1e-2)

    def test_distortion_hessian(self):
        pt.assert_hessian_at(self.quantizer.hessian, self.distortion,
                self.quantizer.value, rtol=1e-2, atol=1e-2)

    def test_minimal_variance(self):
        pt.assert_true(self.quantizer.h_min>0.,
                msg='minimum variance must be positive')
        pt.assert_true(self.quantizer.h_min<self.prev_value[0],
                msg='minimum variance is strictly decreasing in time')

    def assert_inv_trans_is_in_space_for(self, x):
        pt.assert_true(np.diff(self.quantizer.value)>0, msg='is not sorted')
        # The first value point could be equal to minimum variance
        # => test second point ...
        pt.assert_true(self.quantizer.value[1]>self.quantizer.h_min)
        # The first voronoi point must be strictly greater than minimum
        # variance
        pt.assert_true(0.5*(self.quantizer.value[0]+self.quantizer.value[1])
                >self.quantizer.h_min,
                msg='First voronoi tile has no area above minimum variance')

    def test_inv_trans_is_in_space_at_few_values(self):
        self.assert_inv_trans_is_in_space_for(
                np.array([-0.213,0.432,0.135,0.542]))
        self.assert_inv_trans_is_in_space_for(
                np.array([-5.123,-3.243,5.234,2.313]))
        self.assert_inv_trans_is_in_space_for(
                np.array([3.234,-6.3123,-5.123,0.542]))

    def transform(self, x):
        quantizer = pymaat.garch.quant._MarginalVarianceState(
                MODEL, NVAR, self.previous, x)
        return quantizer.value

    def test_trans_jacobian(self):
        pt.assert_jacobian_at(self.quantizer.jacobian, self.transform,
                self.x, rtol=1e-6, atol=1e-8)

    # Test transformed distortion function

    def distortion_from_x(self, x):
        value = self.transform(x)
        distortion = self.distortion(value)
        return distortion

    def test_distortion_from_x_gradient(self):
        pt.assert_gradient_at(self.quantizer.transformed_gradient,
             self.distortion_from_x, self.x, rtol=1e-2)

    def test_distortion_from_x_hessian(self):
        pt.assert_hessian_at(self.quantizer.transformed_hessian,
                self.distortion_from_x, self.x, rtol=1e-2)

    # Test transition probabilities

    def test_trans_proba_size(self):
        pt.assert_equal(self.quantizer.transition_probability.shape, (NVAR, NVAR))

    def test_trans_proba_sum_to_one_and_non_negative(self):
        trans = self.quantizer.transition_probability
        pt.assert_true(trans>=0)
        pt.assert_almost_equal(np.sum(trans,axis=1),1)

    def test_trans_proba(self):
        value = self.quantizer.transition_probability
        expected_value = self.quantized_integral(self.get_integrand(order=0),
                self.quantizer.value)
        pt.assert_almost_equal(value, expected_value)
