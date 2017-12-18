import unittest
from functools import partial

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing
import pymaat.garch
import pymaat.garch.quant

class TestQuantizer(pymaat.testing.TestCase):

    def test_voronoi(self):
        v = pymaat.garch.get_voronoi(np.array([1.,2.,3.]))
        self.assert_almost_equal(v, np.array([-np.inf,1.5,2.5,np.inf]))

    def test_voronoi_positive(self):
        v = pymaat.garch.get_voronoi(np.array([1.,2.,3.]), lb=0.)
        self.assert_almost_equal(v, np.array([0.,1.5,2.5,np.inf]))

    def test_voronoi_negative(self):
        v = pymaat.garch.get_voronoi(np.array([-3.,-2.,-1.]), ub=0.)
        self.assert_almost_equal(v, np.array([-np.inf,-2.5,-1.5,0.]))

class TestMarginalVarianceQuantizer(pymaat.testing.TestCase):

    model = pymaat.garch.Garch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)

    nper = 3
    nquant = 4
    quant = pymaat.garch.VarianceQuantizer(
            model, nper=nper, nquant=nquant)

    prev_grid = 0.18**2./252. * np.array([0.55,0.75,1.,1.43])
    prev_proba = np.array([0.2,0.25,0.45,0.10])
    state = pymaat.garch.quant._VarianceQuantizerState(
            quant, prev_grid, prev_proba)

    x = np.array([ 0.47964318,  -1.07353684,  0.86325162,  0.23725981])
    eval_ = pymaat.garch.quant._VarianceQuantizerEval(state, x)

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

    def quantized_integral(self, integrand, grid):
        OVER = 10 # neglect range outside [-over,over]
        grid = grid.squeeze()
        def do_integration(bounds, prev_h, h):
            assert prev_h.size==1
            assert h.size==1
            assert h>bounds[0] and h<bounds[1]
            def function_to_integrate(z):
                # 0 if between bounds else integrate
                H, _ = self.model.one_step_generate(z, prev_h)
                if H<bounds[0] or H>bounds[1]:
                    return 0.
                else:
                    return integrand(z,H,h)
            # Help integration by providing discontinuities
            critical_pts = np.hstack([
                self.model.one_step_roots(prev_h, b) for b in bounds])
            critical_pts = critical_pts[np.isfinite(critical_pts)]
            # Can safely neglect when z outside [-over,over]
            out = integrate.quad(function_to_integrate, -OVER, OVER,
                    points=critical_pts)
            return out[0]
        # Do computations
        voronoi = pymaat.garch.get_voronoi(grid)
        I = np.empty((self.nquant,self.nquant))
        for (i,prev_h) in enumerate(self.prev_grid):
            for (j,(lb,ub,h)) in enumerate(
                    zip(voronoi[:-1], voronoi[1:], grid)):
                I[i,j] = do_integration((lb,ub), prev_h, h)
        return I

    def test_init(self):
        self.assert_equal(self.quant.nper, self.nper)
        self.assert_equal(self.quant.nquant, self.nquant)

    def test_initialize(self):
        first_variance = 1e-4
        grid, proba, trans = self.quant._initialize(first_variance)
        # Shapes
        self.assert_equal(grid.shape, (self.nper+1, self.nquant))
        self.assert_equal(proba.shape, (self.nper+1, self.nquant))
        self.assert_equal(trans.shape, (self.nper, self.nquant, self.nquant))
        # First grid
        self.assert_equal(grid[0,1:]==first_variance, True)

    def test_zeroth_order_integral(self):
        expected_value = self.quantized_integral(
                self.get_integrand(order=0),
                self.eval_.grid)
        self.assert_almost_equal(self.eval_.integrals[0],
                expected_value, rtol=1e-6,
                msg='Incorrect pdf integral (I0)')

    def test_first_order_integral(self):
        expected_value = self.quantized_integral(
                self.get_integrand(order=1),
                self.eval_.grid)
        self.assert_almost_equal(self.eval_.integrals[1],
                expected_value, rtol=1e-4,
                msg='Incorrect model integral (I1)')

    def test_second_order_integral(self):
        expected_value = self.quantized_integral(
                self.get_integrand(order=2),
                self.eval_.grid)
        self.assert_almost_equal(self.eval_.integrals[2],
                expected_value, rtol=1e-3,
                msg='Incorrect model squared integral (I2)')

    def assert_integral_derivative(self, order=0, lag=0):
        if lag==0:
            dI = self.eval_.integral_derivatives[0]
        elif lag==-1:
            dI = self.eval_.integral_derivatives[1]
        else:
            assert False
        assert order==0 or order==1

        for j in range(self.nquant-np.abs(lag)):

            def func(h):
                new_grid = self.eval_.grid.copy()
                new_grid[j+(lag==1)] = h
                I = self.quantized_integral(
                    self.get_integrand(order=order),
                    new_grid)
                return I[:,j+(lag==-1)]

            derivative = dI[order][:,j+(lag==-1)]

            self.assert_derivative_at(derivative,
                    func, self.eval_.grid[j+(lag==1)], rtol=1e-3)

    def test_zeroth_order_integral_derivative_lag_m1(self):
        self.assert_integral_derivative(order=0, lag=-1)

    def test_first_order_integral_derivative_lag_m1(self):
        self.assert_integral_derivative(order=1, lag=-1)

    def test_zeroth_order_integral_derivative_lag_0(self):
        self.assert_integral_derivative(order=0, lag=0)

    def test_first_order_integral_derivative_lag_0(self):
        self.assert_integral_derivative(order=1, lag=0)

    def distortion(self, grid):
        distortion = self.quantized_integral(
                lambda z,H,h: (H-h)**2. * norm.pdf(z), grid)
        return self.prev_proba.dot(distortion).sum()

    def test_distortion(self):
        self.assert_almost_equal(self.eval_.distortion,
                self.distortion(self.eval_.grid),
                msg='Incorrect distortion', rtol=1e-2)

    def test_distortion_gradient(self):
        self.assert_gradient_at(self.eval_.gradient,
                self.distortion,
                self.eval_.grid,
                rtol=1e-2)

    def test_distortion_hessian(self):
        self.assert_hessian_at(self.eval_.hessian,
                self.distortion,
                self.eval_.grid,
                rtol=1e-2, atol=1e-5)

    def test_minimal_variance(self):
        self.assert_true(self.state.h_min>0.,
                msg='minimum variance must be positive')
        self.assert_true(self.state.h_min<self.prev_grid[0],
                msg='minimum variance is strictly decreasing in time')

    def assert_inv_trans_is_in_space_for(self, x):
        self.assert_true(np.diff(self.eval_.grid)>0, msg='is not sorted')
        # The first grid point could be equal to minimum variance
        # => test second point ...
        self.assert_true(self.eval_.grid[1]>self.state.h_min)
        # The first voronoi point must be strictly greater than minimum
        # variance
        self.assert_true(
                0.5*(self.eval_.grid[0]+self.eval_.grid[1])
                >self.state.h_min,
                msg='First voronoi tile has no area above minimum variance')

    def test_inv_trans_is_in_space(self):
        self.assert_inv_trans_is_in_space_for(
                np.array([-0.213,0.432,0.135,0.542]))
        self.assert_inv_trans_is_in_space_for(
                np.array([-5.123,-3.243,5.234,2.313]))
        self.assert_inv_trans_is_in_space_for(
                np.array([3.234,-6.3123,-5.123,0.542]))

    def transform(self, x):
        eval_ = pymaat.garch.quant._VarianceQuantizerEval(self.state, x)
        return eval_.grid

    def test_trans_jacobian(self):
        self.assert_jacobian_at(self.eval_.jacobian,
                self.transform,
                self.x, rtol=1e-6, atol=1e-8)

    # Test transformed distortion function

    def distortion_from_x(self, x):
        grid = self.transform(x)
        distortion = self.distortion(grid)
        return distortion

    def test_distortion_from_x_gradient(self):
        self.assert_gradient_at(self.eval_.transformed_gradient,
             self.distortion_from_x, self.x, rtol=1e-2)

    def test_distortion_from_x_hessian(self):
        self.assert_hessian_at(self.eval_.transformed_hessian,
                self.distortion_from_x, self.x, rtol=1e-2)

    # Test transition probabilities

    def test_trans_proba_size(self):
        self.assert_equal(self.eval_.transition_probability.shape,
                (self.nquant, self.nquant))

    def test_trans_proba_sum_to_one_and_non_negative(self):
        trans = self.eval_.transition_probability
        self.assert_equal(trans>=0, True)
        self.assert_almost_equal(np.sum(trans,axis=1),1)

    def test_trans_proba(self):
        value = self.eval_.transition_probability
        expected_value = self.quantized_integral(
                self.get_integrand(order=0),
                self.eval_.grid)
        self.assert_almost_equal(value, expected_value)

    def get_optim_grid_from(self, init):
        (success, optim) = self.quant._one_step_quantize(self.state,
                init=init)
        assert success
        return optim

    def test_one_step_quantize_converges_to_same_from_diff_init(self):
        optim1 = self.get_optim_grid_from(
            np.array([0.57234,-0.324234,3.4132,-2.123]))
        optim2 = self.get_optim_grid_from(
            np.array([-1.423,0.0234,-3.234,5.324]))
        optim3 = self.get_optim_grid_from(
            np.array([-0.0234,0.,0.002,1.234]))
        optim4 = self.get_optim_grid_from(
            np.array([-10,2.32,5.324,10.234]))
        self.assert_almost_equal(optim1, optim2)
        self.assert_almost_equal(optim2, optim3)
        self.assert_almost_equal(optim3, optim4)

    def test_quantize_returns_valid_quantization(self):
        q = self.quant.quantize(0.18**2./252.)
        self.assert_true(type(q)==pymaat.garch.quant.Quantization)
        for t in range(1,self.nper+1):
            self.assert_almost_equal(
                    q.transition_probabilities[t-1].sum(axis=1), 1.)
            self.assert_almost_equal(
                    q.probabilities[t-1].dot(q.transition_probabilities[t-1]),
                    q.probabilities[t])
            self.assert_almost_equal(q.probabilities[t].sum(), 1.)
            self.assert_true(q.values[t]>0.)
            self.assert_true(np.diff(q.values[t])>0.)
