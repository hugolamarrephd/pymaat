import unittest
from functools import partial

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing
import pymaat.model
import pymaat.quant

VAR_LEVEL = pymaat.quant.VAR_LEVEL

class TestMarginalVarianceQuantizer(pymaat.testing.TestCase):

    model = pymaat.model.Garch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)
    nper = 3
    nquant = 4
    quant = pymaat.quant.MarginalVarianceQuantizer(
            model, nper=nper, nquant=nquant)

    prev_grid = VAR_LEVEL * np.array(
        [0.55812529,0.7532432,1.,1.4324])
    prev_grid.sort()
    grid = prev_grid.copy()
    prev_proba = np.array([0.2,0.25,0.45,0.10])
    assert prev_proba.sum() == 1.

    prev_grid = prev_grid[:,np.newaxis]
    grid = grid[np.newaxis,:]
    prev_proba = prev_proba[np.newaxis,:]

    voronoi = quant._get_voronoi(grid)
    roots = quant._get_roots(prev_grid, voronoi)
    factor = quant._get_factor_of_integral_derivative(
            prev_grid, voronoi, roots)

    def quantized_integral(self, func, grid):
        OVER = 10 # neglect range outside [-over,over]
        prev_grid = self.prev_grid.squeeze()
        grid = grid.squeeze()
        def do_integration(bounds, prev_h, h):
            assert prev_h.size==1
            assert h.size==1
            assert h>bounds[0] and h<bounds[1]
            def function_to_integrate(z):
                # 0 if between bounds else func
                H, _ = self.model.one_step_simulate(z, prev_h)
                if H<bounds[0] or H>bounds[1]:
                    return 0.
                else:
                    return func(z,H,h)
            # Help integration by providing discontinuities
            critical_pts = np.hstack([
                self.model.one_step_roots(prev_h, b) for b in bounds])
            critical_pts = critical_pts[np.isfinite(critical_pts)]
            # Can safely neglect when z outside [-over,over]
            out = integrate.quad(function_to_integrate, -OVER, OVER,
                    points=critical_pts)
            return out[0]
        # Do computations
        voronoi = self.quant._get_voronoi(grid[np.newaxis,:])
        voronoi = voronoi.squeeze()
        I = np.empty((self.nquant,self.nquant))
        for (i,prev_h) in enumerate(prev_grid):
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
        self.assert_equal(trans.shape,
                (self.nper, self.nquant, self.nquant))
        # First grid
        self.assert_equal(grid[0,1:]==first_variance, True)

    def test_get_voronoi_2d(self):
        v = self.quant._get_voronoi(np.array(
                [[1,2,3],
                [11,12,13]]))
        self.assert_almost_equal(v, np.array(
                [[0,1.5,2.5,np.inf],
                [0,11.5,12.5,np.inf]]))

    def test_zeroth_order_integral(self):
        roots = self.quant._get_voronoi_roots(self.prev_grid, self.grid)
        value = self.quant._get_integral(self.prev_grid, roots, order=0)
        expected_value = self.quantized_integral(
                lambda z,H,h: norm.pdf(z), self.grid)
        self.assert_almost_equal(value, expected_value, rtol=1e-6,
                msg='Incorrect pdf integral (I0)')

    def test_first_order_integral(self):
        roots = self.quant._get_voronoi_roots(self.prev_grid, self.grid)
        value = self.quant._get_integral(self.prev_grid, roots, order=1)
        expected_value = self.quantized_integral(
                lambda z,H,h: H * norm.pdf(z), self.grid)
        self.assert_almost_equal(value, expected_value, rtol=1e-4,
                msg='Incorrect model integral (I1)')

    def test_second_order_integral(self):
        roots = self.quant._get_voronoi_roots(self.prev_grid, self.grid)
        value = self.quant._get_integral(self.prev_grid, roots, order=2)
        expected_value = self.quantized_integral(
                lambda z,H,h: H**2. * norm.pdf(z), self.grid)
        self.assert_almost_equal(value, expected_value, rtol=1e-3,
                msg='Incorrect model squared integral (I2)')

    def assert_integral_derivative(self, order=0, lag=0):
        if order==0:
            integrand = lambda z,H,h: norm.pdf(z)
        elif order==1:
            integrand = lambda z,H,h: H * norm.pdf(z)
        elif order==2:
            integrand = lambda z,H,h: H**2. * norm.pdf(z)

        for j in range(self.nquant-np.abs(lag)):

            def func(h):
                new_grid = self.grid.copy()
                new_grid[0,j+(lag==1)] = h
                I = self.quantized_integral(integrand, new_grid)
                return I[:,j+(lag==-1)]

            def derivative(_):
                out = self.quant._get_integral_derivative(
                        self.factor, self.voronoi, order=order, lag=lag)
                return out[:,j+(lag==-1)]

            self.assert_derivative_at(derivative,
                    func, self.grid[0,j+(lag==1)], rtol=1e-3)

    def test_zeroth_order_integral_derivative_lag_m1(self):
        self.assert_integral_derivative(order=0, lag=-1)

    def test_first_order_integral_derivative_lag_m1(self):
        self.assert_integral_derivative(order=1, lag=-1)

    def test_second_order_integral_derivative_lag_m1(self):
        self.assert_integral_derivative(order=2, lag=-1)

    def test_zeroth_order_integral_derivative_lag_0(self):
        self.assert_integral_derivative(order=0, lag=0)

    def test_first_order_integral_derivative_lag_0(self):
        self.assert_integral_derivative(order=1, lag=0)

    def test_second_order_integral_derivative_lag_0(self):
        self.assert_integral_derivative(order=2, lag=0)

    def test_zeroth_order_integral_derivative_lag_1(self):
        self.assert_integral_derivative(order=0, lag=1)

    def test_first_order_integral_derivative_lag_1(self):
        self.assert_integral_derivative(order=1, lag=1)

    def test_second_order_integral_derivative_lag_1(self):
        self.assert_integral_derivative(order=2, lag=1)

    def marginal_distortion(self, grid):
        distortion = self.quantized_integral(
                lambda z,H,h: (H-h)**2. * norm.pdf(z), grid)
        return self.prev_proba.dot(distortion).sum()

    def test_one_step_distortion(self):
        value, _, _ = self.quant._do_one_step_distortion(
                        self.prev_grid,
                        self.prev_proba,
                        self.grid)
        expected_value = self.marginal_distortion(self.grid)
        self.assert_almost_equal(value, expected_value,
                msg='Incorrect distortion', rtol=1e-5)

    def test_one_step_distortion_gradient(self):
        def gradient(grid):
                _, grad, _ = self.quant._do_one_step_distortion(
                        self.prev_grid,
                        self.prev_proba,
                        grid[np.newaxis,:])
                return grad
        func = self.marginal_distortion
        self.assert_gradient_at(gradient, func, self.grid.squeeze(),
                rtol=1e-5)

    def test_one_step_distortion_hessian(self):
        def hessian(grid):
            _, _, hess = self.quant._do_one_step_distortion(
                    self.prev_grid,
                    self.prev_proba,
                    grid[np.newaxis,:])
            return hess
        func = self.marginal_distortion
        self.assert_hessian_at(hessian, func, self.grid.squeeze(),
                rtol=1e-3, atol=1e-5)

    def test_trans_reverts(self):
        x = np.array([-0.213,0.432,0.135,0.542])
        grid = self.quant._inverse_transform(
                self.prev_grid.squeeze(), x)
        x_ = self.quant._transform(
                self.prev_grid.squeeze(), grid)
        self.assert_almost_equal(x, x_)

    def assert_inv_trans_is_in_space_for(self, x):
        x = np.array(x)
        grid = self.quant._inverse_transform(
                self.prev_grid.squeeze(), x)
        h_ = self.quant._get_minimal_variance(self.prev_grid.squeeze())
        self.assert_equal(np.diff(grid)>0, True, msg='is not sorted')
        self.assert_true(grid[1]>h_)
        self.assert_true(0.5*(grid[0]+grid[1])>h_)

    def test_inv_trans_is_in_space(self):
        self.assert_inv_trans_is_in_space_for([-0.213,0.432,0.135,0.542])
        self.assert_inv_trans_is_in_space_for([-5.123,-3.243,5.234,2.313])
        self.assert_inv_trans_is_in_space_for([3.234,-6.3123,-5.123,0.542])

    def test_trans_jacobian(self):
        x = np.array([-0.213,0.432,0.135,0.542])
        jac = partial(self.quant._trans_jacobian,
                self.prev_grid.squeeze())
        func = partial(self.quant._inverse_transform,
                self.prev_grid.squeeze())
        self.assert_jacobian_at(jac, func, x, rtol=1e-6, atol=1e-8)

    # Test transformed distortion function

    def marginal_distortion_transformed(self, x):
        grid = self.quant._inverse_transform(self.prev_grid.squeeze(), x)
        distortion = self.marginal_distortion(grid)
        return distortion/VAR_LEVEL**2.

    def test_one_step_distortion_transformed(self):
        at = np.array([ 0.47964318,  1.07353684,  0.86325162,  0.23725981])
        # Test distortion
        value = self.quant._transformed_distortion(
                    self.prev_grid.squeeze(),
                    self.prev_proba.squeeze(),
                    at)
        expected_value = self.marginal_distortion_transformed(at)
        self.assert_almost_equal(value, expected_value,
                msg='Incorrect distortion')

    def test_one_step_distortion_transformed_gradient(self):
        at = np.array([ 0.47964318,  1.07353684,  0.86325162,  0.23725981])
        def gradient(x):
                grad = self.quant._transformed_distortion_gradient(
                    self.prev_grid.squeeze(),
                    self.prev_proba.squeeze(),
                    x)
                return grad
        func = self.marginal_distortion_transformed
        self.assert_gradient_at(gradient, func, at)

    def test_one_step_distortion_transformed_hessian(self):
        at = np.array([ 0.47964318,  1.07353684,  0.86325162,  0.23725981])
        def hessian(x):
                hess = self.quant._transformed_distortion_hessian(
                    self.prev_grid.squeeze(),
                    self.prev_proba.squeeze(),
                    x)
                return hess
        func = self.marginal_distortion_transformed
        self.assert_hessian_at(hessian, func, at, rtol=1e-2)

    # Test transition probabilities

    def test_trans_proba_size(self):
        trans = self.quant._transition_probability(
                self.prev_grid.squeeze(),
                self.grid.squeeze())
        self.assert_equal(trans.shape, (self.nquant, self.nquant))

    def test_trans_proba_sum_to_one_and_non_negative(self):
        trans = self.quant._transition_probability(
                self.prev_grid.squeeze(),
                self.grid.squeeze())
        self.assert_equal(trans>=0, True)
        self.assert_almost_equal(np.sum(trans,axis=1),1)

    def test_trans_proba(self):
        value = self.quant._transition_probability(
                self.prev_grid.squeeze(),
                self.grid.squeeze())
        expected_value = self.quantized_integral(
                lambda z,H,h: norm.pdf(z), self.grid)
        self.assert_almost_equal(value, expected_value)

    def test_transition_probability_from_first_grid(self):
        first_variance = 1e-4
        grid, *_ = self.quant._initialize(first_variance)
        trans = self.quant._transition_probability(
                grid[0],self.grid.squeeze())
        first_row = trans[0]
        self.assert_equal(trans==first_row, True)

    def test_one_step_quantize_is_sorted(self):
        (_, _, optim_grid) = self.quant._one_step_quantize(
                self.prev_grid.squeeze(), self.prev_proba.squeeze())
        self.assert_almost_equal(optim_grid, np.sort(optim_grid))

    def get_optim_grid_from(self, init):
        (success, fun, optim) = self.quant._one_step_quantize(
                self.prev_grid.squeeze(), self.prev_proba.squeeze(),
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
