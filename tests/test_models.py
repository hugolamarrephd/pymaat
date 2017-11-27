import unittest
from functools import partial

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.models


class TestTimeseriesGarchFixture():

    __innovations = np.array(
            [[[ 0.31559906, -1.80551443, -0.49468605, -1.04842416],
            [ 1.47092134, -0.18375408,  0.61439572,  0.64938352]],

           [[-1.62316804, -0.76478879, -0.75826613,  0.17750069],
            [ 0.82250737,  0.10701289, -1.15691368,  1.05573836]],

           [[-0.22774868,  0.74240245,  1.11521215,  1.10464703],
            [-0.56968483,  0.54138324, -0.69508153, -0.14254879]]])

    __returns = __innovations * np.sqrt(1e-4) + 1e-3

    __variances =  1e-4 * np.array(
            [[[ 3.40880883,  1.50217308,  1.11125856,  0.14462167],
            [ 0.08321133,  0.36507093,  1.52133949,  1.42666615]],

           [[ 0.78940018,  0.96257061,  0.3341445 ,  0.59402684],
            [ 1.67863795,  1.21765875,  1.50332767,  1.03224149]],

           [[ 0.78179542,  1.32126548,  1.64830936,  1.59253612],
            [ 0.21397881,  0.01596482,  2.48993567,  1.3034988 ]]])

    __first_variance = __variances[0]

    __shape = __innovations.shape

    def timeseries_filter(self,returns=__returns):
        return self.model.timeseries_filter(
                returns,
                self.__first_variance)

    def timeseries_simulate(self, innovations=__innovations):
        return self.model.timeseries_simulate(
                innovations,
                self.__first_variance)

    def test_simulate_revertsto_filter(self):
        simulated_variances, simulated_returns = self.timeseries_simulate()
        filtered_variances, filtered_innovations = \
                self.timeseries_filter(returns=simulated_returns)
        np.testing.assert_allclose(simulated_variances, filtered_variances)
        np.testing.assert_allclose(self.__innovations, filtered_innovations)

    def test_filter_initialize_variance(self):
        filtered_variance, _ = self.timeseries_filter()
        np.testing.assert_allclose(filtered_variance[0],
                self.__first_variance)

    def test_filter_simulate_variance(self):
        simulated_variance, _ = self.timeseries_simulate()
        np.testing.assert_allclose(simulated_variance[0],
                self.__first_variance)

    def test_filter_size(self):
        next_variances, innovations = self.timeseries_filter()
        self.assertEqual(
                next_variances.shape,
                (self.__shape[0]+1,) + self.__shape[1:])
        self.assertEqual(innovations.shape, self.__shape)

    def test_filter_positive_variance(self):
        variances,_ = self.timeseries_filter()
        np.testing.assert_array_equal(variances>0, True)

    def test_filter_throws_exception_when_non_positive_variance(self):
        with self.assertRaises(ValueError):
            self.model.timeseries_filter(np.nan, -1)
        with self.assertRaises(ValueError):
            self.model.timeseries_filter(np.nan, 0)

    def test_simulate_size(self):
        variances, returns = self.timeseries_simulate()
        self.assertEqual(variances.shape,
                (self.__shape[0]+1,) + self.__shape[1:])
        self.assertEqual(returns.shape, self.__shape)

    def test_simulate_positive_variance(self):
        variances,_ = self.timeseries_simulate()
        np.testing.assert_array_equal(variances>0, True)

    def test_simulate_throws_exception_when_non_positive_variance(self):
        with self.assertRaises(ValueError):
            self.model.timeseries_simulate(np.nan, -1)
        with self.assertRaises(ValueError):
            self.model.timeseries_simulate(np.nan, 0)

class TestOneStepGarchFixture():

    __innovations = np.array(
            [[[ 0.36841639, -0.57224794,  0.25019277, -1.24721141],
            [ 1.35128646, -0.92096528, -0.92572966, -0.98178342],
            [-0.84366922, -1.28207108, -1.53467485, -0.22874378]],

            [[-1.05050049, -1.31158806, -1.17961533,  0.05363092],
            [-1.67960499,  0.02364286,  0.28870878,  0.01871441],
            [ 1.71135472,  0.7167683 , -0.93715744, -0.8482234 ]]])

    __returns = __innovations * np.sqrt(1e-4) + 1e-3

    __variances =  1e-4 * np.array(
            [[[ 0.3217341 ,  0.00868056,  1.22722262,  0.58144423],
            [ 0.54204103,  3.68888772,  0.77486123,  0.95989976],
            [ 0.90888372,  0.07370864,  2.91764007,  2.21766175]],

            [[ 0.7019393 ,  2.38621637,  1.24599797,  1.17731278],
            [ 0.81830769,  0.26109424,  0.2980641 ,  0.85817885],
            [ 0.93831248,  0.17303652,  0.45056502,  2.22328546]]])

    __shape = __innovations.shape

    __innovations_broadcast = np.array([[-0.70687441],
            [ 0.72613442],
            [-2.20412955],
            [-0.27571779],
            [ 1.38167603],
            [-0.96744496],
            [-0.05243877],
            [-0.41254891],
            [-1.27985463],
            [ 0.12262462]])

    __variances_broadcast =  1e-4 * np.array([[ 2.70588279,  0.96680575,
            0.09027765, 1.23017577,  1.14468471]])

    __shape_broadcast = (__innovations_broadcast.shape[0],
            __variances_broadcast.shape[1])

    # I/O

    def get_one_step_funcs(self):
        return [self.model.one_step_filter,
               self.model.one_step_simulate,
               self.model.one_step_has_roots,
               self.model.one_step_roots,
               self.model.one_step_expectation_until]

    def test_all_funcs_support_scalar(self):
        allowed_primitives = (bool, float)
        for f in self.get_one_step_funcs():
        # Actual inputs do not matter here
        # Only testing shapes...
            out = f(np.nan, np.nan)
            if isinstance(out, tuple):
                for o in out:
                    msg_ = 'Func was {}, Type was: {}'.format(f, type(o))
                    self.assertTrue(type(o) in allowed_primitives,
                            msg=msg_)
            else:
                msg_ = 'Func was {}, Type was: {}'.format(f, type(out))
                self.assertTrue(type(out) in allowed_primitives,
                            msg=msg_)

    def test_all_funcs_support_array(self):
        for f in self.get_one_step_funcs():
        # Actual inputs do not matter here
        # Only testing shapes...
            msg_ = 'Func was {}'.format(f)
            out = f(self.__innovations, self.__variances)
            if isinstance(out, tuple):
                for o in out:
                    self.assertEqual(o.shape, self.__shape, msg=msg_)
            else:
                self.assertEqual(out.shape, self.__shape, msg=msg_)

    def test_all_one_step_funcs_support_broadcasting(self):
        for f in self.get_one_step_funcs():
            # Actual inputs do not matter here
            # Only testing shapes...
            msg_ = 'Func was {}'.format(f)
            out = f(self.__innovations_broadcast,
                    self.__variances_broadcast)
            if isinstance(out, tuple):
                for o in out:
                    self.assertEqual(o.shape, self.__shape_broadcast,
                            msg=msg_)
            else:
                self.assertEqual(out.shape, self.__shape_broadcast, msg=msg_)

    # One-step filter

    def one_step_filter(self, returns=__returns):
        return self.model.one_step_filter(returns, self.__variances)

    def one_step_simulate(self, innovations=__innovations):
        return self.model.one_step_simulate(innovations, self.__variances)

    def test_one_step_simulate_revertsto_filter(self):
        simulated_variances, simulated_returns = self.one_step_simulate()
        filtered_variances, filtered_innovations = self.one_step_filter(
                returns=simulated_returns)
        np.testing.assert_allclose(simulated_variances, filtered_variances)
        np.testing.assert_allclose(self.__innovations, filtered_innovations)

    def test_one_step_filter_positive_variance(self):
        next_variances, _ = self.one_step_filter()
        np.testing.assert_array_equal(next_variances>0, True)

    def test_one_step_simulate_positive_variance(self):
        next_variances, _ = self.one_step_simulate()
        np.testing.assert_array_equal(next_variances>0, True)

    # TODO: send to estimator
    # def test_neg_log_like_at_few_values(self):
    #     nll = self.model.negative_log_likelihood(1, 1)
    #     self.assertAlmostEqual(nll, 0.5)
    #     nll = self.model.negative_log_likelihood(0, np.exp(1))
    #     self.assertAlmostEqual(nll, 0.5)

    # One-step innovation roots

    def test_one_step_roots_when_valid(self):
        variances_at_singularity = self.get_lowest_one_step_variance()
        next_variances = variances_at_singularity + 1e-5
        left_roots, right_roots = self.model.one_step_roots(
                self.__variances,
                next_variances)
        for (z,s) in zip((left_roots, right_roots), ('left', 'right')):
            next_variances_solved, _ = self.model.one_step_simulate(
                z, self.__variances)
            np.testing.assert_allclose(next_variances_solved, next_variances,
                    err_msg = 'Invalid ' + s + ' roots.')

    def get_lowest_one_step_variance(self):
        return self.model._get_lowest_one_step_variance(self.__variances)

    def test_one_step_roots_left_right_order(self):
        variances_at_singularity = self.get_lowest_one_step_variance()
        next_variances = variances_at_singularity + 1e-5
        left_roots, right_roots = self.model.one_step_roots(
                self.__variances,
                next_variances)
        np.testing.assert_equal(left_roots<right_roots, True)

    def test_roots_nan_when_below_lowest(self):
        variances_at_singularity = self.get_lowest_one_step_variance()
        impossible_variances = variances_at_singularity - 1e-5
        left_roots, right_roots = self.model.one_step_roots(
                self.__variances, impossible_variances)
        np.testing.assert_equal(left_roots, np.nan)
        np.testing.assert_equal(right_roots, np.nan)

    def test_same_roots_at_singularity(self):
        variances_at_singularity = self.get_lowest_one_step_variance()
        left_roots, right_roots = self.model.one_step_roots(
                self.__variances, variances_at_singularity)
        np.testing.assert_allclose(left_roots, right_roots)

    def test_roots_at_inf_are_pm_inf(self):
        [left_root, right_root] = self.model.one_step_roots(
                self.__variances, np.inf)
        np.testing.assert_equal(left_root, -np.inf)
        np.testing.assert_equal(right_root, np.inf)

    def test_roots_at_zero_are_nan(self):
        [left_root, right_root] = self.model.one_step_roots(
                self.__variances, 0)
        np.testing.assert_equal(left_root, np.nan)
        np.testing.assert_equal(right_root, np.nan)

    # One-step variance integration

    def test_one_step_integrate(self):
        from_ = -0.534
        to_ = 0.123
        variance = 1e-4
        def to_integrate(z):
            (h, _) = self.model.one_step_simulate(z, variance)
            return h * norm.pdf(z)
        expected_value = integrate.quad(to_integrate, from_, to_)
        value = (self.model.one_step_expectation_until(variance, to_)
                - self.model.one_step_expectation_until(variance, from_))
        self.assertAlmostEqual(value, expected_value[0])

    def test_one_step_integrate_zero_at_minf(self):
        variance = 1e-4
        value_at_minf = self.model.one_step_expectation_until(
                variance, -np.inf)
        self.assertAlmostEqual(value_at_minf, 0)

    def test_one_step_integrate_zero_at_inf(self):
        variance = 1e-4
        value_at_inf = self.model.one_step_expectation_until(variance)
        expected_value = (self.model.omega + self.model.alpha
                + (self.model.beta+
                self.model.alpha*self.model.gamma**2) * variance)
        self.assertAlmostEqual(value_at_inf, expected_value)

class TestGarch(TestTimeseriesGarchFixture,
        TestOneStepGarchFixture,
        unittest.TestCase):

    model = pymaat.models.Garch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)

    def test_one_step_filter_at_few_values(self):
        next_var, innov = self.model.one_step_filter(0, 1)
        self.assertAlmostEqual(innov, 0.5-self.model.mu)
        self.assertAlmostEqual(next_var,
                  self.model.omega
                + self.model.beta
                + self.model.alpha * (innov - self.model.gamma) ** 2)

    def test_one_step_simulate_at_few_values(self):
        next_var, ret = self.model.one_step_simulate(0, 0)
        self.assertAlmostEqual(ret, 0)
        self.assertAlmostEqual(next_var, self.model.omega)
        next_var, ret = self.model.one_step_simulate(0, 1)
        self.assertAlmostEqual(ret, self.model.mu-0.5)
        self.assertAlmostEqual(next_var,
                  self.model.omega
                + self.model.beta
                + self.model.alpha * self.model.gamma ** 2)

    def test_invalid_param_raise_exception(self):
        with self.assertRaises(ValueError):
            pymaat.models.Garch(
                mu=0,
                omega=0,
                alpha=0.5,
                beta=0.5,
                gamma=1)

class TestGarchQuantizer(unittest.TestCase):

    model = pymaat.models.Garch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)

    nper = 3
    nquant = 4
    quant = pymaat.models.Quantizer(model, nper=nper, nquant=nquant)

    prev_grid = 1e-4 * np.array([0.50873928,0.55812529,0.91400832,1.73936377])
    prev_proba = np.array([0.1,0.15,0.5,0.25])
    most_proba = 2 # ID to most probable in previous grid

    grid = 1e-4 * np.array([0.0002312,1.06478209,1.32183054,2.35018137])

    def test_init(self):
        self.assertEqual(self.quant.nper, self.nper)
        self.assertEqual(self.quant.nquant, self.nquant)


    def test_initialize(self):
        first_variance = 1e-4
        grid, proba, trans = self.quant._initialize(first_variance)
        # Shapes
        self.assertEqual(grid.shape, (self.nper+1, self.nquant))
        self.assertEqual(proba.shape, (self.nper+1, self.nquant))
        self.assertEqual(trans.shape, (self.nper, self.nquant, self.nquant))
        # First grid
        np.testing.assert_array_equal(grid[0,1:]==first_variance, True)

    def test_get_voronoi_2d(self):
        v = self.quant._get_voronoi(np.array(
                [[1,2,3],
                [11,12,13]]))
        np.testing.assert_almost_equal(v, np.array(
                [[0,1.5,2.5,np.inf],
                [0,11.5,12.5,np.inf]]))

    def quantized_integral(self, func, prev_grid, grid):
        assert prev_grid.size==1
        voronoi = self.quant._get_voronoi(grid[np.newaxis,:])
        voronoi = voronoi[0]
        def do_integration(bounds, pg, g):
            assert g>bounds[0] and g<bounds[1]
            def function_to_integrate(z):
                # 0 if between bounds else integral input
                (h, _) = self.model.one_step_simulate(z, pg)
                if h<bounds[0] or h>bounds[1]:
                    return 0
                else:
                    return func(z,h,g)
            # Help integration by providing discontinuities
            critical_pts = np.hstack([
                self.model.one_step_roots(pg, b) for b in bounds])
            critical_pts = critical_pts[np.isfinite(critical_pts)]
            # Can safely neglect when z outside [-30,30]
            out = integrate.quad(function_to_integrate, -30, 30,
                    points=critical_pts)
            return out[0]
        return np.array([do_integration((lb,ub), prev_grid, g)
            for (lb,ub,g) in zip(voronoi[:-1], voronoi[1:], grid)])

    def test_one_step_integrate(self):
        value = self.quant._one_step_integrate(
                self.prev_grid[:,np.newaxis],
                self.grid[np.newaxis,:])
        expected_value = np.empty_like(value)
        dist_grad = lambda z,h,g: (h-g) * norm.pdf(z)
        for (ev,pg) in zip(expected_value, self.prev_grid):
            ev[:] = self.quantized_integral(dist_grad, pg, self.grid)
        np.testing.assert_almost_equal(value, expected_value)

    def numerical_jacobian(self, func, x):
        assert func(x).size==1
        def derivate_1d_at(f,x):
            epsilon = 1e-8
            # Forward differences to make sure variance stays positive
            return (f(x+epsilon)-f(x))/(epsilon)
        grad = np.empty_like(x)
        for (i,x_) in enumerate(x):
            def func_1d(x_scalar):
                xcopy = x.copy()
                xcopy[i] = x_scalar
                return func(xcopy)
            grad[i] = derivate_1d_at(func_1d, x_)
        return grad

    def test_one_step_gradient(self):
        value = self.quant._one_step_gradient(self.prev_grid,
                self.prev_proba,
                self.grid)
        dist_func = lambda z,h,g: np.power(h-g,2) * norm.pdf(z)
        def marginal_distortion(grid):
            dist = np.empty((self.nquant, self.nquant))
            for (d,pg) in zip(dist, self.prev_grid):
                 d[:] = self.quantized_integral(dist_func, pg, grid)
            return self.prev_proba.dot(dist).sum()
        expected_value = self.numerical_jacobian(
                marginal_distortion, self.grid)
        np.testing.assert_almost_equal(value, expected_value)

    def test_transformation(self):
        x = np.array(range(-20,20), float)
        grid = self.quant._inv_transform(x, self.prev_grid)
        x_ = self.quant._transform(grid, self.prev_grid)
        np.testing.assert_almost_equal(x, x_)

    def test_one_step_gradient_transformed(self):
        # By-pass _transform and _inv_transform
        # Test should pass with log transformation only
        value = self.quant._one_step_gradient_transformed(self.prev_grid,
                self.prev_proba,
                np.log(self.grid))
        dist_func = lambda z,h,g: np.power(h-g,2) * norm.pdf(z)
        def marginal_distortion(log_grid):
            grid = np.exp(log_grid)
            dist = np.empty((self.nquant, self.nquant))
            for (d,pg) in zip(dist, self.prev_grid):
                 d[:] = self.quantized_integral(dist_func, pg, grid)
            return self.prev_proba.dot(dist).sum()
        expected_value = self.numerical_jacobian(
                marginal_distortion, np.log(self.grid))
        np.testing.assert_almost_equal(value, expected_value)

    def test_init_grid_from_most_probable(self):
        init_grid = self.quant._init_grid_from_most_probable(
                self.prev_grid, self.prev_proba)
        self.assertTrue(init_grid.shape==(self.nquant,))
        np.testing.assert_array_equal(init_grid>0, True)
        np.testing.assert_array_equal(np.diff(init_grid)>0, True)
        self.assertTrue(init_grid[0]<self.prev_grid[self.most_proba])
        self.assertTrue(init_grid[-1]>self.prev_grid[self.most_proba])

    def test_one_step_quantize_sorted(self):
        (_, new_grid) = self.quant._one_step_quantize(
                self.prev_grid, self.prev_proba)
        np.testing.assert_almost_equal(new_grid, np.sort(new_grid))

    def test_get_init_innov_is_sorted(self):
        init = self.quant._get_init_innov(self.nquant)
        np.testing.assert_array_equal(np.diff(init)>0, True)
        # More concentrated in center
        diff_center = init[0,2]-init[0,1]
        diff_first = init[0,1]-init[0,0]
        diff_last = init[0,3]-init[0,2]
        self.assertTrue(diff_center<diff_first)
        self.assertTrue(diff_center<diff_last)

    def test_trans_proba_size(self):
        trans = self.quant._transition_probability(self.prev_grid, self.grid)
        self.assertEqual(trans.shape, (self.nquant, self.nquant))

    def test_trans_proba_sum_to_one_and_non_negative(self):
        trans = self.quant._transition_probability(self.prev_grid, self.grid)
        np.testing.assert_array_equal(trans>=0, True)
        np.testing.assert_almost_equal(np.sum(trans,axis=1),1)

    def test_trans_proba(self):
        value = self.quant._transition_probability(
                self.prev_grid,
                self.grid)
        expected_value = np.empty_like(value)
        normpdf = lambda z,h,g: norm.pdf(z)
        for (ev,pg) in zip(expected_value, self.prev_grid):
            ev[:] = self.quantized_integral(normpdf, pg, self.grid)
        np.testing.assert_almost_equal(value, expected_value)

    def test_transition_probability_from_first_grid(self):
        first_variance = 1e-4
        grid, *_ = self.quant._initialize(first_variance)
        trans = self.quant._transition_probability(grid[0],self.grid)
        first_row = trans[0]
        np.testing.assert_array_equal(trans==first_row, True)

    def test_quantize(self):
        quant = pymaat.models.Quantizer(self.model)
        quant.quantize((0.18**2)/252)
