import unittest

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

    def test_filter_initialize_variance(self):
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
        np.testing.assert_allclose(left_roots,right_roots)

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

    init_innov = np.array(
       [[ 0.63466271,  1.31646684,  1.57076979, -1.48338134, -0.67547033,
        -0.94501199, -1.7518655 ,  1.24787024, -0.24152209, -0.83622565],
       [-0.0116054 , -0.12164961,  0.35902844,  1.98817555, -0.93355169,
         0.53536977, -0.93841577, -2.26992583,  1.30517792,  0.1969917 ],
       [-0.69899308,  0.05944492,  0.53521406, -0.01461253,  2.15017583,
         0.68809939,  0.17947716, -1.0501256 ,  0.26893459, -0.02800586],
       [ 0.7633553 ,  1.14672781,  0.06308125,  2.84545437,  0.17366648,
         0.34276091, -0.48812534, -1.30142566,  1.3140757 ,  1.67758884],
       [-1.05173776, -0.216082  ,  0.02929778,  0.26620765,  1.88502435,
        -0.6491563 ,  0.29142118,  0.35421171, -0.49258563,  1.45412617]])

    n_per = init_innov.shape[0]+1
    n_quant = init_innov.shape[1]

    def setUp(self):
        self.quant = pymaat.models.Quantizer(
                self.model,
                self.init_innov,
                1e-4)

    def test_init_has_consistent_size(self):
        self.assertEqual(self.quant.n_per, self.n_per)
        self.assertEqual(self.quant.n_quant, self.n_quant)
        self.assertEqual(self.quant.gamma.shape,
                (self.n_per, self.n_quant))
        self.assertEqual(self.quant.proba.shape,
                (self.n_per, self.n_quant))
        self.assertEqual(self.quant.trans.shape,
                (self.n_per-1, self.n_quant, self.n_quant))

    def test_init_first_step(self):
        np.testing.assert_array_equal(
                np.diff(self.quant.gamma[0,1:])==0, True)

    def test_init_is_sorted_other_steps(self):
        np.testing.assert_array_equal(
                np.diff(self.quant.gamma[1:])>0, True)

    def test_get_voronoi_1d(self):
        v = self.quant._get_voronoi(np.array([1,2,3,4,5]))
        np.testing.assert_almost_equal(v,
                np.array([0,1.5,2.5,3.5,4.5,np.inf]))

    def test_get_voronoi_2d(self):
        v = self.quant._get_voronoi(np.array(
                [[1,2,3,4,5],
                [6,7,8,9,10]]))
        np.testing.assert_almost_equal(v, np.array(
                [[0,1.5,2.5,3.5,4.5,np.inf],
                [0,6.5,7.5,8.5,9.5,np.inf]]))
    # def test_quantize(self):
    #     self.quant.quantize()
