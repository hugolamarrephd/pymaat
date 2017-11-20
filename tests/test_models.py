import unittest

import numpy as np

import pymaat.models

class TestGarchParam(unittest.TestCase):
    def test_invalid_param_raise_exception(self):
        with self.assertRaises(ValueError):
            pymaat.models.Garch(
                mu=0,
                omega=0,
                alpha=0.5,
                beta=0.5,
                gamma=1)

class TestGarchFilters(unittest.TestCase):
    randn = np.array(
            [[[ 0.31559906, -1.80551443, -0.49468605, -1.04842416],
            [ 1.47092134, -0.18375408,  0.61439572,  0.64938352]],

           [[-1.62316804, -0.76478879, -0.75826613,  0.17750069],
            [ 0.82250737,  0.10701289, -1.15691368,  1.05573836]],

           [[-0.22774868,  0.74240245,  1.11521215,  1.10464703],
            [-0.56968483,  0.54138324, -0.69508153, -0.14254879]]])

    randg = np.array(
            [[[ 3.40880883,  1.50217308,  1.11125856,  0.14462167],
            [ 0.08321133,  0.36507093,  1.52133949,  1.42666615]],

           [[ 0.78940018,  0.96257061,  0.3341445 ,  0.59402684],
            [ 1.67863795,  1.21765875,  1.50332767,  1.03224149]],

           [[ 0.78179542,  1.32126548,  1.64830936,  1.59253612],
            [ 0.21397881,  0.01596482,  2.48993567,  1.3034988 ]]])

    shape = randg.shape

    model = pymaat.models.Garch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)

    def test_simulate_revertsto_filter(self):
        innov_ts = self.randn
        first_var = 1e-7
        (sim_var_ts, sim_ret_ts) = self.model.simulate(innov_ts, first_var)
        (fil_var_ts, fil_innov_ts) = self.model.filter(sim_ret_ts, first_var)
        np.testing.assert_allclose(sim_var_ts, fil_var_ts)
        np.testing.assert_allclose(innov_ts, fil_innov_ts)

    def test_one_step_simulate_revertsto_filter(self):
        innov = self.randn
        var = self.randg
        sim_next_var, sim_returns = self.model.one_step_simulate(innov,var)
        fil_next_var, fil_innov = self.model.one_step_filter(sim_returns,var)
        np.testing.assert_allclose(sim_next_var, fil_next_var)
        np.testing.assert_allclose(innov, fil_innov)

    def test_filter_len2_tuple(self):
        out = self.model.filter(self.randn, 1e-7)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_filter_size(self):
        var, innov = self.model.filter(self.randn, 1e-7)
        expected_var_shape = (self.shape[0]+1,) + self.shape[1:]
        expected_innov_shape = self.shape
        self.assertEqual(var.shape, expected_var_shape)
        self.assertEqual(innov.shape, expected_innov_shape)

    def test_filter_positive_variance(self):
        (h,*_) = self.model.filter(self.randn, 1e-7)
        np.testing.assert_array_equal(h>0, True)

    def test_filter_throws_exception_when_non_positive_variance(self):
        with self.assertRaises(ValueError):
            self.model.filter(self.randn, -1)
        with self.assertRaises(ValueError):
            self.model.filter(self.randn, 0)

    def test_simulate_len2_tuple(self):
        out = self.model.simulate(self.randn, 1e-7)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_simulate_size(self):
        var, ret = self.model.simulate(self.randn, 1e-7)
        expected_var_shape = (self.shape[0]+1,) + self.shape[1:]
        expected_ret_shape = self.shape
        self.assertEqual(var.shape, expected_var_shape)
        self.assertEqual(ret.shape, expected_ret_shape)

    def test_simulate_positive_variance(self):
        (h,*_) = self.model.simulate(self.randn, 1e-7)
        np.testing.assert_array_equal(h>0, True)

    def test_simulate_throws_exception_when_non_positive_variance(self):
        with self.assertRaises(ValueError):
            self.model.simulate(self.randn, -1)
        with self.assertRaises(ValueError):
            self.model.simulate(self.randn, 0)

    def test_one_step_filter_len2_tuple(self):
        out = self.model.one_step_filter(self.randn, self.randg)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_one_step_filter_consistent_output(self):
        out = self.model.one_step_filter(self.randn, self.randg)
        self.assertEqual(out[0].size, self.randn.size)
        self.assertEqual(out[1].size, self.randn.size)
        out = self.model.one_step_filter(self.randn, 1)
        self.assertEqual(out[0].size, self.randn.size)
        self.assertEqual(out[1].size, self.randn.size)
        out = self.model.one_step_filter(0, 1)
        self.assertIsInstance(out[0], float)
        self.assertIsInstance(out[1], float)

    def test_one_step_filter_positive_variance(self):
        next_var, *_ = self.model.one_step_filter(self.randn, self.randg)
        np.testing.assert_array_equal(next_var>0, True)

    def test_one_step_filter_at_few_values(self):
        next_var, innov = self.model.one_step_filter(0, 1)
        self.assertAlmostEqual(innov, 0.5-self.model.mu)
        self.assertAlmostEqual(next_var,
                  self.model.omega
                + self.model.beta
                + self.model.alpha * (innov - self.model.gamma) ** 2)

    def test_one_step_simulate_len2_tuple(self):
        out = self.model.one_step_simulate(self.randn, self.randg)
        self.assertIsInstance(out,tuple)
        self.assertEqual(len(out),2)

    def test_one_step_simulate_consistent_output(self):
        out = self.model.one_step_simulate(self.randn, self.randg)
        self.assertEqual(out[0].shape, self.shape)
        self.assertEqual(out[1].shape, self.shape)
        out = self.model.one_step_simulate(self.randn, 1)
        self.assertEqual(out[0].shape, self.shape)
        self.assertEqual(out[1].shape, self.shape)
        out = self.model.one_step_simulate(0, 1)
        self.assertIsInstance(out[0], float)
        self.assertIsInstance(out[1], float)

    def test_one_step_simulate_positive_variance(self):
        next_var, *_ = self.model.one_step_simulate(
                self.randn, self.randg)
        np.testing.assert_array_equal(next_var>0, True)

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


    def test_neg_log_like_at_few_values(self):
        nll = self.model.negative_log_likelihood(1, 1)
        self.assertAlmostEqual(nll, 0.5)
        nll = self.model.negative_log_likelihood(0, np.exp(1))
        self.assertAlmostEqual(nll, 0.5)

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


    def setUp(self):
        self.quant = pymaat.models.Quantizer(
                self.model,
                self.init_innov,
                1e-7)
