import unittest

import numpy as np

import pymaat.testing


class TestNumericalDerivatives(unittest.TestCase):

    test_scalar_arg = np.array([1.02, -2.52, 4.78 ])
    test_multi_arg = np.array([4.23, 9.04, 8.34, 1.02])

    def scalar_to_scalar_func(self, x):
       return np.power(x, 2.)

    def scalar_to_scalar_der(self, x):
       return 2.*x

    def multi_to_scalar_func(self, x):
        return x[0]**2. + x[1]**3. + x[2]**4. + x[3]**5.

    def multi_to_scalar_grad(self, x):
        return np.array([2.*x[0], 3.*x[1]**2., 4.*x[2]**3., 5.*x[3]**4.])

    def test_derivative(self):
        der = pymaat.testing.derivative_at(
                self.scalar_to_scalar_func, self.test_scalar_arg)
        expected_der = self.scalar_to_scalar_der(self.test_scalar_arg)
        np.testing.assert_almost_equal(der, expected_der)

    def test_gradient(self):
        grad = pymaat.testing.gradient_at(
                self.multi_to_scalar_func, self.test_multi_arg)
        expected_grad = self.multi_to_scalar_grad(self.test_multi_arg)
        np.testing.assert_almost_equal(grad, expected_grad)

    def test_assert_good_derivative(self):
        # Does not raise AssertionError
        pymaat.testing.assert_is_derivative_at(
                self.scalar_to_scalar_der,
                self.scalar_to_scalar_func,
                self.test_scalar_arg)

    def test_assert_bad_derivative(self):
        with self.assertRaises(AssertionError):
            pymaat.testing.assert_is_derivative_at(
                    lambda x: x+2,
                    self.scalar_to_scalar_func,
                    self.test_scalar_arg)

    def test_assert_good_gradient(self):
        # Does not raise AssertionError
        pymaat.testing.assert_is_gradient_at(
                self.multi_to_scalar_grad,
                self.multi_to_scalar_func,
                self.test_multi_arg)

    def test_assert_bad_gradient(self):
        bad_grad = lambda x: np.array([2., x[3], x[2]**2., x[3]+4.])
        with self.assertRaises(AssertionError):
            pymaat.testing.assert_is_gradient_at(
                    bad_grad,
                    self.multi_to_scalar_func,
                    self.test_multi_arg)
