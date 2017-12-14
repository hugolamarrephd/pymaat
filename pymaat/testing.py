import unittest

import numpy as np
import scipy.integrate as integrate

import pymaat.findiff


class TestCase(unittest.TestCase):
    '''
    rtol and atol are too critical to be set by default.
    '''
    # Light wrapper around np.testing
    def assert_equal(self, a, b, msg=None):
        # Overrides unittest.TestCase to support numpy arrays
        np.testing.assert_equal(a, b, err_msg=msg)

    def assert_true(self, a, msg=None):
        self.assert_equal(a, True, msg=msg)

    def assert_false(self, a, msg=None):
        self.assert_equal(a, False, msg=msg)

    def assert_almost_equal(self, a, b, *, rtol=1e-6, atol=0, msg=None):
        # Overrides unittest.TestCase to support numpy arrays
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)

    # Derivative test utilities
    def assert_derivative_at(self, derivative, func, at, *,
            rtol=1e-6, atol=0, mode='central'):
        value = derivative(at)
        expected_value = pymaat.findiff.derivative_at(func, at, mode=mode)
        self.assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                msg='Incorrect derivative')

    def assert_gradient_at(self, gradient, func, at, *,
            rtol=1e-6, atol=0, mode='central'):
        value = gradient(at)
        expected_value = pymaat.findiff.gradient_at(func, at, mode=mode)
        self.assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                msg=f'Incorrect gradient at {at}')

    def assert_jacobian_at(self, jacobian, func, at, *,
            rtol=1e-6, atol=0, mode='central'):
        value = jacobian(at)
        expected_value = pymaat.findiff.jacobian_at(func, at, mode=mode)
        self.assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                msg=f'Incorrect jacobian at {at}')

    def assert_hessian_at(self, hessian, func, at, *,
            rtol=1e-6, atol=0, mode='central'):
        value = hessian(at)
        expected_value = pymaat.findiff.hessian_at(func, at, mode=mode)
        self.assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                msg=f'Incorrect hessian at {at}')

    # Integral test utilities
    def assert_integral_until(self, integral, func, until,
            lower_bound = -np.inf,
            rtol=1e-6, atol=0):
        value = integral(until)
        expected_value = np.empty_like(until)
        for (i,u) in enumerate(until):
            expected_value[i], *_ = integrate.quad(
                    func, lower_bound, u)
        self.assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                msg='Incorrect integral')

    def assert_is_integral_until(self, integral, lower_bound=-np.inf):
        self.assert_equal(integral(lower_bound), 0,
                msg='Integral until lower bound is non-zero')
