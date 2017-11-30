import unittest

import numpy as np

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
    def assert_derivative_at(self, derivative, func, at, rtol=1e-6, atol=0):
        value = derivative(at)
        expected_value = pymaat.findiff.derivative_at(func, at)
        self.assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                msg='Incorrect derivative')

    def assert_gradient_at(self, gradient, func, at, rtol=1e-6, atol=0):
        value = gradient(at)
        expected_value = pymaat.findiff.gradient_at(func, at)
        self.assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                msg='Incorrect gradient')

    def assert_jacobian_at(self, jacobian, func, at, rtol=1e-6, atol=0):
        value = jacobian(at)
        expected_value = pymaat.findiff.jacobian_at(func, at)
        self.assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                msg='Incorrect jacobian')
