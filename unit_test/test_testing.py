import numpy as np
import pytest

from pymaat.testing import *

class TestBasicAssertions:

    # Assert equal
    def test_assert_equal_when_equal_scalars(self):
        assert_equal(1., 1.)

    def test_assert_equal_when_equal_arrays(self):
        an_array = np.zeros((1,2))
        another_array = np.array([[0.,0.]])
        assert_equal(an_array, another_array)

    def test_assert_equal_when_equal_mixed(self):
        an_array = np.zeros((1,2))
        assert_equal(an_array, 0.)
        assert_equal(0, an_array)

    def test_assert_equal_when_not_equal_scalars(self):
        with pytest.raises(AssertionError):
            assert_equal(1.,2.)

    def test_assert_equal_when_not_equal_arrays(self):
        an_array = np.zeros((1,2))
        another_array = np.array([[0.,1.]])
        with pytest.raises(AssertionError):
            assert_equal(an_array, another_array)

    def test_assert_equal_when_not_equal_mixed(self):
        an_array = np.zeros((1,2))
        an_array[0,1] = 1.
        with pytest.raises(AssertionError):
            assert_equal(an_array, 0.)

    def test_assert_equal_when_size_mismatch(self):
        an_array = np.zeros((1,2))
        another_array = np.zeros((2))
        with pytest.raises(AssertionError):
            assert_equal(an_array, another_array)

    # Assert almost equal
    def test_assert_almost_equal_when_equal_scalars(self):
        assert_almost_equal(1.0000000001, 1.)

    def test_assert_almost_equal_when_equal_arrays(self):
        an_array = np.ones((1,2))
        another_array = np.array([[1.,1.]]) + 0.000000000001
        assert_almost_equal(an_array, another_array)

    def test_assert_almost_equal_when_equal_mixed(self):
        an_array = np.ones((1,2))
        assert_almost_equal(an_array, 1.000000000001)
        assert_almost_equal(1.000000000001, an_array)

    def test_assert_almost_equal_when_not_equal_scalars(self):
        with pytest.raises(AssertionError):
            assert_almost_equal(1.,2.)

    def test_assert_almost_equal_when_not_equal_arrays(self):
        an_array = np.zeros((1,2))
        another_array = np.array([[0.,1.]])
        with pytest.raises(AssertionError):
            assert_almost_equal(an_array, another_array)

    def test_assert_almost_equal_when_not_equal_mixed(self):
        an_array = np.zeros((1,2))
        an_array[0,1] = 1.
        with pytest.raises(AssertionError):
            assert_almost_equal(an_array, 0.)
            assert_almost_equal(0., an_array)

    def test_assert_almost_equal_when_size_mismatch(self):
        an_array = np.zeros((1,2))
        another_array = np.zeros((2))
        with pytest.raises(AssertionError):
            assert_almost_equal(an_array, another_array)

    # Assert True
    def test_assert_true_when_true_scalars(self):
        assert_true(True)

    def test_assert_true_when_true_arrays(self):
        an_array = np.full((1,2), True, dtype=bool)
        assert_true(an_array)

    def test_assert_true_when_not_true_scalars(self):
        with pytest.raises(AssertionError):
            assert_true(False)

    def test_assert_true_when_not_true_arrays(self):
        an_array = np.array([False, True])
        with pytest.raises(AssertionError):
            assert_true(an_array)

    # Assert False
    def test_assert_false_when_false_scalars(self):
        assert_false(False)

    def test_assert_false_when_false_arrays(self):
        an_array = np.full((1,2), False, dtype=bool)
        assert_false(an_array)

    def test_assert_false_when_not_false_scalars(self):
        with pytest.raises(AssertionError):
            assert_false(True)

    def test_assert_false_when_not_false_arrays(self):
        an_array = np.array([False, True])
        with pytest.raises(AssertionError):
            assert_false(an_array)

class TestFiniteDifferenceAssertions:

    # Derivative assertion
    def test_assert_good_callable_derivative(self):
        # Does not raise AssertionError
        assert_derivative_at(lambda x: 2.*x, lambda x: x**2., 1.02)

    def test_assert_good_derivative_value(self):
        # Does not raise AssertionError
        assert_derivative_at(2.04, lambda x: x**2., 1.02)

    def test_assert_bad_callable_derivative(self):
        with pytest.raises(AssertionError):
            assert_derivative_at(lambda x: x+2., lambda x: x**2., 1.02)

    def test_assert_bad_derivative_value(self):
        with pytest.raises(AssertionError):
            assert_derivative_at(32.12, lambda x: x**2., 1.02)

    # Gradient assertion
    def test_assert_good_callable_gradient(self):
        # Does not raise AssertionError
        assert_gradient_at(lambda x: 2.*x, lambda x: x**2., 1.02)

    def test_assert_good_gradient_value(self):
        # Does not raise AssertionError
        assert_gradient_at(2.04, lambda x: x**2., 1.02)

    def test_assert_bad_callable_gradient(self):
        with pytest.raises(AssertionError):
            assert_gradient_at(lambda x: x+2., lambda x: x**2., 1.02)

    def test_assert_bad_gradient_value(self):
        with pytest.raises(AssertionError):
            assert_gradient_at(32.12, lambda x: x**2., 1.02)

    # Jacobian assertion
    def test_assert_good_callable_jacobian(self):
        # Does not raise AssertionError
        assert_jacobian_at(lambda x: 2.*x, lambda x: x**2., 1.02)

    def test_assert_good_jacobian_value(self):
        # Does not raise AssertionError
        assert_jacobian_at(2.04, lambda x: x**2., 1.02)

    def test_assert_bad_callable_jacobian(self):
        with pytest.raises(AssertionError):
            assert_jacobian_at(lambda x: x+2., lambda x: x**2., 1.02)

    def test_assert_bad_jacobian_value(self):
        with pytest.raises(AssertionError):
            assert_jacobian_at(32.12, lambda x: x**2., 1.02)

    # Hessian assertion
    def test_assert_good_callable_hessian(self):
        # Does not raise AssertionError
        assert_hessian_at(lambda x: 2., lambda x: x**2., 1.02)

    def test_assert_good_hessian_value(self):
        # Does not raise AssertionError
        assert_hessian_at(2., lambda x: x**2., 1.02)

    def test_assert_bad_callable_hessian(self):
        with pytest.raises(AssertionError):
            assert_hessian_at(lambda x: x+2., lambda x: x**2., 1.02)

    def test_assert_bad_hessian_value(self):
        with pytest.raises(AssertionError):
            assert_hessian_at(32.12, lambda x: x**2., 1.02)

class TestIntegralAssertions:

    def test_assert_integral_until_good_scalar(self):
        # Does not raise AssertionError
        assert_integral_until(lambda x: np.exp(2.*x)/2.,
            lambda x: np.exp(2.*x), 0.)

    def test_assert_integral_until_good_array(self):
        # Does not raise AssertionError
        assert_integral_until(lambda x: np.exp(2.*x)/2.,
            lambda x: np.exp(2.*x), np.array([-1.,0.,1.]))

    def test_assert_integral_until_bad_scalar(self):
        with pytest.raises(AssertionError):
            assert_integral_until(lambda x: np.exp(2.*x),
                lambda x: np.exp(2.*x), 0.)

    def test_assert_integral_until_bad_array(self):
        with pytest.raises(AssertionError):
            assert_integral_until(lambda x: np.exp(2.*x),
                lambda x: np.exp(2.*x), np.array([-1.,0.,1.]))
