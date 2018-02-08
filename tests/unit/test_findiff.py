import pytest
import numpy as np

import pymaat.testing as pt
import pymaat.findiff

# Scalar-to-scalar
def scalar_to_scalar(x):
   return np.power(x, 2.)

def derivative(x):
   return 2.*x

# Multi-to-scalar
def multi_to_scalar(x):
    return x[0]**2.*x[3] + x[2]**3.*x[1]**3. + x[2]**4. + x[3]**5.

def gradient(x):
    return np.array([2.*x[0]*x[3],
        x[2]**3.*3.*x[1]**2.,
       3.*x[2]**2.*x[1]**3.+4.*x[2]**3.,
       x[0]**2. + 5.*x[3]**4.])

def hessian(x):
    return np.array([
        [2.*x[3], 0, 0, 2.*x[0]],
        [0, x[2]**3.*6.*x[1], 3.*x[2]**2.*3.*x[1]**2., 0],
        [0, 3.*x[2]**2.*3.*x[1]**2., 6.*x[2]*x[1]**3.+12.*x[2]**2., 0],
        [2.*x[0], 0, 0, 20.*x[3]**3.]
        ])

# Multi-to-multi
def multi_to_multi(x):
    return np.array([x[0]**2. + x[1]**3. + x[2]**4. + x[3]**5.,
        6.*x[2] + x[3]**3.,
        x[0]*x[2]**2 + x[1]*x[3]**0.5,
        x[0]*x[1]*x[2]*x[3]])

def jacobian(x):
    return np.array([
        [2.*x[0], 3.*x[1]**2., 4.*x[2]**3., 5.*x[3]**4],
        [0., 0., 6., 3.*x[3]**2.],
        [x[2]**2., x[3]**0.5, 2.*x[0]*x[2], x[1]/(2.*x[3]**0.5)],
        [x[1]*x[2]*x[3], x[0]*x[2]*x[3], x[0]*x[1]*x[3], x[0]*x[1]*x[2]]
        ])

scalar_arg = np.array([1.02, -2.52, 4.78 ])
multi_arg = np.array([4.23, 9.04, 8.34, 1.02])

class TestNumericalDerivatives:


    def test_derivative(self):
        der = pymaat.findiff.derivative_at(scalar_to_scalar, scalar_arg)
        expected_der = derivative(scalar_arg)
        pt.assert_almost_equal(der, expected_der, rtol=1e-6, atol=1e-6)

    def test_derivative_at_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            der = pymaat.findiff.derivative_at(
                    scalar_to_scalar, 0)

    def test_gradient(self):
        grad = pymaat.findiff.gradient_at(multi_to_scalar, multi_arg)
        expected_grad = gradient(multi_arg)
        pt.assert_almost_equal(grad, expected_grad, rtol=1e-6, atol=1e-6)


    def test_jacobian(self):
        jac = pymaat.findiff.jacobian_at(multi_to_multi, multi_arg)
        expected_jac = jacobian(multi_arg)
        pt.assert_almost_equal(jac, expected_jac, rtol=1e-6, atol=1e-6)


    def test_hessian(self):
        hess = pymaat.findiff.hessian_at(multi_to_scalar, multi_arg)
        expected_hess = hessian(multi_arg)
        pt.assert_almost_equal(hess, expected_hess, rtol=1e-1, atol=1e-6)
