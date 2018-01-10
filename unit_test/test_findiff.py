import numpy as np

import pymaat.testing
import pymaat.findiff

class TestNumericalDerivatives(pymaat.testing.TestCase):

    test_scalar_arg = np.array([1.02, -2.52, 4.78 ])
    test_multi_arg = np.array([4.23, 9.04, 8.34, 1.02])

    def scalar_to_scalar_func(self, x):
       return np.power(x, 2.)

    def scalar_to_scalar_der(self, x):
       return 2.*x

    def multi_to_scalar_func(self, x):
        return x[0]**2.*x[3] + x[2]**3.*x[1]**3. + x[2]**4. + x[3]**5.

    def multi_to_scalar_grad(self, x):
        return np.array([2.*x[0]*x[3],
            x[2]**3.*3.*x[1]**2.,
           3.*x[2]**2.*x[1]**3.+4.*x[2]**3.,
           x[0]**2. + 5.*x[3]**4.])

    def multi_to_scalar_hess(self, x):
        return np.array([
            [2.*x[3], 0, 0, 2.*x[0]],
            [0, x[2]**3.*6.*x[1], 3.*x[2]**2.*3.*x[1]**2., 0],
            [0, 3.*x[2]**2.*3.*x[1]**2., 6.*x[2]*x[1]**3.+12.*x[2]**2., 0],
            [2.*x[0], 0, 0, 20.*x[3]**3.]
            ])

    def multi_to_multi_func(self, x):
        return np.array([x[0]**2. + x[1]**3. + x[2]**4. + x[3]**5.,
            6.*x[2] + x[3]**3.,
            x[0]*x[2]**2 + x[1]*x[3]**0.5,
            x[0]*x[1]*x[2]*x[3]])

    def multi_to_multi_jac(self, x):
        return np.array([
            [2.*x[0], 3.*x[1]**2., 4.*x[2]**3., 5.*x[3]**4],
            [0., 0., 6., 3.*x[3]**2.],
            [x[2]**2., x[3]**0.5, 2.*x[0]*x[2], x[1]/(2.*x[3]**0.5)],
            [x[1]*x[2]*x[3], x[0]*x[2]*x[3], x[0]*x[1]*x[3], x[0]*x[1]*x[2]]
            ])

    def test_derivative(self):
        der = pymaat.findiff.derivative_at(
                self.scalar_to_scalar_func, self.test_scalar_arg)
        expected_der = self.scalar_to_scalar_der(self.test_scalar_arg)
        self.assert_almost_equal(der, expected_der, rtol=1e-6, atol=1e-6)

    def test_derivative_at_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            der = pymaat.findiff.derivative_at(
                    self.scalar_to_scalar_func, 0)

    def test_gradient(self):
        grad = pymaat.findiff.gradient_at(
                self.multi_to_scalar_func, self.test_multi_arg)
        expected_grad = self.multi_to_scalar_grad(self.test_multi_arg)
        self.assert_almost_equal(grad, expected_grad, rtol=1e-6, atol=1e-6)

    def test_jacobian(self):
        jac = pymaat.findiff.jacobian_at(
                self.multi_to_multi_func, self.test_multi_arg)
        expected_jac = self.multi_to_multi_jac(self.test_multi_arg)
        np.testing.assert_allclose(jac, expected_jac, rtol=1e-6, atol=1e-6)

    def test_hessian(self):
        hess = pymaat.findiff.hessian_at(
                self.multi_to_scalar_func, self.test_multi_arg)
        expected_hess = self.multi_to_scalar_hess(self.test_multi_arg)
        np.testing.assert_allclose(hess, expected_hess, rtol=1e-1, atol=1e-6)
