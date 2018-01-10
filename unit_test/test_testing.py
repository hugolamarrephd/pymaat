import numpy as np

import pymaat.testing

class TestBasicAssertions(pymaat.testing.TestCase):
    pass
    # def test_assertEqual_supports_builtin(self):
    #     self.fail()

    # def test_assertEqual_supports_np_array(self):
    #     self.fail()

    # def test_assertAlmostEqual_supports_builtin(self):
    #     self.fail()

    # def test_assertAlmostEqual_supports_np_array(self):
    #     self.fail()

class TestDerivativesAssertions(pymaat.testing.TestCase):

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

    def test_assert_good_derivative(self):
        # Does not raise AssertionError
        self.assert_derivative_at(
                self.scalar_to_scalar_der,
                self.scalar_to_scalar_func,
                self.test_scalar_arg,
                rtol=1e-6, atol=1e-6)

    def test_assert_bad_derivative(self):
        with self.assertRaises(AssertionError):
            self.assert_derivative_at(
                    lambda x: x+2,
                    self.scalar_to_scalar_func,
                    self.test_scalar_arg,
                    rtol=1e-6, atol=1e-6)

    def test_assert_good_gradient(self):
        # Does not raise AssertionError
        self.assert_gradient_at(
                self.multi_to_scalar_grad,
                self.multi_to_scalar_func,
                self.test_multi_arg,
                rtol=1e-6, atol=1e-6)

    def test_assert_bad_gradient(self):
        bad_grad = lambda x: np.array([2., x[3], x[2]**2., x[3]+4.])
        with self.assertRaises(AssertionError):
            self.assert_gradient_at(
                    bad_grad,
                    self.multi_to_scalar_func,
                    self.test_multi_arg,
                    rtol=1e-6, atol=1e-6)

    def test_assert_good_jacobian(self):
        # Does not raise AssertionError
        self.assert_jacobian_at(
                self.multi_to_multi_jac,
                self.multi_to_multi_func,
                self.test_multi_arg,
                rtol=1e-6, atol=1e-6)

    def test_assert_bad_jacobian(self):
        bad_jac = lambda x: np.array([
            [2., x[3], x[2]**2., x[3]+4.],
            [2., x[3], x[2]**2., x[3]+4.],
            [2., x[3], x[2]**2., x[3]+4.],
            [2., x[3], x[2]**2., x[3]+4.],
            ])
        with self.assertRaises(AssertionError):
            self.assert_jacobian_at(
                    bad_jac,
                    self.multi_to_multi_func,
                    self.test_multi_arg,
                    rtol=1e-6, atol=1e-6)
