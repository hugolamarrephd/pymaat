import numpy as np
import numpy.testing as npt
import scipy.integrate as integrate

import pymaat.findiff

# Light wrapper around numpy testing
def assert_equal(a, b, msg=''):
    __tracebackhide__ = True # Hide traceback for py.test
    npt.assert_equal(a, b, err_msg=msg)

def assert_almost_equal(a, b, *, rtol=1e-6, atol=0., msg=''):
    __tracebackhide__ = True # Hide traceback for py.test
    npt.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)

def assert_true(a, msg=''):
    __tracebackhide__ = True # Hide traceback for py.test
    npt.assert_equal(a, True, err_msg=msg)

def assert_false(a, msg=''):
    __tracebackhide__ = True # Hide traceback for py.test
    npt.assert_equal(a, False, err_msg=msg)

# Derivative test utilities
def assert_derivative_at(derivative, function, at, *,
        rtol=1e-4, atol=0, mode='central'):
    __tracebackhide__ = True # Hide traceback for py.test
    if callable(derivative):
        value = derivative(at)
    else:
        value = derivative
    expected_value = pymaat.findiff.derivative_at(function, at, mode=mode)
    npt.assert_allclose(value, expected_value,
            rtol=rtol, atol=atol, err_msg='Incorrect derivative')

def assert_gradient_at(gradient, function, at, *,
        rtol=1e-4, atol=0, mode='central'):
    __tracebackhide__ = True # Hide traceback for py.test
    if callable(gradient):
        value = gradient(at)
    else:
        value = gradient
    expected_value = pymaat.findiff.gradient_at(function, at, mode=mode)
    npt.assert_allclose(value, expected_value,
            rtol=rtol, atol=atol, err_msg=f'Incorrect gradient at {at}')

def assert_jacobian_at(jacobian, function, at, *,
        rtol=1e-4, atol=0, mode='central'):
    __tracebackhide__ = True # Hide traceback for py.test
    if callable(jacobian):
        value = jacobian(at)
    else:
        value = jacobian
    expected_value = pymaat.findiff.jacobian_at(function, at, mode=mode)
    npt.assert_allclose(value, expected_value,
            rtol=rtol, atol=atol, err_msg=f'Incorrect jacobian at {at}')

def assert_hessian_at(hessian, function, at, *,
        rtol=1e-2, atol=0, mode='central'):
    __tracebackhide__ = True # Hide traceback for py.test
    if callable(hessian):
        value = hessian(at)
    else:
        value = hessian
    expected_value = pymaat.findiff.hessian_at(function, at, mode=mode)
    npt.assert_allclose(value, expected_value,
            rtol=rtol, atol=atol, err_msg=f'Incorrect hessian at {at}')

# Integral test utilities
def assert_integral_until(integral, function, until,
        lower_bound = -np.inf, rtol=1e-4, atol=0.):
    __tracebackhide__ = True # Hide traceback for py.test
    until = np.atleast_1d(until)
    value = integral(until)
    expected_value = np.empty_like(until)
    for (i,u) in enumerate(until):
        expected_value[i], *_ = integrate.quad(function, lower_bound, u)
    npt.assert_allclose(value, expected_value, rtol=rtol, atol=atol,
            err_msg='Incorrect integral')
