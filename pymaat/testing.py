import numpy as np
import scipy.misc

EPSILON = np.power(np.finfo(float).eps, 1/3)

def derivative_at(func, x):
    x = np.atleast_1d(x)
    dx = np.maximum(np.abs(x),1.)*EPSILON
    dx = dx.mean() #Handles multiple value
    return scipy.misc.derivative(func, x, dx=dx, order=5)

def gradient_at(func, x):
    assert np.atleast_1d(func(x)).size==1
    grad = np.empty_like(x)
    for (i,x_) in enumerate(x):
        def func_1d(x_scalar):
            xcopy = x.copy()
            xcopy[i] = x_scalar
            return func(xcopy)
        grad[i] = derivative_at(func_1d, x_)
    return grad

def jacobian_at():
    pass

def assert_is_derivative_at(derivative, func, at):
    value = derivative(at)
    expected_value = derivative_at(func, at)
    np.testing.assert_allclose(value, expected_value, rtol=EPSILON)

def assert_is_gradient_at(gradient, func, at):
    value = gradient(at)
    expected_value = gradient_at(func, at)
    np.testing.assert_allclose(value, expected_value, rtol=EPSILON)

def assert_is_jacobian(func, jacobian):
    pass
