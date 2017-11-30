import numpy as np
import scipy.misc

FD_SPACE = np.power(np.finfo(float).eps, 1/3)
'''
Optimal delta(x) spacing for central difference method
'''

def derivative_at(func, x):
    x = np.atleast_1d(x)
    dx = np.maximum(np.abs(x),1.)*FD_SPACE
    dx = dx.mean() #Handles multiple value
    return scipy.misc.derivative(func, x, dx=dx, order=5)

def gradient_at(func, x):
    assert np.atleast_1d(func(x)).size==1
    grad = np.empty_like(x)
    for (i,x_) in enumerate(x):
        def func_1d_to_1d(x_scalar):
            xcopy = x.copy()
            xcopy[i] = x_scalar
            return func(xcopy)
        grad[i] = derivative_at(func_1d_to_1d, x_)
    return grad

def jacobian_at(func, x):
    m = np.atleast_1d(func(x)).size
    x = np.atleast_1d(x)
    n = x.size
    jac = np.empty((m,n),float)
    for i in range(m):
        def func_nd_to_1d(x):
            f_ = func(x)
            return f_[i]
        jac[i] = gradient_at(func_nd_to_1d, x)
    return jac


