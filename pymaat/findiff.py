from functools import partial

import numpy as np
import scipy.misc

SPACE = np.power(np.finfo(float).eps, 1/3)
'''
Optimal delta(x) spacing for central difference method
'''

def derivative_at(func, x, *, mode='central'):
    x = np.atleast_1d(x)
    # dx = np.nanmin(np.maximum(np.abs(x),1.)*SPACE)
    dx = np.nanmin(np.abs(x)*SPACE)
    if mode=='central':
        return scipy.misc.derivative(func, x, dx=dx, order=3)
    elif mode=='forward':
        return (func(x+dx) - func(x))/dx
    elif mode=='backward':
        return (func(x) - func(x-dx))/dx


def gradient_at(func, x, *, mode='central'):
    assert np.atleast_1d(func(x)).size==1
    grad = np.empty_like(x)
    for (i,x_) in enumerate(x):
        def func_1d_to_1d(x_scalar):
            xcopy = x.copy()
            xcopy[i] = x_scalar
            return func(xcopy)
        grad[i] = derivative_at(func_1d_to_1d, x_, mode=mode)
    return grad

def jacobian_at(func, x, *, mode='central'):
    m = np.atleast_1d(func(x)).size
    x = np.atleast_1d(x)
    n = x.size
    jac = np.empty((m,n),float)
    for i in range(m):
        def func_nd_to_1d(x):
            f_ = func(x)
            return f_[i]
        jac[i] = gradient_at(func_nd_to_1d, x, mode=mode)
    return jac

def hessian_at(func, x, *, mode='central'):
    n = np.atleast_1d(x).size
    hess = np.empty((n,n), float)
    for (i,x_) in enumerate(x):
        def grad_i(x_scalar):
            xcopy = x.copy()
            xcopy[i] = x_scalar
            return gradient_at(func, xcopy, mode=mode)
        hess[i] = derivative_at(grad_i, x_, mode=mode)
    return hess

