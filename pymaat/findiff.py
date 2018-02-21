from functools import partial

import numpy as np
import scipy.misc

from pymaat.nputil import atleast_1d, elbyel

SPACE = np.power(np.finfo(float).eps, 1/3)
'''
Optimal delta(x) spacing for central difference method
'''

@atleast_1d
def derivative_at(x, *, function=None, mode='central'):
    _chk_fcn(function)

    def _get_dx(x):
        dx = np.nanmin(np.abs(x)*SPACE)
        if np.all(dx == 0.):
            raise ValueError("At least one x must be non-zero")
        return dx

    dx = _get_dx(x)
    if mode.lower()=='central':
        return (function(x+dx) - function(x-dx))/(2.*dx)
    elif mode.lower()=='forward':
        return (function(x+dx) - function(x))/dx
    elif mode.lower()=='backward':
        return (function(x) - function(x-dx))/dx
    else:
        raise ValueError("Unexpected finite difference mode: "
                "expecting 'central', 'forward', or 'backward'.")


@atleast_1d
def gradient_at(x, *, function=None, mode='central'):
    _chk_fcn(function)
    grad = np.empty(x.shape, dtype=np.float_)
    for (i,_x) in enumerate(x):
        def _1d_to_1d(x_scalar):
            xcopy = x.copy()
            xcopy[i] = x_scalar
            return function(xcopy)
        grad[i] = derivative_at(_x, function=_1d_to_1d,  mode=mode)

    return grad

@atleast_1d
def jacobian_at(x, *, function=None, mode='central'):
    _chk_fcn(function)
    # Determine size of function
    m = np.atleast_1d(function(x)).size
    n = x.size
    jac = np.empty((m,n), dtype=np.float_)
    for i in range(m):
        def _nd_to_1d(x):
            f_ = function(x)
            return f_[i]
        jac[i] = gradient_at(x, function=_nd_to_1d, mode=mode)
    return jac

@atleast_1d
def hessian_at(x, *, function=None, mode='central'):
    _chk_fcn(function)
    n = x.size
    hess = np.empty((n,n), dtype=np.float_)
    for (i,_x) in enumerate(x):
        def _grad_i(x_scalar):
            xcopy = x.copy()
            xcopy[i] = x_scalar
            return gradient_at(xcopy, function=function, mode=mode)
        hess[i] = derivative_at(_x, function=_grad_i,  mode=mode)
    return hess

def _chk_fcn(func):
    if not callable(func):
        raise ValueError('Please provide valid function')

