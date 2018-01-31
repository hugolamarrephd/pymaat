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
    if callable(integral):
        value = integral(until)
    else:
        value = integral
    expected_value = np.empty_like(until)
    for (i,u) in enumerate(until.flat):
        index = np.unravel_index(i, until.shape)
        expected_value[index], *_ = integrate.quad(function, lower_bound, u)
    npt.assert_allclose(value, expected_value, rtol=rtol, atol=atol,
            err_msg='Incorrect integral')

# Basic Element-by-Element test utility
# TODO: generalizes for different input / output type
# now only supports float inputs

def assert_elbyel(function, ninput, noutput):
    __tracebackhide__ = True # Hide traceback for py.test
    with np.errstate(invalid='ignore'):
        # np.float64
        _assert_elbyel_supports_scalar(function, ninput, noutput)
        # np.ndarray
        _assert_elbyel_supports_same_size_array(function, ninput, noutput)
        _assert_elbyel_supports_broadcasting(function, ninput, noutput)
        # np.ma.MaskedArray
        _assert_elbyel_supports_masking(function, ninput, noutput)

def _assert_elbyel_supports_scalar(f, nin, nout):
    in_ = (np.nan,)*nin
    out_ = f(*in_)
    if nout==1:
        out_ = (out_,)
    assert isinstance(out_, tuple)
    assert len(out_) == nout
    for i in range(nout):
        assert out_[i].ndim == 0

def _assert_elbyel_supports_same_size_array(f, nin, nout):
    SHAPE = (1,2,3,1,4,1) # 24 elements
    in_ = (np.full(SHAPE, np.nan),)*nin
    out_ = f(*in_)
    if nout==1:
        out_ = (out_,)
    assert isinstance(out_, tuple)
    assert len(out_) == nout
    for i in range(nout):
        assert out_[i].shape==SHAPE

def _assert_elbyel_supports_broadcasting(f, nin, nout):
    SHAPE = tuple(np.arange(nin)+2) # ~nin! elements
    assert not np.any(SHAPE==1)
    in_ = []
    for (i,s) in enumerate(SHAPE):
        shape = np.ones((nin,), dtype=np.int)
        shape[i] = s
        shape = tuple(shape)
        in_.append(np.full(shape, np.nan))
    out_ = f(*in_)
    if nout==1:
        out_ = (out_,)
    assert isinstance(out_, tuple)
    assert len(out_) == nout
    for i in range(nout):
        assert out_[i].shape==SHAPE

def _assert_elbyel_supports_masking(f, nin, nout):
    SHAPE = (1,2,3,1,4,1) # 24 elements
    in_ = (np.ma.array(np.full(SHAPE, np.nan),
        mask=np.full(SHAPE,True), copy=False),)*nin
    out_ = f(*in_)
    if nout==1:
        out_ = (out_,)
    for i in range(nout):
        assert isinstance(out_[i], np.ma.MaskedArray)
        assert_true(out_[i].mask)
