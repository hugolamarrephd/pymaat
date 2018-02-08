import operator

import numpy as np
from numpy.testing.utils import assert_array_compare
from numpy.core.numeric import isclose
import scipy.integrate as integrate

import pymaat.findiff


###########
# Boolean #
###########

def assert_all(x, *, msg=''):
    __tracebackhide__ = True # Hide traceback for py.test
    assert np.all(x), msg

def assert_any(x, *, msg=''):
    __tracebackhide__ = True # Hide traceback for py.test
    assert np.any(x), msg

###########
# Numeric #
###########

def assert_valid(x):
    __tracebackhide__ = True # Hide traceback for py.test
    header = 'Invalid value(s) detected [True when NaN]'
    assert_array_compare(operator.__eq__, np.isnan(x), False,
            header=header)

def assert_invalid(x):
    __tracebackhide__ = True # Hide traceback for py.test
    header = 'Valid value(s) detected [True when non-NaN]'
    assert_array_compare(operator.__eq__, ~np.isnan(x), False,
            header=header)

def assert_finite(x, *, invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    header = 'Infinite value(s) detected [True when +-inf]'
    compare_args, x = _warmup_compare(invalid, header, x)
    assert_array_compare(operator.__eq__, ~np.isfinite(x), False,
            **compare_args)

def assert_equal(x, y, *, invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    header = "Arrays are not equal (==)"
    compare_args, x, y = _warmup_compare(invalid, header, x, y)
    assert_array_compare(operator.__eq__, x, y, **compare_args)

def assert_not_equal(x, y, *, invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    header = "Arrays are equal (!=)"
    compare_args, x, y = _warmup_compare(invalid, header, x, y)
    assert_array_compare(operator.__ne__, x, y, **compare_args)

def assert_almost_equal(x, y, *, rtol=1e-6, atol=0., invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    if rtol<0 or atol<0:
        raise ValueError('Tolerance must be positive')
    header = ('Not almost equal (~=) to tolerance rtol={}, atol={}.'
            .format(rtol, atol))
    compare_args, x, y = _warmup_compare(invalid, header, x, y)
    def compare(x, y):
        return isclose(x, y, rtol=rtol, atol=atol,
                equal_nan=compare_args['equal_nan'])
    assert_array_compare(compare, x, y, **compare_args)

def assert_less_equal(x, y, *, invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    header = 'Arrays are not less or equal (<=)-ordered.'
    compare_args, x, y = _warmup_compare(invalid, header, x, y)
    assert_array_compare(operator.__le__, x, y, **compare_args)

def assert_less(x, y, *, invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    header = 'Arrays are not less (<)-ordered.'
    compare_args, x, y = _warmup_compare(invalid, header, x, y)
    assert_array_compare(operator.__lt__, x, y, **compare_args)

def assert_greater_equal(x, y, *, invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    header = 'Arrays are not greater or equal (>=)-ordered'
    compare_args, x, y = _warmup_compare(invalid, header, x, y)
    assert_array_compare(operator.__ge__, x, y, **compare_args)

def assert_greater(x, y, *, invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    header = 'Arrays are not greater (>)-ordered.'
    compare_args, x, y = _warmup_compare(invalid, header, x, y)
    assert_array_compare(operator.__gt__, x, y, **compare_args)

_warmup_compare_flag_err = ValueError("Unexpected 'invalid' flag:"
        "must be either 'fail', 'allow', or boolean array")

def _warmup_compare(flag, header, *args):
    __tracebackhide__ = True # Hide traceback for py.test
    if isinstance(flag, str):
        if flag.lower()=='fail':
            for a in args: # Preemptively fails here if any NaNs
                assert_valid(a)
        elif flag.lower()=='allow':
            pass # Do nothing
        else:
            raise _warmup_compare_flag_err
        flag_str = flag.lower()
        out = args
    elif isinstance(flag, np.ndarray) and flag.dtype==np.bool:
        # Flag is an array of boolean indexing invalid values
        #    that are to be ignored by the main assertion
        flag_str = 'ignore'
        out = []
        for a in args:
            # Mark ignored values as NaNs
            a = a.copy()
            a[flag] = np.nan
            out.append(a)
        out = tuple(out)
    else:
        raise _warmup_compare_flag_err
    header = header + " [invalid='{}']".format(flag_str)
    return ({'header':header,
        'equal_nan':True,
        'equal_inf':False},) + out

##############
# Derivative #
##############

def assert_derivative_at(derivative, function, at, *,
        rtol=1e-4, atol=0, mode='central', invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    if callable(derivative):
        value = derivative(at)
    else:
        value = derivative
    expected_value = pymaat.findiff.derivative_at(function, at, mode=mode)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
            invalid=invalid)

def assert_gradient_at(gradient, function, at, *,
        rtol=1e-4, atol=0, mode='central', invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    if callable(gradient):
        value = gradient(at)
    else:
        value = gradient
    expected_value = pymaat.findiff.gradient_at(function, at, mode=mode)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
            invalid=invalid)

def assert_jacobian_at(jacobian, function, at, *,
        rtol=1e-4, atol=0, mode='central', invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    if callable(jacobian):
        value = jacobian(at)
    else:
        value = jacobian
    expected_value = pymaat.findiff.jacobian_at(function, at, mode=mode)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
            invalid=invalid)

def assert_hessian_at(hessian, function, at, *,
        rtol=1e-2, atol=0, mode='central', invalid='fail'):
    __tracebackhide__ = True # Hide traceback for py.test
    if callable(hessian):
        value = hessian(at)
    else:
        value = hessian
    expected_value = pymaat.findiff.hessian_at(function, at, mode=mode)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
            invalid=invalid)

###############
# Integration #
###############

def assert_integral_until(integral, function, until,
        lower_bound = -np.inf, rtol=1e-4, atol=0., invalid='fail'):
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
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
            invalid=invalid)

############
# Function #
############

def assert_elbyel(function, ninput, noutput):
    """
    Basic test suite for element-by-element functions
    """
    # TODO: generalizes for different input / output type
    # now only supports float inputs
    __tracebackhide__ = True # Hide traceback for py.test
    with np.errstate(invalid='ignore'):
        _assert_elbyel_supports_scalar(function, ninput, noutput)
        _assert_elbyel_supports_same_size_array(function, ninput, noutput)
        _assert_elbyel_supports_broadcasting(function, ninput, noutput)

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
    # To properly test broadcasting, all dimensions from output shape must
    # **not** be one
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
