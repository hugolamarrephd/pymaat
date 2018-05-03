import warnings

import numpy as np
import scipy.integrate as integrate
from tabulate import tabulate

import pymaat.findiff
from pymaat.nputil import atleast_1d

DEBUG = True

###########
# Boolean #
###########


def assert_all(x, *, msg=''):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    assert np.all(x), msg


def assert_any(x, *, msg=''):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    assert np.any(x), msg

###########
# Numeric #
###########


@atleast_1d
def assert_valid(x):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    failures = np.isnan(x)
    if np.any(failures):
        to_cmp = np.full(x.shape, True)
        msg = 'Is valid\n'
        msg += _debug_table(failures, to_cmp, x, headers=('x'))
        raise AssertionError(msg)


@atleast_1d
def assert_invalid(x):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    failures = np.logical_not(np.isnan(x))
    if np.any(failures):
        to_cmp = np.full(x.shape, True)
        msg = 'Is invalid\n'
        msg += _debug_table(failures, to_cmp, x, headers=('x'))
        raise AssertionError(msg)


@atleast_1d
def assert_finite(x, *, invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    msg = 'Is finite\n'
    msg, failures, to_cmp = _treat_invalid(invalid, x, msg=msg)
    failures[to_cmp] = np.logical_not(np.isfinite(x[to_cmp]))
    if np.any(failures):
        msg += _debug_table(failures, to_cmp, x, headers=('x'))
        raise AssertionError(msg)


@atleast_1d
def assert_equal(x, y, *, shape='strict', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    msg = 'Equal\n'
    msg, x, y = _validate_shape(shape, x, y, msg=msg)
    msg, failures, to_cmp = _treat_invalid(invalid, x, y, msg=msg)
    failures[to_cmp] = x[to_cmp] != y[to_cmp]
    if np.any(failures):
        msg += _debug_table(failures, to_cmp, x, y, headers=('x', 'y'))
        raise AssertionError(msg)


@atleast_1d
def assert_not_equal(x, y, *, shape='strict', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    msg = 'Not equal\n'
    msg, x, y = _validate_shape(shape, x, y, msg=msg)
    msg, failures, to_cmp = _treat_invalid(invalid, x, y, msg=msg)
    failures[to_cmp] = x[to_cmp] == y[to_cmp]
    if np.any(failures):
        msg += _debug_table(failures, to_cmp, x, y, headers=('x', 'y'))
        raise AssertionError(msg)


@atleast_1d
def assert_less_equal(x, y, *, shape='strict', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    msg = 'Less or equal-ordered\n'
    msg, x, y = _validate_shape(shape, x, y, msg=msg)
    msg, failures, to_cmp = _treat_invalid(invalid, x, y, msg=msg)
    failures[to_cmp] = x[to_cmp] > y[to_cmp]
    if np.any(failures):
        msg += _debug_table(failures, to_cmp, x, y, headers=('x', 'y'))
        raise AssertionError(msg)


@atleast_1d
def assert_less(x, y, *, shape='strict', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    msg = 'Less-ordered\n'
    msg, x, y = _validate_shape(shape, x, y, msg=msg)
    msg, failures, to_cmp = _treat_invalid(invalid, x, y, msg=msg)
    failures[to_cmp] = x[to_cmp] >= y[to_cmp]
    if np.any(failures):
        msg += _debug_table(failures, to_cmp, x, y, headers=('x', 'y'))
        raise AssertionError(msg)


@atleast_1d
def assert_greater_equal(x, y, *, shape='strict', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    msg = 'Greater or equal-ordered\n'
    msg, x, y = _validate_shape(shape, x, y, msg=msg)
    msg, failures, to_cmp = _treat_invalid(invalid, x, y, msg=msg)
    failures[to_cmp] = x[to_cmp] < y[to_cmp]
    if np.any(failures):
        msg += _debug_table(failures, to_cmp, x, y, headers=('x', 'y'))
        raise AssertionError(msg)


@atleast_1d
def assert_greater(x, y, *, shape='strict', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    msg = 'Greater-ordered\n'
    msg, x, y = _validate_shape(shape, x, y, msg=msg)
    msg, failures, to_cmp = _treat_invalid(invalid, x, y, msg=msg)
    failures[to_cmp] = x[to_cmp] <= y[to_cmp]
    if np.any(failures):
        msg += _debug_table(failures, to_cmp, x, y, headers=('x', 'y'))
        raise AssertionError(msg)


@atleast_1d
def assert_almost_equal(
        x, y, *, rtol=1e-6, atol=0., shape='strict', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    if rtol < 0 or atol < 0:
        raise ValueError('Tolerance must be positive')
    msg = (
        'Almost equal to tolerance [relative={}, absolute={}]\n'
        .format(rtol, atol)
    )
    msg, x, y = _validate_shape(shape, x, y, msg=msg)
    msg, failures, to_cmp = _treat_invalid(invalid, x, y, msg=msg)
    close = np.isclose(
        x[to_cmp], y[to_cmp], rtol=rtol, atol=atol, equal_nan=False)
    failures[to_cmp] = np.logical_not(close)
    if np.any(failures):
        # Special columns
        with np.errstate(invalid='ignore', divide='ignore'):
            absolute_error = np.absolute(x-y)
            relative_error = absolute_error/np.absolute(y)
        # Discard cases when relative error is undetermined
        relative_error[np.logical_not(np.isfinite(relative_error))] = np.nan
        # Build table
        msg += _debug_table(
            failures, to_cmp, x, y,
            relative_error, absolute_error,
            headers=('x', 'y', 'rel', 'abs')
        )
        # Add footers
        with warnings.catch_warnings():
            # Hide mean of empty slice warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_rel = np.nanmean(relative_error[to_cmp])
            mean_abs = np.nanmean(absolute_error[to_cmp])
        msg += 'Mean(rel): ' + str(mean_rel) + '\n'
        msg += 'Mean(abs): ' + str(mean_abs) + '\n'
        raise AssertionError(msg)


def _validate_shape(mode, *args, msg=''):
    if isinstance(mode, tuple):
        shape = mode
        mode = 'strict'
    else:
        shape = args[0].shape

    if isinstance(mode, str) and mode.lower() == 'strict':
        for a in args[1:]:
            if shape != a.shape:
                msg += ('Arrays do not have same shape: '
                        'expecting [{}], was [{}]\n'.format(shape, a.shape))
                raise AssertionError(msg)
        # if passing...
        msg += 'Shape: All arrays have shape ' + str(shape) + '\n'
    elif isinstance(mode, str) and mode.lower() == 'broad':
        try:  # Raises ValueError if not broadcastable
            *args, = np.broadcast_arrays(*args)
        except ValueError as err:
            msg += str(err) + '\n'
            raise AssertionError(msg)
        msg += 'Shape: Arrays were broadcasted to shape '
        msg += str(args[0].shape) + '\n'
    else:
        raise ValueError(
            "Unexpected 'shape' mode: must be either 'strict' or 'broad'"
        )
    if args[0].size == 0:
        warnings.warn("Comparing 0-size arrays", UserWarning)
    return (msg,) + tuple(args)


def _treat_invalid(mode, *args, msg=''):
    msg += 'Invalid: '

    # Preemptively catch empty arrays and return
    if args[0].size == 0:
        msg += 'Comparing empty arrays'
        empty = np.array([], dtype=np.bool_)
        return (msg, empty, empty)

    if isinstance(mode, str) and mode.lower() == 'fail':
        # Fail as soon as NaN is encountered
        failures = False
        for a in args:
            failures = np.logical_or(failures, np.isnan(a))
        # Compare only values for which all arguments are valid
        to_cmp = np.logical_not(failures)
        # Format msg
        msg += 'Failing if any NaNs '
    elif isinstance(mode, str) and mode.lower() == 'allow':
        # Fail when the number of NaNs is greater than zero, but
        # lower than the number of arguments
        nans_counter = np.zeros(args[0].shape, np.int_)
        for a in args:
            nans_counter += np.isnan(a)
        failures = np.logical_and(
            nans_counter > 0,
            nans_counter < len(args))
        # Compare only when no NaN encountered
        to_cmp = nans_counter == 0
        # Format msg
        msg += 'Passing if all NaNs '
    elif isinstance(mode, np.ndarray) and mode.dtype == np.bool:
        # Here, mode is an array of boolean indexing invalid values
        #    to be ignored by the assertion
        mode = np.broadcast_to(mode, args[0].shape)
        failures = np.broadcast_to(False, args[0].shape)
        # Check not-ignored for NaNs
        not_ign = np.logical_not(mode)
        for a in args:
            # Persisting failures...
            new_failures = np.logical_or(failures, np.isnan(a))
            # Only on not-ignored
            failures = np.logical_and(
                not_ign,
                new_failures
            )
        # Compare not-ignored and non-NaN only
        to_cmp = np.logical_and(not_ign, np.logical_not(failures))
        # Format msg
        msg += 'Failing if any NaNs with ' + str(np.sum(mode)) + ' ignored '
    else:
        raise ValueError("Unexpected 'invalid' mode:"
                         "must be either 'fail', 'allow', or boolean array")

    # Add invalid summary statistics
    total_failures = np.sum(failures)
    total_to_compare = np.sum(to_cmp)
    msg += '['
    msg += str(total_failures) + ' invalid(s) over '
    msg += str(total_to_compare) + ' valid(s) '
    if total_to_compare > 0:
        rate = total_failures/total_to_compare*100.
        msg += '(' + '{:.2f}'.format(rate) + '%)'
    msg += ']\n'

    return (msg, failures, to_cmp)


def _debug_table(failures, to_cmp, *args, headers=None):
    # Collect indices
    idx = np.where(failures)
    idx_headers = ['dim' + str(n) for n in range(len(idx))]

    # Extract failing data points
    data = []
    for a in args:
        data.append(a[failures])

    # Make failure table
    msg = '\n' + 'Failures:' + '\n'
    msg += str(tabulate(zip(*idx, *data), headers=(*idx_headers, *headers)))
    msg += '\n'

    # Make table footer
    cmp_total = int(np.sum(to_cmp))
    cmp_failures = int(np.sum(np.logical_and(failures, to_cmp)))
    perc = int(cmp_failures/cmp_total*100)
    msg += ('\n' + str(cmp_failures) + ' failure(s) out of '
            + str(cmp_total) + ' (' + str(perc) + '%)\n')

    return msg

##############
# Derivative #
##############


def assert_derivative_at(derivative_fcn_or_val, at, *, function=None,
                         rtol=1e-4, atol=0, mode='central', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    if callable(derivative_fcn_or_val):
        value = derivative_fcn_or_val(at)
    else:
        value = derivative_fcn_or_val
    expected_value = pymaat.findiff.derivative_at(at,
                                                  function=function, mode=mode)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                        invalid=invalid)


def assert_gradient_at(gradient_fcn_or_val, at, *, function=None,
                       rtol=1e-4, atol=0, mode='central', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    if callable(gradient_fcn_or_val):
        value = gradient_fcn_or_val(at)
    else:
        value = gradient_fcn_or_val
    value = np.atleast_1d(value)
    expected_value = pymaat.findiff.gradient_at(at,
                                                function=function, mode=mode)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                        invalid=invalid)


def assert_jacobian_at(jacobian_fcn_or_val, at, *, function=None,
                       rtol=1e-4, atol=0, mode='central', invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    if callable(jacobian_fcn_or_val):
        value = jacobian_fcn_or_val(at)
    else:
        value = jacobian_fcn_or_val
    value = np.atleast_2d(value)
    expected_value = pymaat.findiff.jacobian_at(at,
                                                function=function, mode=mode)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                        invalid=invalid)


def assert_hessian_at(hessian_fcn_or_val, at, *,
                      function=None, gradient=None,
                      rtol=1e-2, atol=0, mode='central',
                      invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    if callable(hessian_fcn_or_val):
        value = hessian_fcn_or_val(at)
    else:
        value = hessian_fcn_or_val
    value = np.atleast_2d(value)
    if gradient is not None:
        expected_value = pymaat.findiff.jacobian_at(at,
                                                    function=gradient, mode=mode)
    else:
        expected_value = pymaat.findiff.hessian_at(at,
                                                   function=function, mode=mode)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                        invalid=invalid)


###############
# Integration #
###############

# TODO improve coverage
def assert_integral_until(integral, function, until,
                          lower_bound=-np.inf, rtol=1e-4, atol=0., invalid='fail'):
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    until = np.atleast_1d(until)
    if callable(integral):
        value = integral(until)
    else:
        value = integral
    expected_value = np.empty_like(until)
    for (i, u) in enumerate(until.flat):
        index = np.unravel_index(i, until.shape)
        expected_value[index], *_ = integrate.quad(function, lower_bound, u)
    assert_almost_equal(value, expected_value, rtol=rtol, atol=atol,
                        invalid=invalid)


############
# Function #
############


# TODO improve coverage
def assert_elbyel(function, ninput, noutput):
    """
    Basic test suite for element-by-element functions
    """
    # TODO: generalizes for different input / output type
    # now only supports float inputs
    __tracebackhide__ = not DEBUG  # Hide traceback for py.test
    with np.errstate(invalid='ignore'):
        _assert_elbyel_supports_scalar(function, ninput, noutput)
        _assert_elbyel_supports_same_size_array(function, ninput, noutput)
        _assert_elbyel_supports_broadcasting(function, ninput, noutput)


def _assert_elbyel_supports_scalar(f, nin, nout):
    in_ = (np.nan,)*nin
    out_ = f(*in_)
    if nout == 1:
        out_ = (out_,)
    assert isinstance(out_, tuple)
    assert len(out_) == nout
    for i in range(nout):
        assert out_[i].ndim == 0


def _assert_elbyel_supports_same_size_array(f, nin, nout):
    SHAPE = (1, 2, 3, 1, 4, 1)  # 24 elements
    in_ = (np.full(SHAPE, np.nan),)*nin
    out_ = f(*in_)
    if nout == 1:
        out_ = (out_,)
    assert isinstance(out_, tuple)
    assert len(out_) == nout
    for i in range(nout):
        assert out_[i].shape == SHAPE


def _assert_elbyel_supports_broadcasting(f, nin, nout):
    SHAPE = tuple(np.arange(nin)+2)  # ~nin! elements
    # To properly test broadcasting, all dimensions from output shape must
    # **not** be one
    assert not np.any(SHAPE == 1)
    in_ = []
    for (i, s) in enumerate(SHAPE):
        shape = np.ones((nin,), dtype=np.int)
        shape[i] = s
        shape = tuple(shape)
        in_.append(np.full(shape, np.nan))
    out_ = f(*in_)
    if nout == 1:
        out_ = (out_,)
    assert isinstance(out_, tuple)
    assert len(out_) == nout
    for i in range(nout):
        assert out_[i].shape == SHAPE
