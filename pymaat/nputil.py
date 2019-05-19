import contextlib
from functools import wraps

import numpy as np

from pymaat.util import PymaatException

##############
# Decorators #
##############


def workon_axis(function):
    r"""Extend operations on first axis to any specified axes.

    This decorator allows a behavior similar to NumPy
    when providing a keyword `axis` such as::

        np.cumsum(ndarr, axis=2)

    Wrapped functions must respect the following conditions:
        * All positional arguments are NumPy arrays
        * Performs operations on first axis only (i.e. axis==0)
        * All return values are NumPy arrays having
            the same number of dimensions as input arrays

    If all previous conditions are respected,
    decorating a function with `workon_axis` extends its functionality to any
    axes.

    Following NumPy convention, if no value is inputted for `axis`,
    input arrays are raveled.

    Parameters
    ----------
    None

    Returns
    -------
    function
        Decorated function.

    See Also
    --------
    reduced_axis

    Notes
    -----
    Prior to a wrapped call, the decorator swaps the first (axis==0) axis
    and the specified axis for all *positional* arguments.
    The axes of return values are later swapped back to their original
    positions after the call.

    Examples
    --------

    >>> def my_cumsum(a):
    ...     return np.cumsum(a, axis=0)
    >>> my_cumsum = workon_axis(my_cumsum)
    >>> print(my_cumsum([[1, 2, 3], [4, 5, 6]]))
    [  1  3  6 10 15 21]
    >>> print(my_cumsum([[1, 2, 3], [4, 5, 6]], axis=0))
    [[1 2 3]
     [5 7 9]]
    >>> print(my_sum([[1, 2, 3], [4, 5, 6]], axis=1))
    [[ 6]
     [15]]
    """

    @wraps(function)
    def wrapper(*args, **kargs):
        axis = kargs.pop('axis', None)

        if axis is None:
            # The default is to ravel input and output arrays
            # Warning: may copy data!
            def swap_axes(ndarr): return np.ravel(ndarr)
        else:
            # Always returns a view

            def swap_axes(ndarr): return np.swapaxes(ndarr, 0, axis)

        inputs = []
        for a in args:
            inputs.append(swap_axes(a))

        # Call wrapped function
        outputs = function(*inputs, **kargs)

        return _apply_to_all(outputs, swap_axes)

    return wrapper


def reduce_axis(function):
    r"""Extend reducing operations on first axis to any specified axis.

    This decorator allows a behavior similar to NumPy
    when providing keywords `axis` and `keepdims` such as::

        np.sum(ndarr, axis=2, keepdims=False)

    Wrapped functions must respect the following conditions:
        * All positional arguments are NumPy arrays
        * Performs operations on first axis only (i.e. axis=0)
        * Keep first axis i.e. `output.shape[0]=1` in all outputs
            (typically by setting `keepdims` to `True` in NumPy)
        * All return values are NumPy arrays

    If all previous conditions are respected,
    decorating a function with `reduce_axis`
    extends its functionality to any axes.
    When the `keepdims` flag is set to `False`,
    the decorator further squeezes the reduced axis for all output arrays.

    Following NumPy convention, if no value is inputted for `axis`,
    input arrays are raveled. The default for `keepdims` is `False.`

    Parameters
    ----------
    None

    Returns
    -------
    function
        Decorated function.

    See Also
    --------
    workon_axis

    Notes
    -----
    Prior to a wrapped call, the decorator swaps the first (axis==0) axis
    and the specified axis for all *positional* arguments.
    The axes of return values are later swapped back to their original
    positions after the call.
    Finally, the specified axis is purged from outputs when `keepdims` is set
    to `False`.

    Examples
    --------
    In the following example, the initial (i.e. non-decorated)
    function acts on the first axis only and keeps it in the final result:

    >>> def my_sum(a):
    ...     return np.sum(a, axis=0, keepdims=True)
    >>> my_sum = reduce_axis(my_sum)
    >>> print(my_sum([[1, 2, 3], [4, 5, 6]]))
    21
    >>> print(my_sum([[1, 2, 3], [4, 5, 6]], axis=0))
    [5 7 9]
    >>> print(my_sum([[1, 2, 3], [4, 5, 6]], axis=1, keepdims=False))
    [ 6 15]
    >>> print(my_sum([[1, 2, 3], [4, 5, 6]], axis=1, keepdims=True))
    [[ 6]
     [15]]
    """

    @wraps(function)
    def wrapper(*args, **kargs):
        axis = kargs.pop('axis', None)
        keepdims = kargs.pop('keepdims', False)

        outputs = workon_axis(function)(*args, **kargs, axis=axis)

        if keepdims:
            return outputs
        else:
            return _apply_to_all(
                outputs, lambda ndarr: np.squeeze(ndarr, axis=axis))

    return wrapper


def ravel(function):
    '''
    Decorated functions may safely assume all positional arguments are
        one-dimensional numpy arrays,
        but leaves keyword arguments **untouched**.
    '''
    def wrapper(*args, **kargs):
        old_args = args
        args = []
        for a in old_args:
            args.append(np.ravel(a))
        return function(*args, **kargs)
    return wraps(function)(wrapper)


def atleast_1d(function):
    '''
    Decorated functions may safely assume all positional arguments are
        numpy arrays with **at least* one dimension,
        but leaves keyword arguments **untouched**.
    '''
    def wrapper(*args, **kargs):
        old_args = args
        args = []
        for a in old_args:
            args.append(np.atleast_1d(a))
        return function(*args, **kargs)
    return wraps(function)(wrapper)


def elbyel(function):
    '''
    Broacasts all positional arguments,
        but leaves keyword arguments **untouched**.

    Decorated functions may hence safely assume all positional
        arguments are numpy arrays of same shape (with ndim>0).


    '''
    def wrapper(*args, **kargs):
        old_args = args
        args = []
        for a in old_args:
            args.append(np.atleast_1d(a))
        *args, = np.broadcast_arrays(*args)
        return function(*args, **kargs)
    return wraps(function)(wrapper)


##########################
# Inverse Cumulative Ops #
##########################


@workon_axis
def icumsum(ndarr):
    r"""Performs a cumulative sum from the last to the first element.

    Parameters
    ----------
    ndarr: array_like
        Input array

    Returns
    -------
    array_like
        Inverse cumulative sum

    See Also
    --------
    numpy.cumsum, icumprod

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print([x + 3 for x in a])
    [4, 5, 6]
    """

    return np.cumsum(ndarr[::-1])[::-1]


def icumprod(a):
    return np.cumprod(a[::-1])[::-1]

#################
# Special Views #
#################


class CouldNotGetView(PymaatException):
    r"""NumPy could not get a view.

    This exception essentially acts as a light wrapper around numpy's::

        AttributeError: incompatible shape for a non-contiguous array

    e.g. when trying to flatten array by setting `shape=-1`.

    Attributes
    ----------
    None

    See Also
    --------
    flat_view, diag_view

    Examples
    --------
    >>> a = np.arange(200)
    >>> a.shape = (20,10)
    >>> flat_view(a[::2])
    Traceback (most recent call last):
    ...
    nputil.CouldNotGetView: Numpy could not get view.
    Consider copying array prior to requesting view if memory is not a concern.
    """

    def __init__(self):
        super().__init__('Numpy could not get view.\n'
                         'Consider copying array prior to requesting view if '
                         'memory is not a concern.')


def flat_view(ndarr):
    r"""When possible, returns flattened view (i.e. 1D) in no specific order.

    This function guarantees `ndarr` is not copied,
    which may be useful when memory is a concern
    and/or when changes made in a flattened array must be reflected
    in the initial array i.e. `ndarr`.
    Flattened views follow the order in which `ndarr`
    is stored in memory, either row-major or column-major.
    These behaviors are in stark contrast with `numpy.ravel`,
    which guarantees row-major order by default,
    but may have to *copy* the underlying data to do so.

    'flat_view' may be used to perform element-by-element computations
    *when numpy vectorization is unavailable*.
    For example when relying on native C (double to double) functions
    in Cython, we may explicitly cast arrays on 1D memoryviews
    as follows::

        cdef:
            int n, N = x.size
            double[:] _x, _out
        _x = flat_view(x)
        _out = flat_view(out)
        for n in range(N):
            _out[n] = _native_C_function(_x[n])

    Parameters
    ----------
    ndarr : array_like
        Input array

    Returns
    -------
    array_like
        Flat view of `ndarr`

    Raises
    ------
    CouldNotGetView
        Raised when `numpy.ravel` would have to *copy* the underlying data.

    See Also
    --------
    numpy.ravel, numpy.nditer

    Examples
    --------
    The following example is for illustration purposes only.
    Vectorized code (i.e. `a**2`) is always more efficient here.

    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> out = np.empty_like(a)
    >>> _a = flat_view(a)
    >>> _out = flat_view(out)
    >>> for i in range(a.size):
    ...     _out[i] = _a[i]**2
    >>> print(out)
    [[ 1  4  9]
     [16 25 36]]

    The following example shows how the behavior of `flat_view`
    differs from `numpy.ravel`.
    Note that when `OWNDATA` is set to False, the array is a view.

    >>> a = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    >>> raveled = np.ravel(a)
    >>> flattened = flat_view(a)
    >>> print(raveled)
    [1 2 3 4 5 6]
    >>> print(flattened)
    [1 4 2 5 3 6]
    >>> print(raveled.flags['OWNDATA'])
    True
    >>> print(flattened.flags['OWNDATA'])
    False

    """

    if ndarr.flags['C_CONTIGUOUS']:
        view = ndarr.view()
    elif ndarr.flags['F_CONTIGUOUS']:
        view = ndarr.view().transpose()
    else:  # Non-contiguous array
        raise CouldNotGetView()
    try:
        view.shape = (-1,)  # Numpy *never* copies here!
    except (AttributeError):
        raise CouldNotGetView()
    return view


def diag_view(matrix, k=0):
    r"""When possible, returns a view of the specified diagonal for a matrix.

    As opposed to `numpy.diag`, this function can *not* be used to build
    diagonal matrices and does not accept 1D arrays.

    Parameters
    ----------
    matrix: array_like
        2D input array.
    k : int
        Order of the diagonal. Use k>0 for diagonals above the main diagonal,
        and k<0 for diagonals below the main diagonal.

    Returns
    -------
    array_like
        `k`-th diagonal view of `matrix`

    Raises
    ------
    CouldNotGetView
        Raised when `numpy.ravel` would have to *copy* the underlying data.

    See Also
    --------
    numpy.diag

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> print(diag_view(a))
    [1 5 9]
    >>> print(diag_view(a, k=1))
    [2 6]
    >>> print(diag_view(a, k=-1))
    [4 8]
    >>> print(diag_view(a).flags['OWNDATA'])
    False
    """

    if matrix.ndim != 2:
        raise ValueError('Only supports matrices (i.e. ndim==2)')
    else:
        shape = matrix.shape

    # The following may raise CouldNotGetView
    view = flat_view(matrix)

    if matrix.flags['C_CONTIGUOUS']:
        return _diag_view_C(view, shape, k)
    elif matrix.flags['F_CONTIGUOUS']:
        return _diag_view_F(view, shape, k)


############
# Printing #
############

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

#############
# Utilities #
#############


def _apply_to_all(outputs, fcn):
    # Standardize output to tuple
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    new_outputs = []
    for o in outputs:
        new_outputs.append(fcn(o))

    # De-standardize output
    if len(new_outputs) > 1:
        return new_outputs
    else:
        return new_outputs[0]


def _diag_view_C(view, shape, k):
    # C-Contiguous
    if k >= 0:
        if k >= shape[1]:
            raise ValueError('Unexpected k')
        start = k
    elif k < 0:
        if k <= -shape[0]:
            raise ValueError('Unexpected k')
        start = -k * shape[1]
    else:
        raise ValueError

    step = shape[1]+1
    stop = start + _diag_size(shape, k) * step + 1

    return view[start:stop:step]


def _diag_view_F(view, shape, k):
    if k >= 0:
        if k >= shape[1]:
            raise ValueError('Unexpected k')
        start = k * shape[0]
    elif k < 0:
        if k <= -shape[0]:
            raise ValueError('Unexpected k')
        start = -k
    else:
        raise ValueError

    step = shape[0]+1
    stop = start + _diag_size(shape, k) * step + 1

    return view[start:stop:step]


def _diag_size(shape, k):
    min_shape = min(shape[0], shape[1])
    if k >= 0:
        return min_shape - 1 - max(0, min_shape + k - shape[1])
    else:
        return min_shape - 1 - max(0, min_shape - k - shape[0])
