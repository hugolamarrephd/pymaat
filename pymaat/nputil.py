from functools import wraps, update_wrapper
import contextlib

import numpy as np

from pymaat.util import PymaatException

def icumsum(a):
    return np.cumsum(a[::-1])[::-1]

def icumprod(a):
    return np.cumprod(a[::-1])[::-1]

#################
# Special Views #
#################

class CouldNotGetView(PymaatException):
    def __init__(self):
        super().__init__('Numpy could not preserve view when flattening. '
               'Consider copying prior to requesting view if '
               'memory is not a concern.')

def flat_view(ndarr):
    if ndarr.flags['C_CONTIGUOUS']:
        view = ndarr.view()
    elif ndarr.flags['F_CONTIGUOUS']:
        view = ndarr.view().transpose()
    else:
        raise CouldNotGetView()
    try:
        view.shape = (-1,) # Numpy *never* copies here!
    except (AttributeError):
        raise CouldNotGetView()
    return view

def diag_view(ndarr, k=0):
    if ndarr.ndim != 2:
        raise ValueError('Only supports ndim==2')
    else:
        shape = ndarr.shape

    view = flat_view(ndarr)

    if ndarr.flags['C_CONTIGUOUS']:
        return _diag_view_C(view, shape, k)
    elif ndarr.flags['F_CONTIGUOUS']:
        return _diag_view_F(view, shape, k)

# C-Contiguous
def _diag_view_C(view, shape, k):
    if k>=0:
        if k>=shape[1]:
            raise ValueError('Unexpected k')
        start = k
    elif k<0:
        if k<=-shape[0]:
            raise ValueError('Unexpected k')
        start = -k * shape[1]
    else:
        raise ValueError

    step = shape[1]+1
    stop = start + _diag_size(shape, k) * step + 1

    return view[start:stop:step]

# F-Contiguous
def _diag_view_F(view, shape, k):
    if k>=0:
        if k>=shape[1]:
            raise ValueError('Unexpected k')
        start = k * shape[0]
    elif k<0:
        if k<=-shape[0]:
            raise ValueError('Unexpected k')
        start = -k
    else:
        raise ValueError

    step = shape[0]+1
    stop = start + _diag_size(shape, k) * step + 1

    return view[start:stop:step]

def _diag_size(shape, k):
    min_shape = min(shape[0], shape[1])
    if k>=0:
        return min_shape - 1 - max(0, min_shape + k - shape[1])
    else:
        return min_shape - 1 - max(0, min_shape - k - shape[0])

##############
# Decorators #
##############

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

def workon_axis(function):
    '''
    Use when performing operations across a (single) axis
        specified by the keyword argument `axis`, eg.
    ```
        function(input1, input2, axis=0)
    ```

    This decorator swaps the inputted axis to the first position
        such that wrapped functions may systematically perform operations
        across axis=0.

    Raises AssertionError if output is not np.ndarray

    Rem. Other keyword arguments (besides `axis`) are left untouched.
    '''

    def wrapper(*args, **kargs):

        if 'axis' not in kargs:
            axis = 0
        else:
            axis = kargs.pop('axis')

        # Process inputs
        inputs = []
        for a in args:
            a = np.asanyarray(a)
            inputs.append(np.swapaxes(a, 0, axis)) # Always returns a view

        # Call wrapped function
        outputs = function(*inputs, **kargs)

        # Process outputs (clean up)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        new_outputs = []
        for o in outputs:
            assert isinstance(o, np.ndarray)
            new_outputs.append(np.swapaxes(o, 0, axis))

        if len(new_outputs)>1:
            return new_outputs
        else:
            return new_outputs[0]
    wrapper = atleast_1d(wrapper)
    return wraps(function)(wrapper)


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
