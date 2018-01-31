from functools import wraps

import numpy as np

######################
# Language Utilities #
######################

class lazy_property:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            # replace method by property...
            setattr(instance, self.func.__name__, value)
            return value

###################
# Numpy Utilities #
###################

def np_method(instance_method):
    '''
    Decorator to be applied to an object's (instance) method

    Pre-processing:
        Casts scalar (ie non-ndarray) positional arguments
        to ndarray **with at least 1D**.
        In particular, this allows wrapped methods to use assignment
            and positional indexing.
        But leaves keyword arguments **untouched**.

    Post-processing:
        If **all** inputs are scalar, return scalars

    Rem. When inputting arrays, the decorated method maintains array
        subclasses (eg. np.ma.array)
    Rem. Outputted scalars are not of native types.
        In particular, outputs are np.float64, instead of float.
    '''
    @wraps(instance_method)
    def wrapper(self, *args, **kargs):
        # Processing positional inputs
        inputs = []
        scalar = True
        for a in args:
            if not isinstance(a, np.ndarray):
                # Assumes input is scalar and
                #   try to convert to ndim-1 array of size 1
                inputs.append(np.atleast_1d(np.asarray(a)))
            else:
                # Input untouched (ie preserve subclass)
                inputs.append(a)
                scalar = False

        # Call wrapped instance method with all 1D inputs
        outputs = instance_method(self, *inputs, **kargs)
        if scalar:
            # Post-process output (if any)
            #   and revert to 0D (scalar) outputs
            if outputs is None:
                return outputs
            elif isinstance(outputs, tuple):
                return tuple(out_[0] for out_ in outputs)
            else:
                return outputs[0]
        else:
            return outputs
    return wrapper
