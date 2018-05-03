from functools import wraps, update_wrapper

class PymaatException(Exception):
    pass

##############
# Properties #
##############

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

##############
# Decorators #
##############

def no_exception(function):
    def wrapper(*args, **kargs):
        try:
            return function(*args, **kargs)
        except:
            pass
    return wraps(function)(wrapper)

def method_decorator(decorator):
    """
    Converts a function decorator into a method decorator
        Rem. inspiration from django (See django/utils/decorators.py)
    """
    def new_decorator(function):
        def wrapper(self, *args, **kwargs):
            def bound_function(*args2, **kwargs2):
                return function(self, *args2, **kwargs2)
            return decorator(bound_function)(*args, **kwargs)
        return wraps(function)(wrapper)
    update_wrapper(new_decorator, decorator)
    new_decorator.__name__ = 'method_decorator(%s)' % decorator.__name__
    return new_decorator
