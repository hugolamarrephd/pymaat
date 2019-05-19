import numpy as np
import pytest

from pymaat.util import lazy_property, method_decorator


# Lazy property

@pytest.fixture
def class_with_lazy_property():

    class dummy_class:
        side_effect = 0

        @lazy_property
        def dummy_property(self):
            dummy_class.side_effect += 1
            return 1234

    return dummy_class


def test_lazy_property(class_with_lazy_property):
    instance = class_with_lazy_property()
    assert instance.dummy_property == 1234
    assert class_with_lazy_property.side_effect == 1

# Method decorator


@pytest.fixture
def class_with_method_decorator():

    def dummy_decorator(function):
        return function

    class dummy_class:

        @method_decorator(dummy_decorator)
        def instance_method(self, a, *, b=3):
            self.a = a
            self.b = b

    return dummy_class


def test_method_decorator_for_instances(class_with_method_decorator):
    instance = class_with_method_decorator()
    instance.instance_method(1, b=2)
    assert instance.a == 1
    assert instance.b == 2
