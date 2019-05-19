import numpy as np
import pytest

import pymaat.testing as pt
from pymaat.nputil import (CouldNotGetView, atleast_1d, diag_view, elbyel,
                           flat_view, reduce_axis, workon_axis)

##############
# Decorators #
##############


class TestWorkOnAxis():

    @pytest.fixture
    def single_output(self):
        @workon_axis
        def dummy_function(a, *, c=10):
            assert c == -99
            return a.cumsum(axis=0)
        return dummy_function

    @pytest.fixture
    def multi_output(self):
        @workon_axis
        def dummy_function(a, b, *, c=10):
            assert c == -99
            return (a.cumsum(axis=0),
                    b.cumsum(axis=0))
        return dummy_function

    @pytest.mark.parametrize("axis", [0, 1, 2, 3])
    def test_single_output(self, axis, single_output):
        shape = (2, 1, 4, 2)
        input = np.random.randint(-100, 100, size=shape)
        value = single_output(input, c=-99, axis=axis)
        expected = input.cumsum(axis=axis)
        pt.assert_equal(value, expected)

    @pytest.mark.parametrize("axis", [0, 1, 2, 3])
    def test_multi_output(self, axis, multi_output):
        shape = (2, 1, 4, 2)
        input1 = np.random.randint(-100, 100, size=shape)
        in2 = np.random.randint(-100, 100, size=shape)
        (out1, out2) = multi_output(input1, in2, c=-99, axis=axis)
        pt.assert_equal(out1, input1.cumsum(axis=axis))
        pt.assert_equal(out2, in2.cumsum(axis=axis))

    def test_default_axis(self, single_output):
        shape = (2, 1, 4, 2)
        ndarr = np.random.randint(-100, 100, size=shape)
        out = single_output(ndarr, c=-99)
        pt.assert_equal(out, ndarr.cumsum())

    def test_supports_non_ndarray(self, single_output):
        a_list = [1, 2, 3, 4]
        out = single_output(a_list, c=-99, axis=0)
        pt.assert_equal(out, np.cumsum(a_list, axis=0))

    def test_supports_non_ndarray_with_default(self, single_output):
        a_list = [1, 2, 3, 4]
        out = single_output(a_list, c=-99)
        pt.assert_equal(out, np.cumsum(a_list))


class TestReduceAxis():

    @pytest.fixture
    def single_output(self):
        @reduce_axis
        def dummy_function(a, *, c=10):
            assert c == -99
            return a.sum(axis=0, keepdims=True)
        return dummy_function

    @pytest.fixture
    def multi_output(self):
        @reduce_axis
        def dummy_function(a, b, *, c=10):
            assert c == -99
            return (a.sum(axis=0, keepdims=True),
                    b.sum(axis=0, keepdims=True))
        return dummy_function

    @pytest.mark.parametrize("axis", [0, 1, 2, 3])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_single_output(self, axis, keepdims, single_output):
        shape = (2, 1, 4, 2)
        ndarr = np.random.randint(-100, 100, size=shape)
        value = single_output(ndarr, c=-99, axis=axis, keepdims=keepdims)
        expected = ndarr.sum(axis=axis, keepdims=keepdims)
        pt.assert_equal(value, expected)

    @pytest.mark.parametrize("axis", [0, 1, 2, 3])
    def test_multi_output(self, axis, multi_output):
        shape = (2, 1, 4, 2)
        input1 = np.random.randint(-100, 100, size=shape)
        in2 = np.random.randint(-100, 100, size=shape)
        (out1, out2) = multi_output(input1, in2, c=-99, axis=axis)
        pt.assert_equal(out1, input1.cumsum(axis=axis))
        pt.assert_equal(out2, in2.cumsum(axis=axis))

    def test_default_axis(self, single_output):
        shape = (2, 1, 4, 2)
        ndarr = np.random.randint(-100, 100, size=shape)
        out = single_output(ndarr, c=-99)
        pt.assert_equal(out, ndarr.cumsum())

    def test_supports_non_ndarray(self, single_output):
        a_list = [1, 2, 3, 4]
        out = single_output(a_list, c=-99, axis=0)
        pt.assert_equal(out, np.cumsum(a_list, axis=0))

    def test_supports_non_ndarray_with_default(self, single_output):
        a_list = [1, 2, 3, 4]
        out = single_output(a_list, c=-99)
        pt.assert_equal(out, np.cumsum(a_list))


class TestICum:
    pass


class TestViews:

    @pytest.mark.parametrize("order", ['C', 'F'])
    def test_flat_view_0D(self, order):
        nparr = np.array([], copy=True, order=order)

        pt.assert_equal(flat_view(nparr), nparr)

    @pytest.mark.parametrize("order", ['C', 'F'])
    def test_flat_view_1D(self, order):
        nparr = np.empty((4,), dtype=np.int, order=order)

        pt.assert_equal(flat_view(nparr), nparr)

    def test_flat_view_2D_C_Order(self):
        shape = (2, 3)
        nparr = np.empty(shape, dtype=np.int, order='C')
        expected = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Last index first (j) for C order (i.e. Row-Major)
                expected.append(nparr[i, j])

        pt.assert_equal(flat_view(nparr), expected)

    def test_flat_view_2D_F_Order(self):
        shape = (4, 3)
        nparr = np.empty(shape, dtype=np.int, order='F')
        expected = []
        for j in range(shape[1]):
            for i in range(shape[0]):
                # First index first(i) for F order (i.e. Column-Majorh)
                expected.append(nparr[i, j])

        pt.assert_equal(flat_view(nparr), expected)

    def test_flat_view_3D_C_Order(self):
        shape = (4, 3, 7)
        nparr = np.empty(shape, dtype=np.int, order='C')
        expected = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    # Last index first (k)
                    expected.append(nparr[i, j, k])

        pt.assert_equal(flat_view(nparr), expected)

    def test_flat_view_3D_F_order(self):
        shape = (4, 3, 7)
        nparr = np.empty(shape, dtype=np.int, order='F')
        expected = []
        for k in range(shape[2]):
            for j in range(shape[1]):
                for i in range(shape[0]):
                    # First index first (i)
                    expected.append(nparr[i, j, k])
        pt.assert_equal(flat_view(nparr), expected)

    def test_flat_view_fails_after_slicing_C_order(self):
        initial = np.array(range(100), order='C')
        flat_view(initial)  # Ok after init
        reshaped = initial.reshape((10, 10), order='C')
        flat_view(reshaped)  # Ok after reshape
        flat_view(reshaped[5, :])  # Ok to slice rows when row-major
        with pytest.raises(CouldNotGetView):
            flat_view(reshaped[:, 5])
        with pytest.raises(CouldNotGetView):
            flat_view(reshaped[::2])  # Always fail on irregular slicing

    def test_flat_view_fails_after_slicing_F_order(self):
        initial = np.array(range(100), order='F')
        flat_view(initial)  # ok after init
        reshaped = initial.reshape((10, 10), order='F')
        flat_view(reshaped)  # ok after reshape
        flat_view(reshaped[:, 3])  # Ok to slice columns when column-major
        with pytest.raises(CouldNotGetView):
            flat_view(reshaped[3, :])
        with pytest.raises(CouldNotGetView):
            flat_view(reshaped[::2])  # Always fail on irregular slicing

    @pytest.mark.parametrize("order", ['C', 'F'])
    def test_diag_view(self, order):
        shape = (3, 10)
        matrix = np.empty(shape, dtype=np.int)
        max_shape = max(shape[0], shape[1])
        for k in range(-shape[0]+1, shape[1]):
            expected = []
            for i in range(max_shape):
                if i < shape[0] and (i+k >= 0 and i+k < shape[1]):
                    expected.append(matrix[i, i+k])
            value = diag_view(matrix, k=k)
            pt.assert_equal(value, expected)

#
# class TestAtLeast1dDecorator:
#
#     @pytest.fixture
#     def atleast_1d_function(self):
#         @atleast_1d
#         def dummy_function(a, b, *, c=10):
#             # print(f"a={a}, b={b}, c={c}")
#             assert c is 10
#             return (a, b)
#         return dummy_function
#
#     @pytest.mark.parametrize("shape", ([2, 2], [2, 1, 4], [2, 4, 1, 3]))
#     def test_for_array(self, shape, random_int,
#                        atleast_1d_function):
#         in1 = random_int[0]
#         in2 = random_int[1]
#         out1, out2 = atleast_1d_function(in1, in2)
#         assert isinstance(out1, np.ndarray)
#         assert isinstance(out2, np.ndarray)
#         assert out1.dtype == np.int_
#         assert out2.dtype == np.int_
#         pt.assert_equal(in1, out1)
#         pt.assert_equal(in2, out2)
#
#     def test_for_scalar_float(self, atleast_1d_function):
#         out1, out2 = atleast_1d_function(10., 10.)
#         assert isinstance(out1, np.ndarray)
#         assert isinstance(out2, np.ndarray)
#         assert out1.dtype == np.float_
#         assert out2.dtype == np.float_
#         assert out1.ndim == 1
#         assert out2.ndim == 1
#         assert out1.size == 1
#         assert out2.size == 1
#
#     def test_for_scalar_int(self, atleast_1d_function):
#         out1, out2 = atleast_1d_function(10, 10)
#         assert isinstance(out1, np.ndarray)
#         assert isinstance(out2, np.ndarray)
#         assert out1.dtype == np.int_
#         assert out2.dtype == np.int_
#         assert out1.ndim == 1
#         assert out2.ndim == 1
#         assert out1.size == 1
#         assert out2.size == 1
#
#     def test_for_scalar_bool(self, atleast_1d_function):
#         out1, out2 = atleast_1d_function(True, True)
#         assert isinstance(out1, np.ndarray)
#         assert isinstance(out2, np.ndarray)
#         assert out1.dtype == np.bool_
#         assert out2.dtype == np.bool_
#         assert out1.ndim == 1
#         assert out2.ndim == 1
#         assert out1.size == 1
#         assert out2.size == 1
#
#
# class TestElByElDecorator():
#
#     @pytest.fixture
#     def single_output(self):
#         @elbyel
#         def dummy_function(a, *, c=10):
#             # print(f"a={a}, b={b}, c={c}")
#             assert c == -99
#             return a
#         return dummy_function
#
#     @pytest.fixture
#     def multi_output(self):
#         @elbyel
#         def dummy_function(a, b, *, c=10):
#             # print(f"a={a}, b={b}, c={c}")
#             assert c == -99
#             return (a, b)
#         return dummy_function
#
#     @pytest.mark.parametrize("shape", ([2, 2], [2, 4, 2], [2, 4, 1, 3]))
#     def test_multi_for_array(self, shape, random_int, single_output):
#         out = single_output(random_int, c=-99)
#         assert isinstance(out, np.ndarray)
#         assert out.dtype == np.int_
#         pt.assert_equal(out, random_int)
#
#     def test_multi_for_scalar_float(self, single_output):
#         out = single_output(10., c=-99)
#         assert out.ndim == 0
#         assert out.size == 1
#         assert isinstance(out, np.float_)
#
#     def test_multi_for_scalar_int(self, single_output):
#         out = single_output(10, c=-99)
#         assert out.ndim == 0
#         assert out.size == 1
#         assert isinstance(out, np.int_)
#
#     def test_single_for_scalar_bool(self, single_output):
#         out = single_output(True, c=-99)
#         assert out.ndim == 0
#         assert out.size == 1
#         assert isinstance(out, np.bool_)
#
#     @pytest.mark.parametrize("shape", ([2, 2], [2, 4, 2], [2, 4, 1, 3]))
#     def test_multi_for_array(self, shape, random_int, multi_output):
#         in1 = random_int[0]
#         in2 = random_int[1]
#         out1, out2 = multi_output(in1, in2, c=-99)
#         assert isinstance(out1, np.ndarray)
#         assert isinstance(out2, np.ndarray)
#         assert out1.dtype == np.int_
#         assert out2.dtype == np.int_
#         pt.assert_equal(out1, in1)
#         pt.assert_equal(out2, in2)
#
#     @pytest.mark.parametrize("shape", ([2, 2], [2, 4, 2], [2, 4, 1, 3]))
#     def test_multi_for_array_mixed_scalar_first(self,
#                                                 shape, random_int, multi_output):
#         in1 = 10
#         in2 = random_int
#         out1, out2 = multi_output(in1, in2, c=-99)
#         assert isinstance(out1, np.ndarray)
#         assert isinstance(out2, np.ndarray)
#         assert out1.dtype == np.int_
#         assert out2.dtype == np.int_
#         pt.assert_equal(out1, in1)
#         pt.assert_equal(out2, in2)
#
#     @pytest.mark.parametrize("shape", ([2, 2], [2, 4, 2], [2, 4, 1, 3]))
#     def test_multi_for_array_mixed_scalar_second(self,
#                                                  shape, random_int, multi_output):
#         in1 = random_int
#         in2 = 10
#         out1, out2 = multi_output(in1, in2, c=-99)
#         assert isinstance(out1, np.ndarray)
#         assert isinstance(out2, np.ndarray)
#         assert out1.dtype == np.int_
#         assert out2.dtype == np.int_
#         pt.assert_equal(out1, in1)
#         pt.assert_equal(out2, in2)
#
#     def test_multi_for_scalar_float(self, multi_output):
#         out1, out2 = multi_output(10., 10., c=-99)
#         assert out1.ndim == 0
#         assert out2.ndim == 0
#         assert out1.size == 1
#         assert out2.size == 1
#         assert isinstance(out1, np.float_)
#         assert isinstance(out2, np.float_)
#
#     def test_multi_for_scalar_int(self, multi_output):
#         out1, out2 = multi_output(10, 10, c=-99)
#         assert out1.ndim == 0
#         assert out2.ndim == 0
#         assert out1.size == 1
#         assert out2.size == 1
#         assert isinstance(out1, np.int_)
#         assert isinstance(out2, np.int_)
#
#     def test_multi_for_scalar_bool(self, multi_output):
#         out1, out2 = multi_output(True, True, c=-99)
#         assert out1.ndim == 0
#         assert out2.ndim == 0
#         assert out1.size == 1
#         assert out2.size == 1
#         assert isinstance(out1, np.bool_)
#         assert isinstance(out2, np.bool_)
#
#
