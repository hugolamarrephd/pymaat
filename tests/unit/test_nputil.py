import pytest
import numpy as np

from pymaat.nputil import CouldNotGetView
from pymaat.nputil import flat_view, diag_view
from pymaat.nputil import atleast_1d, elbyel, workon_axis
import pymaat.testing as pt


class TestViews:

    @pytest.fixture(params=('C','F'), ids=['Row-Major', 'Column-Major'])
    def order(self, request):
        return request.param

    def test_flat_view_0D(self, order):
        nparr = np.array([], copy=True, order=order)
        expected = np.array([])
        pt.assert_equal(flat_view(nparr), expected)

    @pytest.mark.parametrize("shape", [(1,),(2,),(13,)])
    def test_flat_view_1D(self, shape, random_int, order):
        nparr = np.array(random_int, copy=True, order=order)
        expected = []
        for i in range(shape[0]):
            expected.append(nparr[i])
        pt.assert_equal(flat_view(nparr), expected)

    @pytest.mark.parametrize("shape", [(1,2),(2,3),(21,12)])
    def test_flat_view_2D(self, shape, random_int, order):
        nparr = np.array(random_int, copy=True, order=order)
        expected = []

        if order == 'C':
            for i in range(shape[0]):
                for j in range(shape[1]):
                        # Last index fastest (j)
                        expected.append(nparr[i,j])
        elif order == 'F':
            for j in range(shape[1]):
                for i in range(shape[0]):
                        # First index fastest (i)
                        expected.append(nparr[i,j])

        pt.assert_equal(flat_view(nparr), expected)

    @pytest.mark.parametrize("shape", [(1,2,3),(2,3,4),(2,7,5)])
    def test_flat_view_3D(self, shape, random_int, order):
        nparr = np.array(random_int, copy=True, order=order)
        expected = []

        if order == 'C':
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        # Last index fastest (k)
                        expected.append(nparr[i,j,k])
        elif order == 'F':
            for k in range(shape[2]):
                for j in range(shape[1]):
                    for i in range(shape[0]):
                        # First index fastest (i)
                        expected.append(nparr[i,j,k])
        pt.assert_equal(flat_view(nparr), expected)

    def test_flat_view_fails_after_slicing(self, order):
        initial = np.array(range(100), order=order)
        flat_view(initial) # ok after init
        reshaped = initial.reshape((10,10), order=order)
        flat_view(reshaped) # ok after reshape
        if order == 'C':
            flat_view(reshaped[5,:]) # Ok to slice rows when row-major
            with pytest.raises(CouldNotGetView):
                flat_view(reshaped[:,5])
        elif order == 'F':
            flat_view(reshaped[:,3]) # Ok to slice columns when column-major
            with pytest.raises(CouldNotGetView):
                flat_view(reshaped[3,:])
        # Always fail on irregular slicing
        with pytest.raises(CouldNotGetView):
            flat_view(reshaped[::2])

    @pytest.mark.parametrize("shape", [(1,10),(10,1),(2,2),(10,3),(3,10)])
    def test_diag_view(self, random_int, order):
        matrix = random_int
        max_shape = max(matrix.shape[0], matrix.shape[1])
        for k in range(-matrix.shape[0]+1, matrix.shape[1]):
            expected = []
            for i in range(max_shape):
                if i<matrix.shape[0] and (i+k>=0 and
                        i+k<matrix.shape[1]):
                    expected.append(matrix[i,i+k])
            value = diag_view(matrix, k=k)
            # print(f"k={k}")
            # print(f"matrix={matrix}")
            # print(f"expected={expected}")
            # print(f"value={value}")
            pt.assert_equal(value, expected)

class TestAtLeast1dDecorator:

    @pytest.fixture
    def atleast_1d_function(self):
        @atleast_1d
        def dummy_function(a, b, *, c=10):
            # print(f"a={a}, b={b}, c={c}")
            assert c is 10
            return (a, b)
        return dummy_function

    @pytest.mark.parametrize("shape", ([2,2],[2,1,4],[2,4,1,3]))
    def test_for_array(self, shape, random_int,
            atleast_1d_function):
        in1 = random_int[0]
        in2 = random_int[1]
        out1, out2 = atleast_1d_function(in1, in2)
        assert isinstance(out1, np.ndarray)
        assert isinstance(out2, np.ndarray)
        assert out1.dtype == np.int_
        assert out2.dtype == np.int_
        pt.assert_equal(in1, out1)
        pt.assert_equal(in2, out2)

    def test_for_scalar_float(self, atleast_1d_function):
        out1, out2 = atleast_1d_function(10., 10.)
        assert isinstance(out1, np.ndarray)
        assert isinstance(out2, np.ndarray)
        assert out1.dtype == np.float_
        assert out2.dtype == np.float_
        assert out1.ndim== 1
        assert out2.ndim== 1
        assert out1.size == 1
        assert out2.size == 1

    def test_for_scalar_int(self, atleast_1d_function):
        out1,out2 = atleast_1d_function(10, 10)
        assert isinstance(out1, np.ndarray)
        assert isinstance(out2, np.ndarray)
        assert out1.dtype == np.int_
        assert out2.dtype == np.int_
        assert out1.ndim== 1
        assert out2.ndim== 1
        assert out1.size == 1
        assert out2.size == 1

    def test_for_scalar_bool(self, atleast_1d_function):
        out1,out2 = atleast_1d_function(True, True)
        assert isinstance(out1, np.ndarray)
        assert isinstance(out2, np.ndarray)
        assert out1.dtype == np.bool_
        assert out2.dtype == np.bool_
        assert out1.ndim== 1
        assert out2.ndim== 1
        assert out1.size == 1
        assert out2.size == 1

class TestElByElDecorator():

    @pytest.fixture
    def single_output(self):
        @elbyel
        def dummy_function(a, *, c=10):
            # print(f"a={a}, b={b}, c={c}")
            assert c == -99
            return a
        return dummy_function

    @pytest.fixture
    def multi_output(self):
        @elbyel
        def dummy_function(a, b, *, c=10):
            # print(f"a={a}, b={b}, c={c}")
            assert c == -99
            return (a, b)
        return dummy_function

    @pytest.mark.parametrize("shape", ([2,2],[2,4,2],[2,4,1,3]))
    def test_multi_for_array(self, shape, random_int, single_output):
        out = single_output(random_int, c=-99)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int_
        pt.assert_equal(out, random_int)

    def test_multi_for_scalar_float(self, single_output):
        out = single_output(10., c=-99)
        assert out.ndim== 0
        assert out.size == 1
        assert isinstance(out, np.float_)

    def test_multi_for_scalar_int(self, single_output):
        out = single_output(10, c=-99)
        assert out.ndim== 0
        assert out.size == 1
        assert isinstance(out, np.int_)

    def test_single_for_scalar_bool(self, single_output):
        out = single_output(True, c=-99)
        assert out.ndim== 0
        assert out.size == 1
        assert isinstance(out, np.bool_)

    @pytest.mark.parametrize("shape", ([2,2],[2,4,2],[2,4,1,3]))
    def test_multi_for_array(self, shape, random_int, multi_output):
        in1 = random_int[0]
        in2 = random_int[1]
        out1, out2 = multi_output(in1, in2, c=-99)
        assert isinstance(out1, np.ndarray)
        assert isinstance(out2, np.ndarray)
        assert out1.dtype == np.int_
        assert out2.dtype == np.int_
        pt.assert_equal(out1, in1)
        pt.assert_equal(out2, in2)

    @pytest.mark.parametrize("shape", ([2,2],[2,4,2],[2,4,1,3]))
    def test_multi_for_array_mixed_scalar_first(self,
            shape, random_int, multi_output):
        in1 = 10
        in2 = random_int
        out1, out2 = multi_output(in1, in2, c=-99)
        assert isinstance(out1, np.ndarray)
        assert isinstance(out2, np.ndarray)
        assert out1.dtype == np.int_
        assert out2.dtype == np.int_
        pt.assert_equal(out1, in1)
        pt.assert_equal(out2, in2)

    @pytest.mark.parametrize("shape", ([2,2],[2,4,2],[2,4,1,3]))
    def test_multi_for_array_mixed_scalar_second(self,
            shape, random_int, multi_output):
        in1 = random_int
        in2 = 10
        out1, out2 = multi_output(in1, in2, c=-99)
        assert isinstance(out1, np.ndarray)
        assert isinstance(out2, np.ndarray)
        assert out1.dtype == np.int_
        assert out2.dtype == np.int_
        pt.assert_equal(out1, in1)
        pt.assert_equal(out2, in2)

    def test_multi_for_scalar_float(self, multi_output):
        out1,out2 = multi_output(10., 10., c=-99)
        assert out1.ndim== 0
        assert out2.ndim== 0
        assert out1.size == 1
        assert out2.size == 1
        assert isinstance(out1, np.float_)
        assert isinstance(out2, np.float_)

    def test_multi_for_scalar_int(self, multi_output):
        out1,out2 = multi_output(10, 10, c=-99)
        assert out1.ndim== 0
        assert out2.ndim== 0
        assert out1.size == 1
        assert out2.size == 1
        assert isinstance(out1, np.int_)
        assert isinstance(out2, np.int_)

    def test_multi_for_scalar_bool(self, multi_output):
        out1,out2 = multi_output(True, True, c=-99)
        assert out1.ndim== 0
        assert out2.ndim== 0
        assert out1.size == 1
        assert out2.size == 1
        assert isinstance(out1, np.bool_)
        assert isinstance(out2, np.bool_)

class TestWorkOnAxis():

    @pytest.fixture
    def single_output(self):
        @workon_axis
        def dummy_function(a, *, c=10):
            # print(f"a={a}, b={b}, c={c}")
            assert c == -99
            return a.sum(axis=0, keepdims=True)
        return dummy_function

    @pytest.fixture
    def multi_output(self):
        @workon_axis
        def dummy_function(a, b, *, c=10):
            # print(f"a={a}, b={b}, c={c}")
            assert c == -99
            return (a.sum(axis=0, keepdims=True),
                    b.sum(axis=0, keepdims=True))
        return dummy_function

    @pytest.mark.parametrize("shape", [(2,4), (2,2,1), (2,1,4,1)])
    @pytest.mark.parametrize("axis", [0,1,2,3])
    def test_single_output(self, shape, axis, random_int,
            single_output):
        axis = min(axis, len(shape)-1)
        value = single_output(random_int, c=-99, axis=axis)
        expected = random_int.sum(axis=axis, keepdims=True)
        pt.assert_equal(value, expected)

    @pytest.mark.parametrize("shape", [(2,4), (2,2,1), (2,1,4,1)])
    @pytest.mark.parametrize("axis", [0,1,2])
    def test_multi_output(self, shape, axis, random_int,
            multi_output):
        in1 = random_int[0]
        in2 = random_int[1]
        axis = min(axis, in1.ndim-1)
        (out1, out2) = multi_output(in1, in2, c=-99, axis=axis)
        pt.assert_equal(out1, in1.sum(axis=axis, keepdims=True))
        pt.assert_equal(out2, in2.sum(axis=axis, keepdims=True))

    @pytest.mark.parametrize("shape", [(2,4), (2,2,1), (2,1,4,1)])
    def test_defaults_to_0(self, random_int,
            single_output):
        out = single_output(random_int, c=-99)
        pt.assert_equal(out, random_int.sum(axis=0, keepdims=True))

    def test_supports_non_ndarray(self, single_output):
        a_list = [1,2,3,4]
        out = single_output(a_list, c=-99)
        pt.assert_equal(out, np.sum(a_list, axis=0, keepdims=True))
