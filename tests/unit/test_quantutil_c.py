import numpy as np
import pytest
import pymaat.quantutil_c as qutilc
from math import pi, ceil

import time
import pymaat.testing as pt

# class TestPartitionIdx:
#     size = 1000
#     nidx = 1000

#     @pytest.fixture
#     def values(self, seed):
#         np.random.seed(seed)
#         out = np.random.randint(-1000, 1000, size=self.size)
#         out = np.asarray(out, np.float_)
#         # Make sure we have some repeated values
#         out[0] = out[5] = out[10]
#         return out

#     @pytest.fixture
#     def idx(self, seed):
#         np.random.seed(seed>>1)
#         return np.random.randint(
#                 0, self.size, size=self.nidx, dtype=np.uint32)

#     @pytest.mark.parametrize("at", range(0,nidx))
#     def test(self, at, idx, values):
#         at = np.uint32(at)
#         qutilc._partition_idx(at, idx, values)
#         pt.assert_less_equal(values[idx[:at]], values[idx[at]],
#                 shape='broad')
#         pt.assert_greater_equal(values[idx[at+1:]], values[idx[at]],
#                 shape='broad')


@pytest.fixture(params=[189234, 23418, 12345])
def seed(self, request):
    np.random.seed(request.param)


@pytest.fixture(params=[2])
def ndim(request):
    return request.param


@pytest.fixture(params=[100, 1000, 10000])
def values(request, ndim):
    return np.random.normal(size=(request.param, ndim))


@pytest.fixture(params=[1000000])
def at(request, ndim):
    return np.random.normal(size=(request.param, ndim))


@pytest.fixture
def weight(ndim):
    return np.ones((ndim,))


def test_kd_tree_versus_naive(values, at, weight):
    start_naive = time.time()
    out_naive = qutilc.naive_nearest(values, at, weight)
    start_tree = end_naive = time.time()
    print("Naive: ", end='')
    print(end_naive - start_naive)
    out_tree = qutilc.KDTree(values).nearest(at, weight)
    end_tree = time.time()
    print("Tree: ", end='')
    print(end_tree - start_tree)
    print("Speed-up(%): ", end='')
    print(100.*((end_naive-start_naive)/(end_tree-start_tree)-1.))
    pt.assert_equal(out_naive, out_tree)
