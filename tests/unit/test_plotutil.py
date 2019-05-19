import pytest
import numpy as np

import pymaat.testing as pt

import pymaat.plotutil as pu


@pytest.fixture
def args():
    return (-np.ones((3, 4)), np.zeros((2, 1)), np.ones((1, 2)))


@pytest.fixture(params=[1, 5, 10, 20, 25])
def base(request):
    return request.param


def test_get_lim(args):
    lim = pu.get_lim(*args)
    pt.assert_almost_equal(lim, np.array([-1., 1.]))


def test_get_lim_margin(args):
    lim = pu.get_lim(*args, margin=0.1)
    pt.assert_almost_equal(lim, np.array([-2., 2.]))


def test_get_lim_formatter(args):
    lim = pu.get_lim(*args,
                     margin=0.1, formatter=lambda x: 100.*x)
    pt.assert_almost_equal(lim, np.array([-120., 120.]))


def test_get_lim_bounds(args):
    lim = pu.get_lim(*args,
                     bounds=[-50, 50],
                     formatter=lambda x: 100.*x)
    pt.assert_almost_equal(lim, np.array([-50., 50.]))


def test_get_lim_base_defaults_to_1(args, base):
    lim = pu.get_lim(*args, formatter=lambda x: 100.*x)
    pt.assert_equal(lim % 1, np.array([0, 0]))


def test_get_lim_base(args, base):
    lim = pu.get_lim(*args, base=base, formatter=lambda x: 100.*x)
    pt.assert_equal(lim % base, np.array([0, 0]))
