import math

import pytest
import numpy as np
from scipy.stats import norm
import pymaat.testing as pt

import pymaat.mathutil as mu
import pymaat.mathutil_c as muc

import time

@pytest.fixture(params=[(1,), (10,9), (2,4,7), (100,100)])
def shape(request):
    return request.param

#####################
# Special functions #
#####################

# Logistic

def test_logistic_at_0():
    np.isclose(mu.logistic(np.array([0.])), 0.5)

def test_logistic_at_inf():
    np.isclose(mu.logistic(np.array([np.inf])), 1.)

def test_logistic_at_minf():
    np.isclose(mu.logistic(np.array([-np.inf])), 0.)

def test_dlogistic(random_normal):
    x = random_normal
    pt.assert_derivative_at(mu.dlogistic(x), x,
            function=mu.logistic,
            atol=1e-6)

def test_dlogistic_at_pm_inf():
    np.isclose(mu.dlogistic(np.inf), np.array([0.]))
    np.isclose(mu.dlogistic(-np.inf), np.array([0.]))

def test_ddlogistic_at_pm_inf():
    np.isclose(mu.ddlogistic(np.inf), np.array([0.]))
    np.isclose(mu.ddlogistic(-np.inf), np.array([0.]))

def test_ddlogistic(random_normal):
    x = random_normal
    pt.assert_derivative_at(mu.ddlogistic(x), x,
            function=mu.dlogistic, atol=1e-6)

def test_ilogistic(random_normal):
    x = random_normal
    l = mu.logistic(x)
    pt.assert_almost_equal(x, mu.ilogistic(l))

def test_ilogistic_at_one():
    assert mu.ilogistic(np.array([1.])) == np.inf

def test_ilogistic_at_zero():
    assert mu.ilogistic(np.array([0.])) == -np.inf

def test_ilogistic_outside_space():
    assert np.isnan(mu.ilogistic(np.array([1.5])))
    assert np.isnan(mu.ilogistic(np.array([-1.5])))


# C-level Normal PDF/CDF (Scalar only)

def test_normcdf(random_normal):
    x = random_normal
    value = np.empty_like(x)
    s = time.time()
    muc.normcdf(x, value)
    e = time.time()
    print("My cdf: {}".format(e-s))
    s = time.time()
    expected = norm.cdf(x)
    e = time.time()
    print("Scipy cdf: {}".format(e-s))
    pt.assert_almost_equal(
            value, expected, rtol=1e-6, atol=0)

def test_normcdf_at_pm_inf():
    v = np.empty((1,))
    muc.normcdf(np.array(np.inf),v)
    assert v == 1.
    muc.normcdf(np.array(-np.inf),v)
    assert v == 0.

def test_normpdf(random_normal):
    x = random_normal
    value = np.empty_like(x)
    muc.normpdf(x, value)
    expected = norm.pdf(x)
    pt.assert_almost_equal(value, expected,
            rtol=1e-6, atol=0)

def test_normpdf_at_pm_inf():
    v = np.empty((1,))
    muc.normpdf(np.array(np.inf), v)
    assert v == 0.
    muc.normpdf(np.array(-np.inf), v)
    assert v == 0.
