import math

import pytest
import numpy as np
from scipy.stats import norm
import pymaat.testing as pt

import pymaat.mathutil as mu

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

# Error function (and normal cdf)

def test_fast_erfc(random_normal):
    x = random_normal
    value = mu.fast_erfc(x)
    assert value.shape == x.shape
    expected = np.empty_like(value)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        expected[it.multi_index] = math.erfc(x[it.multi_index])
        it.iternext()
    pt.assert_almost_equal(value, expected, atol=1e-6)

def test_fast_erfc_at_pm_inf():
    assert mu.fast_erfc(np.array([np.inf])) == 0.
    assert mu.fast_erfc(np.array([-np.inf])) == 2.

def test_fast_erf(random_normal):
    x = random_normal
    value = mu.fast_erf(x)
    assert value.shape == x.shape
    expected = np.empty_like(value)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        expected[it.multi_index] = math.erf(x[it.multi_index])
        it.iternext()
    pt.assert_almost_equal(value, expected, atol=1e-6)

def test_fast_erf_at_pm_inf():
    assert mu.fast_erf(np.array([np.inf])) == 1.
    assert mu.fast_erf(np.array([-np.inf])) == -1.

def test_fast_normcdf(random_normal):
    x = random_normal
    value = mu.fast_normcdf(x)
    expected = norm.cdf(x)
    pt.assert_almost_equal(value, expected, atol=1e-6)

def test_fast_normcdf_at_pm_inf():
    assert mu.fast_normcdf(np.array([np.inf])) == 1.
    assert mu.fast_normcdf(np.array([-np.inf])) == 0.

def test_normpdf(random_normal):
    x = random_normal
    value = mu.normpdf(x)
    expected = norm.pdf(x)
    pt.assert_almost_equal(value, expected)

def test_fast_normpdf_at_pm_inf():
    assert mu.normpdf(np.array([np.inf])) == 0.
    assert mu.normpdf(np.array([-np.inf])) == 0.

