import numpy as np
import pytest

import pymaat.testing as pt


class TestBasicAssertions:

    @pytest.fixture
    def shape(self):
        return (3, 4)

    ###########
    # Any/All #
    ###########

    def test_assert_all_scalar(self):
        pt.assert_all(True)

    def test_assert_all_scalar_FAILURE(self):
        with pytest.raises(AssertionError):
            pt.assert_all(False)

    def test_assert_all_array(self, true):
        pt.assert_all(true)

    def test_assert_all_array_FAILURE(self, true):
        with pytest.raises(AssertionError):
            true[0] = False
            pt.assert_all(true)

    def test_assert_any_scalar(self):
        pt.assert_any(True)

    def test_assert_any_scalar_FAILURE(self):
        with pytest.raises(AssertionError):
            pt.assert_any(False)

    def test_assert_any_array(self, false):
        false[0] = True
        pt.assert_any(false)

    def test_assert_any_array_FAILURE(self, false):
        with pytest.raises(AssertionError):
            pt.assert_any(false)

    ############
    # Validity #
    ############

    def test_assert_valid_scalar(self):
        pt.assert_valid(1)

    def test_assert_valid_scalar_FAILURE(self):
        with pytest.raises(AssertionError):
            pt.assert_valid(np.nan)

    def test_assert_valid_array(self, random_normal):
        pt.assert_valid(random_normal)

    def test_assert_valid_array_FAILURE(self, random_normal):
        with pytest.raises(AssertionError):
            random_normal[0] = np.nan
            pt.assert_valid(random_normal)

    def test_assert_int_always_valid(self, random_int):
        pt.assert_valid(random_int)

    def test_assert_invalid_scalar(self):
        pt.assert_invalid(np.nan)

    def test_assert_invalid_scalar_FAILURE(self):
        with pytest.raises(AssertionError):
            pt.assert_invalid(1)

    def test_assert_invalid_array(self, nan):
        pt.assert_invalid(nan)

    def test_assert_invalid_array_FAILURE(self, nan):
        with pytest.raises(AssertionError):
            nan[0] = 1.
            pt.assert_invalid(nan)

    def test_assert_int_never_invalid_FAILURE(self, random_int):
        with pytest.raises(AssertionError):
            pt.assert_invalid(random_int)

    ##########
    # Finite #
    ##########

    def test_assert_finite_scalar(self):
        pt.assert_finite(1.)

    def test_assert_finite_scalar_FAILURE(self):
        with pytest.raises(AssertionError):
            pt.assert_finite(np.inf)
        with pytest.raises(AssertionError):
            pt.assert_finite(-np.inf)

    def test_assert_finite_array(self, random_normal):
        pt.assert_finite(random_normal)

    def test_assert_finite_array_inf_FAILURE(self, random_normal):
        random_normal[0] = np.inf
        with pytest.raises(AssertionError):
            pt.assert_finite(random_normal)

    def test_assert_finite_array_minf_FAILURE(self, random_normal):
        random_normal[0] = -np.inf
        with pytest.raises(AssertionError):
            pt.assert_finite(random_normal)


_strict_equal = [
    (1, 1),
    (True, True),
    (False, False),
    (0., 0.),
    (1., 1.),
    (np.inf, np.inf),
    (-np.inf, -np.inf)]

_strict_inequal = [
    (1, 2),
    (True, False),
    (False, True),
    (0., 1.),
    (np.inf, -np.inf)]

_strict_less = [
    (0, 1),
    (False, True),
    (0., 1.),
    (0., np.inf),
    (-np.inf, 0.),
    (-np.inf, np.inf)]

_strict_greater = [(2, 1),
                   (True, False),
                   (2., 1.),
                   (np.inf, 0.),
                   (0, -np.inf),
                   (np.inf, -np.inf)]

BASIC_COMPARE_SETUP = {

    'equal': {
        'assert': pt.assert_equal,
        'pass': _strict_equal,
        'fail': _strict_inequal,
        'floatarr': (
            [[1., 2.], [3., 4.]],
            [[1., 2.], [3., 4.]],
        ),
    },

    'not_equal': {
        'assert': pt.assert_not_equal,
        'pass': _strict_inequal,
        'fail': _strict_equal,
        'floatarr': (
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
        ),
    },

    'almost_equal': {
        'assert': pt.assert_almost_equal,
        'pass': _strict_equal + [
            (1.-1e-7, 1.),  # Relative tolerance defaults to 1e-6
        ],
        'fail': _strict_inequal + [
            (1.-1e-5, 1.),  # Relative tolerance defaults to 1e-6
            (0., np.finfo(np.float_).tiny),
        ],
        'floatarr': ([[1., 2.], [3., 4.]],
                     [[1.*(1.+1e-7), 2.*(1.-1e-7)], [3.*(1.+1e-7), 4.*(1.-1e-7)]])
    },

    'less_equal': {
        'assert': pt.assert_less_equal,
        'pass': _strict_equal + _strict_less,
        'fail': _strict_greater,
        'floatarr': (
            [[1., 2.], [3., 4.]],
            [[1., 2.], [4., 5.]],
        ),
    },

    'less': {
        'assert': pt.assert_less,
        'pass': _strict_less,
        'fail': _strict_equal + _strict_greater,
        'floatarr': (
            [[1., 2.], [3., 4.]],
            [[2., 3.], [4., 5.]],
        ),
    },

    'greater_equal': {
        'assert': pt.assert_greater_equal,
        'pass': _strict_equal + _strict_greater,
        'fail': _strict_less,
        'floatarr': (
            [[1., 2.], [3., 4.]],
            [[1., 2.], [2., 3.]],
        ),
    },

    'greater': {
        'assert': pt.assert_greater,
        'pass': _strict_greater,
        'fail': _strict_equal + _strict_less,
        'floatarr': (
            [[1., 2.], [3., 4.]],
            [[0., 1.], [2., 3.]],
        ),
    },
}


def all_():
    out = []
    for (n, param) in BASIC_COMPARE_SETUP.items():
        out.append((n, param['assert']))
    return out


def all_pass():
    out = []
    for (n, param) in BASIC_COMPARE_SETUP.items():
        out.extend([
            (str(p[0]) + ' ' + n + ' ' + str(p[1]),
             param['assert'],
             p)
            for p in param['pass']])
    return out


def all_fail():
    out = []
    for (n, param) in BASIC_COMPARE_SETUP.items():
        out.extend([
            (str(f[0]) + ' ' + n + ' ' + str(f[1]),
             param['assert'],
             f)
            for f in param['fail']])
    return out


def all_floatarr():
    out = []
    for (n, param) in BASIC_COMPARE_SETUP.items():
        out.append((n, param['assert'], param['floatarr']))
    return out


class TestComparePass:

    @pytest.fixture(params=all_pass(), ids=lambda x: x[0])
    def case(self, request):
        return request.param

    @pytest.fixture
    def assert_(self, case):
        return case[1]

    @pytest.fixture
    def at(self, case):
        return case[2]

    def test(self, assert_, at):
        assert_(*at)


class TestCompareFail:

    @pytest.fixture(params=all_fail(), ids=lambda x: x[0])
    def case(self, request):
        return request.param

    @pytest.fixture
    def assert_(self, case):
        return case[1]

    @pytest.fixture
    def at(self, case):
        return case[2]

    def test(self, assert_, at):
        with pytest.raises(AssertionError):
            assert_(*at)


class TestEmptyArraysCompare:

    @pytest.fixture(params=all_(), ids=lambda x: x[0])
    def case(self, request):
        case = request.param
        return case

    @pytest.fixture
    def assert_(self, case):
        return case[1]

    def test(self, assert_):
        assert_(np.array([], dtype=np.float_),
                np.array([], dtype=np.float_))


class TestFloatArrayCompare:

    @pytest.fixture(params=all_floatarr(),
                    ids=[f_[0] for f_ in all_floatarr()])
    def case(self, request):
        case = request.param
        return case

    @pytest.fixture
    def assert_(self, case):
        return case[1]

    @pytest.fixture
    def at(self, case):
        at = case[2]
        return (np.array(at[0], dtype=np.float_),
                np.array(at[1], dtype=np.float_))

    def test(self, assert_, at):
        assert_(*at)

    def test_wrong_shape_value_error(self, assert_, at):
        with pytest.raises(ValueError):
            assert_(*at, shape=None)

    def test_wrong_invalid_value_error(self, assert_, at):
        with pytest.raises(ValueError):
            assert_(*at, invalid=None)

    def test_shape_defaults_to_strict_and_mismatch_FAIL(self, assert_, at):
        at[1].shape = -1
        with pytest.raises(AssertionError):
            assert_(*at)
        with pytest.raises(AssertionError):
            assert_(*at, shape='strict')

    def test_specified_shape(self, assert_, at):
        shape = at[0].shape
        assert_(*at, shape=shape)

    def test_specified_shape_FAIL(self, assert_, at):
        wrong_shape = list(at[0].shape)
        wrong_shape[0] += 1
        with pytest.raises(AssertionError):
            assert_(*at, shape=tuple(wrong_shape))

    def test_shape_broad_mismatch_FAIL(self, assert_, at):
        at[1].shape = -1
        with pytest.raises(AssertionError):
            assert_(*at, shape='broad')

    def test_shape_broad(self, assert_, at):
        to_be_broad = at[0].copy()
        to_be_broad.shape = -1
        to_be_broad = to_be_broad[:, np.newaxis]
        full_sized = at[1]
        full_sized.shape = -1
        full_sized = np.tile(full_sized[:, np.newaxis], (1, 5))
        assert_(to_be_broad, full_sized, shape='broad')

    def test_defaults_to_fail_and_both_NaN_FAILURE(self, assert_, at):
        at[0][0, 0] = np.nan
        at[1][0, 0] = np.nan
        # Fails even if both are NaN...
        with pytest.raises(AssertionError):
            assert_(*at, invalid='fail')
        with pytest.raises(AssertionError):
            assert_(*at)  # Defaults to invalid='fail'

    def test_allow_when_both_NaN(self, assert_, at):
        at[0][0, 0] = np.nan
        at[1][0, 0] = np.nan
        assert_(*at, invalid='allow')  # Ok if both NaN

    def test_allow_when_only_one_is_NaN_FAILURE(self, assert_, at):
        at[0][0, 0] = np.nan
        with pytest.raises(AssertionError):
            assert_(*at, invalid='allow')

    def test_ignore_NaN(self, assert_, at):
        ignore = np.full(at[0].shape, False)
        ignore[0, 0] = True
        at[0][0, 0] = np.nan
        assert_(*at, invalid=ignore)  # Ok since ignoring element [0,0]


class TestAlmostEquality:

    def test_rtol_defauls_to_1em6(self):
        pt.assert_almost_equal(1.-1e-7, 1.)
        with pytest.raises(AssertionError):
            pt.assert_almost_equal(1.-1e-5, 1.)

    def test_atol_defauls_to_zero(self):
        tiny = np.finfo(np.float_).tiny
        with pytest.raises(AssertionError):
            pt.assert_almost_equal(tiny, tiny*2, rtol=0)

    def test_negative_tolerance_raises_value_error(self):
        with pytest.raises(ValueError):
            pt.assert_almost_equal(1., 1., rtol=-1)
        with pytest.raises(ValueError):
            pt.assert_almost_equal(1., 1., atol=-1)


class TestFiniteDifferenceAssertions:

    # Derivative assertion
    def test_assert_good_callable_derivative(self):
        # Does not raise AssertionError
        pt.assert_derivative_at(lambda x: 2.*x, 1.02,
                                function=lambda x: x**2.)

    def test_assert_good_derivative_value(self):
        # Does not raise AssertionError
        pt.assert_derivative_at(2.04, 1.02, function=lambda x: x**2.)

    def test_assert_bad_callable_derivative(self):
        with pytest.raises(AssertionError):
            pt.assert_derivative_at(lambda x: x+2., 1.02,
                                    function=lambda x: x**2.)

    def test_assert_bad_derivative_value(self):
        with pytest.raises(AssertionError):
            pt.assert_derivative_at(32.12, 1.02,
                                    function=lambda x: x**2.)

    # Gradient assertion
    def test_assert_good_callable_gradient(self):
        # Does not raise AssertionError
        pt.assert_gradient_at(lambda x: 2.*x, 1.02,
                              function=lambda x: x**2.)

    def test_assert_good_gradient_value(self):
        # Does not raise AssertionError
        pt.assert_gradient_at(2.04, 1.02,
                              function=lambda x: x**2.)

    def test_assert_bad_callable_gradient(self):
        with pytest.raises(AssertionError):
            pt.assert_gradient_at(lambda x: x+2., 1.02,
                                  function=lambda x: x**2.)

    def test_assert_bad_gradient_value(self):
        with pytest.raises(AssertionError):
            pt.assert_gradient_at(32.12, 1.02,
                                  function=lambda x: x**2.)

    # Jacobian assertion
    def test_assert_good_callable_jacobian(self):
        # Does not raise AssertionError
        pt.assert_jacobian_at(lambda x: 2.*x, 1.02,
                              function=lambda x: x**2.)

    def test_assert_good_jacobian_value(self):
        # Does not raise AssertionError
        pt.assert_jacobian_at(2.04, 1.02, function=lambda x: x**2.)

    def test_assert_bad_callable_jacobian(self):
        with pytest.raises(AssertionError):
            pt.assert_jacobian_at(lambda x: x+2., 1.02,
                                  function=lambda x: x**2.)

    def test_assert_bad_jacobian_value(self):
        with pytest.raises(AssertionError):
            pt.assert_jacobian_at(32.12, 1.02,
                                  function=lambda x: x**2.)

    # Hessian assertion
    def test_assert_good_callable_hessian(self):
        # Does not raise AssertionError
        pt.assert_hessian_at(lambda x: 2., 1.02,
                             function=lambda x: x**2.)

    def test_assert_good_hessian_value(self):
        # Does not raise AssertionError
        pt.assert_hessian_at(2., 1.02,
                             function=lambda x: x**2.)

    def test_assert_good_hessian_from_gradient(self):
        # Does not raise AssertionError
        pt.assert_hessian_at(2., 1.02,
                             gradient=lambda x: 2.*x)

    def test_assert_bad_callable_hessian(self):
        with pytest.raises(AssertionError):
            pt.assert_hessian_at(lambda x: x+2., 1.02,
                                 function=lambda x: x**2.)

    def test_assert_bad_hessian_value(self):
        with pytest.raises(AssertionError):
            pt.assert_hessian_at(32.12, 1.02,
                                 function=lambda x: x**2.)


class TestIntegralAssertions:

    def test_assert_integral_until_good_scalar(self):
        # Does not raise AssertionError
        pt.assert_integral_until(lambda x: np.exp(2.*x)/2.,
                                 lambda x: np.exp(2.*x), 0.)

    def test_assert_integral_until_good_array(self):
        # Does not raise AssertionError
        pt.assert_integral_until(lambda x: np.exp(2.*x)/2.,
                                 lambda x: np.exp(2.*x), np.array([-1., 0., 1.]))

    def test_assert_integral_until_bad_scalar(self):
        with pytest.raises(AssertionError):
            pt.assert_integral_until(lambda x: np.exp(2.*x),
                                     lambda x: np.exp(2.*x), 0.)

    def test_assert_integral_until_bad_array(self):
        with pytest.raises(AssertionError):
            pt.assert_integral_until(lambda x: np.exp(2.*x),
                                     lambda x: np.exp(2.*x), np.array([-1., 0., 1.]))
