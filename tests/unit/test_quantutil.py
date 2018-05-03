import numpy as np
import pytest

from scipy.stats import norm
import scipy.integrate as integrate
from scipy.optimize import brute

import pymaat.testing as pt
import pymaat.quantutil as qutil
from pymaat.nputil import diag_view
from pymaat.mathutil import dlogistic, ddlogistic



def test_check_value_inclusive_bounds():
    with pytest.raises(ValueError):
        qutil._check_value(0., [0., 1.])
    with pytest.raises(ValueError):
        qutil._check_value(1., [0., 1.])


@pytest.mark.parametrize("wrong", [np.nan, np.inf])
def test_check_value_invalid(wrong):
    with pytest.raises(ValueError):
        qutil._check_value(wrong, [0., 1.])


def test_check_value_ok():
    qutil._check_value(0.5, [0., 1.])


@pytest.mark.parametrize("wrong", [np.nan, np.inf, 1.01])
def test_check_proba_invalid(wrong):
    with pytest.raises(ValueError):
        qutil._check_proba(np.array([0.5, wrong]))


def test_check_proba_strictly_positive():
    with pytest.raises(ValueError):
        qutil._check_proba(np.array([1., 0.]))


def test_check_proba_ok():
    qutil._check_proba(np.array([0.5, 0.5]))


@pytest.mark.parametrize("wrong", [np.nan, np.inf, 1.01])
def test_check_trans_invalid(wrong):
    with pytest.raises(ValueError):
        qutil._check_trans(np.array([[wrong]]))


def test_check_trans_ok():
    trans = np.array([[0.3, 0.7], [1., 0.]])  # Need not be strictly positive
    qutil._check_trans(trans)  # defaults to ax = (1,)
    qutil._check_trans(trans, ax=(1,))


def test_match_shape_invalid():
    a = np.ones((3, 1))
    b = np.ones((3,))
    with pytest.raises(ValueError):
        qutil._check_match_shape(a, b, a)


def test_match_shape_ok():
    a = np.ones((3, 1))
    b = np.ones((3, 1))
    qutil._check_match_shape(a, b, a)


def test_check_dist_invalid():
    with pytest.raises(ValueError):
        qutil._check_dist(-1.)


def test_check_dist_ok():
    qutil._check_dist(0.5)

class TestQuantization1D():

    def test_init(self):
        # Value must be between -np.inf and inf by default
        value = np.array([1., 2., 3., 4.])
        # Proba must sum to one and be strictly positive
        proba = np.array([0.25, 0.25, 0.3, 0.2])
        # Rows of transition proba must sum to one and be positive
        trans = np.array([[0.25, 0.25, 0.25, 0.25],
                          [0.25, 0.25, 0.5, 0.]])
        # Distortion must be positive
        dist = 1.
        # Does not raise value error
        q = qutil.Quantization1D(value, proba, trans, dist)
        # Checks
        pt.assert_almost_equal(q.value, value)
        pt.assert_almost_equal(q.probability, proba)
        pt.assert_almost_equal(q.transition_probability, trans)
        assert q.distortion == dist

    def test_check_value(self):
        value = np.array([0.5, 1., 2.5])
        proba = np.array([0.3, 0.6, 0.1])
        with pytest.raises(ValueError):
            q = qutil.Quantization1D(value, proba, bounds=[0., 2.])

    def test_check_proba(self):
        value = np.array([0.5, 1., 2.5])
        proba = np.array([0.3, 0.7, 0.1])
        with pytest.raises(ValueError):
            q = qutil.Quantization1D(value, proba)

    def test_check_trans(self):
        value = np.array([0.5, 1., 2.5])
        proba = np.array([0.3, 0.6, 0.1])
        trans = np.array([[0.2, 0.8, 0.], [0.3, 0.8, 1.5]])
        with pytest.raises(ValueError):
            q = qutil.Quantization1D(value, proba, trans)

    def test_check_distortion(self):
        value = np.array([0.5, 1., 2.5])
        proba = np.array([0.3, 0.6, 0.1])
        trans = np.array([[0.2, 0.8, 0.], [0.3, 0.7, 0.]])
        with pytest.raises(ValueError):
            q = qutil.Quantization1D(value, proba, trans, -0.05)

    def test_voronoi(self):
        value = np.array([0.5, 1.])
        proba = np.array([0.3, 0.7])
        q = qutil.Quantization1D(value, proba, bounds=[0., 2.])
        pt.assert_almost_equal(q.voronoi, np.array([0., 0.75, 2.]))

    def test_bounds_default_to_pm_inf(self):
        value = np.array([0.5, 1.])
        proba = np.array([0.3, 0.7])
        q = qutil.Quantization1D(value, proba)
        pt.assert_almost_equal(q.voronoi, np.array([-np.inf, 0.75, np.inf]))

    def test_quantize_outisde(self):
        value = np.array([0.5, 1.])
        proba = np.array([0.3, 0.7])
        q = qutil.Quantization1D(value, proba, bounds=[0., 2.])
        with pytest.raises(ValueError):
            q.quantize(3.)
        with pytest.raises(ValueError):
            q.quantize(-1.)

    def test_quantize(self):
        value = np.array([0.5, 1., 1.5])
        proba = np.array([0.3, 0.6, 0.1])
        q = qutil.Quantization1D(value, proba, bounds=[0., 2.])
        expected = np.array([[0.5, 0.5, 1.], [1.5, 0.5, 1.]])
        quantized = q.quantize(
            np.array([[0.45, 0.55, 1.20], [1.99, 0.01, 0.92]]))
        pt.assert_almost_equal(expected, quantized)


class _QuantizerFactory1DStub:

    def __init__(self, prev_proba):
        self.prev_proba = prev_proba

    def make(self, value):
        return _Quantizer1DStub(value, self.prev_proba)


class _Quantizer1DStub(qutil.AbstractQuantizer1D):

    """
        Gaussian quantizer implementation
    """

    def __init__(self, value, prev_proba):
        super().__init__(value, prev_proba)
        self._set_integral()
        self._set_delta()

    def _set_integral(self):
        _s = self.prev_proba.shape + (self.size,)
        I0 = norm.cdf(self.voronoi)
        I1 = -norm.pdf(self.voronoi)
        I2 = np.empty_like(I1)
        I2[..., 1:-1] = norm.cdf(self.voronoi[..., 1:-1]) \
            - norm.pdf(self.voronoi[..., 1:-1])*self.voronoi[..., 1:-1]
        I2[..., 0] = 0.
        I2[..., -1] = 1.
        self._integral = [
            np.broadcast_to(np.diff(i, axis=-1), _s)
            for i in [I0, I1, I2]]

    def _set_delta(self):
        _s = self.prev_proba.shape + (self.size+1,)
        D0 = 0.5 * norm.pdf(self.voronoi)
        D1 = np.empty_like(D0)
        D1[..., 1:-1] = D0[..., 1:-1] * self.voronoi[..., 1:-1]
        D1[..., 0] = 0.
        D1[..., -1] = 0.
        self._delta = [np.broadcast_to(d, _s) for d in [D0, D1]]


class TestAbstractQuantizer1D:

    @pytest.fixture(params=[(1,), (4, 2), (3, 3), (1, 10)],
                    ids=['prev_shape(1)',
                         'prev_shape(4,2)',
                         'prev_shape(3,3)',
                         'prev_shape(1,10)'
                         ])
    def prev_shape(self, request):
        return request.param

    @pytest.fixture
    def prev_proba(self, prev_shape):
        p = np.random.uniform(
            low=0.05,
            high=1,
            size=prev_shape)
        p /= np.sum(p)  # Normalize
        return p

    @pytest.fixture
    def factory(self, prev_proba):
        return _QuantizerFactory1DStub(prev_proba)

    @pytest.fixture(params=[1, 2, 3, 5],
                    ids=['size(1)',
                         'size(2)',
                         'size(3)',
                         'size(5)'])
    def size(self, request):
        return request.param

    @pytest.fixture
    def value(self, size):
        np.random.seed(907831247)
        value = np.random.normal(size=size)
        value.sort()
        return value

    ###############################
    # Testing test implementation #
    ###############################

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_integral(self, factory, size, value, order):
        if order == 0:
            def integrand(z): return norm.pdf(z)
        elif order == 1:
            def integrand(z): return z*norm.pdf(z)
        elif order == 2:
            def integrand(z): return z**2.*norm.pdf(z)
        else:
            assert False  # should never happen...

        def _quantized_integral(value, order=None):
            def do(low, high):
                return integrate.quad(integrand, low, high)[0]
            vor = qutil.voronoi_1d(np.ravel(value))
            out = np.empty((value.size,))
            for j in range(value.size):
                out[j] = do(vor[j], vor[j+1])
            return out
        # Expected
        expected = _quantized_integral(value)
        expected.shape = (1,)*factory.prev_proba.ndim + (size,)
        expected = np.broadcast_to(expected,
                                   factory.prev_proba.shape + (size,))
        # Value
        quantizer = factory.make(value)
        value = quantizer._integral[order]
        # Assertion
        pt.assert_almost_equal(value, expected)

    @pytest.mark.parametrize("order", [0, 1])
    def test_delta(self, factory, value, order):
        qutil._assert_valid_delta_at(factory, value, order)

    #########
    # Tests #
    #########

    def test_bounds_default_to_pm_inf(self, factory):
        assert factory.make(np.array([1.])).bounds == [-np.inf, np.inf]

    def test_multidim_value_raises_value_error(self, factory):
        with pytest.raises(ValueError):
            factory.make(np.array([[1.]]))

    def test_size(self, factory, value):
        q = factory.make(value)
        assert q.size == value.size

    def test_infinite_value_raises_value_error(self, factory):
        value = np.array([0.25, 0.25, 0.5, np.inf])
        with pytest.raises(ValueError):
            factory.make(value)

    def test_nan_value_raises_value_error(self, factory):
        value = np.array([0.25, 0.25, 0.5, np.nan])
        with pytest.raises(ValueError):
            factory.make(value)

    def test_non_increasing_value_raises_value_error(self, factory):
        value = np.array([0.25, 0.2, 0.5, 1.])
        with pytest.raises(ValueError):
            factory.make(value)

    def test_value_is_broadcast_ready(self, prev_shape, factory):
        value = np.array([-1., 1.])
        q = factory.make(value)
        expected = value.copy()
        expected.shape = (1,)*len(prev_shape) + (2,)
        pt.assert_almost_equal(q.value, expected)

    def test_voronoi_is_broadcast_ready(self, prev_shape, factory):
        value = np.array([-1., 1.])
        q = factory.make(value)
        expected = np.array([-np.inf, 0., np.inf])
        expected.shape = (1,)*len(prev_shape) + (3,)
        pt.assert_almost_equal(q.voronoi, expected)

    def test_3D_previous_not_supported(self):
        with pytest.raises(ValueError):
            _Quantizer1DStub(np.array([1.]),
                             np.array([[[1.]]]))

    @pytest.fixture
    def quantizer(self, factory, value):
        return factory.make(value)

    def test_probability_shape(self, size, quantizer):
        assert quantizer.probability.shape == (size,)

    def test_probability_sums_to_one_and_strictly_positive(
            self, size, quantizer):
        pt.assert_greater(quantizer.probability, np.zeros((size,)))
        assert np.isclose(np.sum(quantizer.probability), 1.)

    def test_trans_proba_shape(self, factory, quantizer):
        assert (quantizer.transition_probability.shape
            == factory.prev_proba.shape + (quantizer.size,))

    def test_trans_proba_sums_to_one_and_non_negative(
            self, factory, quantizer):
        # Non negative...
        _s = factory.prev_proba.shape + (quantizer.size,)
        pt.assert_greater_equal(quantizer.transition_probability,
                                np.zeros(_s))
        # Sums to one...
        marginal_probability = np.sum(quantizer.transition_probability,
                                      axis=-1)
        pt.assert_almost_equal(marginal_probability,
                               np.ones(factory.prev_proba.shape))

    def test_trans_proba(self, quantizer):
        pt.assert_almost_equal(quantizer.transition_probability,
                               quantizer._integral[0])

    def test_proba_and_trans_proba(self, factory, quantizer):
        value = quantizer.probability
        it = np.nditer(factory.prev_proba, flags=['multi_index'])
        expected = np.zeros_like(value)
        while not it.finished:
            expected += factory.prev_proba[it.multi_index] \
                * quantizer.transition_probability[it.multi_index]
            it.iternext()
        pt.assert_almost_equal(value, expected)

    def test_distortion_elements(self, factory, quantizer):
        def _numerical_distortion(value):
            def do(low, high, value):
                def f(z): return (z-value)**2.*norm.pdf(z)
                return integrate.quad(f, low, high)[0]
            # Perform integration for each value
            vor = qutil.voronoi_1d(np.ravel(value))
            out = np.empty((value.size,))
            for j in range(value.size):
                out[j] = do(vor[j], vor[j+1], value[j])
            return out
        expected = _numerical_distortion(np.ravel(quantizer.value))
        expected.shape = (1,)*factory.prev_proba.ndim \
            + (quantizer.size,)
        expected = np.broadcast_to(expected,
                                   factory.prev_proba.shape + (quantizer.size,))
        value = quantizer._distortion_elements
        pt.assert_almost_equal(value, expected, rtol=1e-6)

    def test_distortion(self, factory, quantizer):
        # See below for _distortion_elements test
        distortion = np.sum(quantizer._distortion_elements, axis=-1)
        it = np.nditer(factory.prev_proba, flags=['multi_index'])
        expected = 0.
        while not it.finished:
            expected += np.sum(factory.prev_proba[it.multi_index]
                               * distortion[it.multi_index])
            it.iternext()
        pt.assert_almost_equal(expected, quantizer.distortion)

    def test_gradient(self, factory, quantizer):
        pt.assert_gradient_at(quantizer.gradient,
                              np.ravel(quantizer.value),
                              function=lambda x: factory.make(x).distortion)

    def test_hessian(self, factory, quantizer):
        pt.assert_hessian_at(quantizer.hessian,
                             np.ravel(quantizer.value),
                             gradient=lambda x: factory.make(x).gradient)

    def test_hessian_is_symmetric(self, quantizer):
        pt.assert_almost_equal(
            quantizer.hessian, np.transpose(quantizer.hessian))


class TestOptimFactory1D:

    @pytest.fixture(params=[(2,), (5,), (10,), (100,)],
                    ids=['size(2)', 'size(5)', 'size(10)', 'size(100)'])
    def shape(self, request):
        return request.param

    @pytest.fixture
    def factory(self):
        return _QuantizerFactory1DStub(np.array([1.]))

    @pytest.fixture(params=[(-1., 1.), (2., 10.), (-10., -8.)],
                    ids=['bounds(-1,1)', 'bounds(2,10)', 'bounds(-10,8)'])
    def search_bounds(self, request):
        return request.param

    @pytest.fixture
    def optim_factory(self, factory, search_bounds):
        return qutil._OptimFactory1D(factory, search_bounds)

    def test_factory(self, factory, optim_factory):
        assert optim_factory.factory is factory

    def test_search_bounds(self, search_bounds, optim_factory):
        assert optim_factory.search_bounds == search_bounds

    def test_make(self, optim_factory, random_normal):
        q = optim_factory.make(random_normal)
        assert isinstance(q, qutil._OptimQuantizer1D)

    def test_in_cache(self, optim_factory, random_normal):
        x = random_normal
        x.sort()
        q1 = optim_factory.make(x)
        q2 = optim_factory.make(x)
        assert q1 is q2

    def test_not_in_cache(self, optim_factory, random_normal):
        x = random_normal
        x.sort()
        q1 = optim_factory.make(x)
        x = x.copy()  # Make copy
        q2 = optim_factory.make(x)
        assert q1 is not q2

    def test_clear_cache(self, optim_factory, random_normal):
        x = random_normal
        x.sort()
        q1 = optim_factory.make(x)
        optim_factory.clear()
        q2 = optim_factory.make(x)
        assert q1 is not q2

    def test_clear_cache_impl(self, optim_factory, random_normal):
        x = random_normal
        x.sort()
        q1 = optim_factory.make(x)
        assert optim_factory._cache._Cache__currsize == 1
        optim_factory.clear()
        assert optim_factory._cache._Cache__currsize == 0


class TestOptimQuantizer1D:

    @pytest.fixture(params=[1,2,5,10],
                    ids=['size(1)', 'size(2)', 'size(5)', 'size(10)'])
    def size(self, request):
        return request.param

    @pytest.fixture
    def shape(self, size):
        return (size,)

    @pytest.fixture(params=[(-1., 1.), (-0.5, 0.75), (0.25, 0.5)])
    def search_bounds(self, request):
        return request.param

    @pytest.fixture
    def optim_factory(self, search_bounds):
        f = _QuantizerFactory1DStub(np.array([1.]))
        return qutil._OptimFactory1D(f, search_bounds)

    @pytest.fixture
    def optim_quantizer(self, optim_factory, random_normal):
        return optim_factory.make(random_normal)

    def test_value_is_in_space(self, size, optim_factory, optim_quantizer):
        pt.assert_greater(optim_quantizer.value,
                          optim_factory.search_bounds[0], shape='broad')
        pt.assert_less(optim_quantizer.value,
                       optim_factory.search_bounds[1], shape='broad')
        if size > 1:
            pt.assert_greater(np.diff(optim_quantizer.value),
                              0.0, shape='broad')

    def test_distortion(self, optim_quantizer):
        pt.assert_almost_equal(optim_quantizer.distortion,
                               optim_quantizer.quantizer.distortion)

    def test_jacobian(self, optim_factory, optim_quantizer):
        def f(x): return optim_factory.make(x).value
        pt.assert_jacobian_at(
            optim_quantizer.jacobian,
            optim_quantizer.x,
            function=f)

    def test_gradient(self, optim_factory, optim_quantizer):
        def f(x): return optim_factory.make(x).distortion
        pt.assert_gradient_at(
            optim_quantizer.gradient,
            optim_quantizer.x,
            function=f
        )

    def test_hessian(self, optim_factory, optim_quantizer):
        def f(x): return optim_factory.make(x).gradient
        pt.assert_hessian_at(
            optim_quantizer.hessian,
            optim_quantizer.x,
            gradient=f
        )


    def test_theta0(self, optim_quantizer):
        expected = optim_quantizer.x[0] - optim_quantizer.OFFSET
        assert np.isclose(expected, optim_quantizer.theta[0])

    def test_thetam(self, size, optim_quantizer):
        off = optim_quantizer.OFFSET
        x = optim_quantizer.x
        for i in range(1,size):
            expected = x[0] - off
            for ii in range(1,i+1):
                expected += 2.*off/(size-1.)*np.exp(x[ii])
            assert np.isclose(expected, optim_quantizer.theta[i])

    def test_jacobian_analytic(self, size, optim_quantizer):
        expected = np.zeros((size,size))
        sb = optim_quantizer.search_bounds
        span = sb[1] - sb[0]
        k1 = optim_quantizer._kappa1
        theta = optim_quantizer.theta
        for m in range(size):
            for n in range(size):
                expected[m,n] = (
                        np.double(m>=n)*span*k1[n]*dlogistic(theta[m]))
        pt.assert_almost_equal(expected, optim_quantizer.jacobian)


    def test_hessian_analytic(self, size, optim_quantizer):
        sb = optim_quantizer.search_bounds
        span = sb[1] - sb[0]
        jac = optim_quantizer.jacobian
        hess = optim_quantizer.quantizer.hessian
        grad = optim_quantizer.quantizer.gradient
        theta = optim_quantizer.theta
        off = optim_quantizer.OFFSET
        x = optim_quantizer.x
        kappa0 = np.zeros((size,))
        for i in range(1,size):
            kappa0[i] = 2.*off/(size-1.)*np.exp(x[i])
        kappa1 = kappa0.copy(); kappa1[0] = 1.
        df = np.diag(kappa0)
        ddf = kappa1[np.newaxis,:] * kappa1[:,np.newaxis]
        expected = jac.T.dot(hess).dot(jac)
        for (i, g) in enumerate(grad):
            expected[:i+1, :i+1] += \
                    span*g*df[:i+1,:i+1]*dlogistic(theta[i])
            expected[:i+1, :i+1] += \
                    span*g*ddf[:i+1,:i+1]*ddlogistic(theta[i])
        pt.assert_almost_equal(expected, optim_quantizer.hessian)


class TestOptimizer1D:

    @pytest.fixture(scope='class')
    def factory(self):
        return _QuantizerFactory1DStub(np.array([1.]))

    @pytest.fixture(scope='class')
    def size(self):
        return 2

    @pytest.fixture(scope='class', params=[True, False])
    def robust(self, request):
        return request.param

    @pytest.fixture(scope='class', params=[1000])
    def maxiter(self, request):
        return request.param

    @pytest.fixture(scope='class', params=[0.5])
    def scale(self, request):
        return request.param

    @pytest.fixture(scope='class', params=[10,100])
    def niter(self, request):
        return request.param

    @pytest.fixture(scope='class', params=[5,50])
    def niter_success(self, request):
        return request.param

    @pytest.fixture(scope='class', params=[True,False])
    def verbose(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def optimizer(self, factory, size,
            robust,
            maxiter,
            scale,
            niter,
            niter_success,
            verbose):
        return qutil.Optimizer1D(factory, size,
                robust=robust,
                maxiter=maxiter,
                scale=scale,
                niter=niter,
                niter_success=niter_success,
                verbose=verbose)

    @pytest.fixture(scope='class')
    def solution(self, factory):
        ranges = ((-1.,0.),(0.,1.))
        sol = brute(lambda x: factory.make(x).distortion, ranges)
        return factory.make(sol)

    @pytest.fixture(scope='class', params=[(-1.,1.),])
    def search_bounds(self, request):
        return request.param

    def test_robust(self, robust, optimizer):
        assert optimizer.robust == robust

    def test_maxiter(self, maxiter, optimizer):
        assert optimizer.maxiter == maxiter

    def test_scale(self, size, scale, optimizer):
        if scale is None:  # Default...
            assert np.isclose(optimizer.scale,
                    optimizer._func(np.zeros((size,))))
        else:
            assert np.isclose(optimizer.scale, scale)

    def test_niter(self, niter, optimizer):
        assert optimizer.niter == niter

    def test_niter_success(self, niter_success, optimizer):
        assert optimizer.niter_success == niter_success

    def test_verbose(self, verbose, optimizer):
        assert optimizer.verbose == verbose

    def test_initialize(self, optimizer):
        sb = (-5., 5.)
        optimizer._initialize(sb)
        assert optimizer.factory.search_bounds == sb

    def test_default_scale(self, factory, size):
        opt = qutil.Optimizer1D(factory, size)
        opt._initialize((-1.,1.))
        assert opt.scale > 0.
        assert opt.scale < 3.

    def test_optimize(self, search_bounds, optimizer, solution):
        opt = optimizer.optimize(search_bounds)
        assert isinstance(opt,  _Quantizer1DStub)
        pt.assert_almost_equal(opt.value, solution.value, rtol=1e-4)

    def test_optimize_raises_when_tight(self, factory):
        optimizer = qutil.Optimizer1D(factory, 2, verbose=True)
        with pytest.raises(qutil.ExpandBounds):
            opt = optimizer.optimize((-1.,-0.5))

    def test_optimize_raises_when_null_proba(self, factory, solution):
        optimizer = qutil.Optimizer1D(factory, 2, verbose=True)
        with pytest.raises(qutil.ShrinkBounds):
            opt = optimizer.optimize((-100.,100.))

    # Test internals...

    def test_free_memory_after_optimize(self, search_bounds, optimizer):
        optimizer.optimize(search_bounds)
        assert optimizer.factory._cache._Cache__currsize == 0

    def test_trust_ncg_options(self, optimizer):
        options = optimizer._format_trust_ncg_options()
        assert options['maxiter'] == optimizer.maxiter
        assert np.isclose(options['gtol'], optimizer.scale * 1e-6)

    def test_accept_test_ok(self, size, search_bounds, optimizer):
        optimizer._initialize(search_bounds)
        assert optimizer._accept_test(
                f_new=0.0001,
                f_old=0.1,
                x_new=np.zeros((size,)))

    def test_reject_insignificant(self, size, search_bounds, optimizer):
        optimizer._initialize(search_bounds)
        assert not optimizer._accept_test(
                f_new=0.1*(1-1e-12),
                f_old=0.1,
                x_new=np.zeros((size,)))

    def test_reject_escaping_left(self, size, search_bounds, optimizer):
        optimizer._initialize(search_bounds)
        x = np.zeros((size,))
        x[0] = -100.
        x[1] = 100.
        assert not optimizer._accept_test(
                f_new=0.001,
                f_old=0.1,
                x_new=x)

    def test_reject_escaping_right(self, size, search_bounds, optimizer):
        optimizer._initialize(search_bounds)
        x = np.zeros((size,))
        x[1] = 100.
        assert not optimizer._accept_test(
                f_new=0.001,
                f_old=0.1,
                x_new=x)

    def test_reject_non_increasing(self, size, search_bounds, optimizer):
        optimizer._initialize(search_bounds)
        x = np.zeros((size,))
        x[1] = -100.
        assert not optimizer._accept_test(
                f_new=0.001,
                f_old=0.1,
                x_new=x)

class CoreStub(qutil.AbstractCore):

    def _make_first_quant(self):
        return qutil.Quantization1D(np.array([0.]), np.array([1.]))

    def _get_all_shapes(self):
        return np.array([100,]*10)

    def _one_step_optimize(self, shape, previous):
        f = _QuantizerFactory1DStub(previous.probability)
        optimizer = qutil.Optimizer1D(f,  shape,
                robust=False,
                verbose=self.verbose)
        raw_quant = optimizer.optimize((-5.,5.))
        return qutil.Quantization1D(value=np.ravel(raw_quant.value),
                probability=np.ravel(raw_quant.probability))

class TestAbstractCore:

    @pytest.fixture(scope='class')
    def core(self):
        return CoreStub(verbose=True).optimize()

    @pytest.fixture(scope='class')
    def size(self, core):
        return core._get_all_shapes().size+1

    def test_optimize_returns_core(self, core):
        assert isinstance(core, CoreStub)

    def test_len_all_quant(self, core, size):
        assert isinstance(core.all_quant, list)
        assert(len(core.all_quant) == size)

    def test_all_quant_elements(self, core, size):
        for i in range(size):
            assert isinstance(core.all_quant[i], qutil.Quantization1D)


class TestVoronoi1D:

    @pytest.fixture(params=[1, 2])
    def ndim(self, request):
        return request.param

    @pytest.fixture(params=[0, 1])
    def axis(self, request, ndim):
        return min(request.param, ndim-1)

    @pytest.fixture(params=[
        (np.array([1.]), np.array([])),
        (np.array([1., 2., 3.]), np.array([1.5, 2.5])),
        (np.array([1., 3., 4., 10.]), np.array([2., 3.5, 7.])),
    ])
    def setup(self, request):
        """
        (quantizer, voronoi (without bounds))
        """
        return request.param

    @pytest.fixture(params=[(-np.inf, np.inf), (0.0, np.inf)],
                    ids=['unbounded', 'positive'])
    def bounds(self, request):
        return request.param

    @pytest.fixture
    def quantizer(self, setup, ndim, axis):
        values = setup[0]
        size = values.size
        shape = np.full((ndim,), 1, dtype=np.int)
        shape[axis] = size
        return np.reshape(values, tuple(shape))

    @pytest.fixture
    def voronoi_with_bounds(self, setup, ndim, axis, bounds):
        values = setup[1]
        size = values.size
        if size > 0:
            # Append bounds to both ends
            out = np.empty((size+2,))
            out[0] = bounds[0]
            out[-1] = bounds[-1]
            out[1:-1] = values
        else:
            out = np.array(bounds)
        shape = np.full((ndim,), 1, dtype=np.int)
        shape[axis] = size
        shape[axis] += 2
        return np.reshape(out, tuple(shape))

    def test_with_bounds(self,
                         axis, quantizer, bounds, voronoi_with_bounds):
        v = qutil.voronoi_1d(quantizer, axis=axis,
                             lb=bounds[0], ub=bounds[-1])
        pt.assert_almost_equal(v, voronoi_with_bounds)

    def test_supports_non_array(self):
        v = qutil.voronoi_1d([1., 2., 3.])
        pt.assert_almost_equal(v, np.array([-np.inf, 1.5, 2.5, np.inf]))

    def test_vector_when_N_is_zero(self):
        with pytest.raises(ValueError):
            qutil.voronoi_1d(np.array([]))

    def test_matrices_when_N_is_zero(self):
        with pytest.raises(ValueError):
            qutil.voronoi_1d(np.array([[]]), axis=0)
        with pytest.raises(ValueError):
            qutil.voronoi_1d(np.array([[]]), axis=1)

    # Inverse

    def test_inv_with_bounds(self, axis, quantizer, voronoi_with_bounds):
        if axis == 0:
            first_quantizer = quantizer[0]
        elif axis == 1:
            first_quantizer = quantizer.T[0]
        q = qutil.inv_voronoi_1d(voronoi_with_bounds,
                                 first_quantizer=first_quantizer,
                                 axis=axis, with_bounds=True)
        pt.assert_almost_equal(q, quantizer)

    def test_inv_supports_non_array(self):
        q = qutil.inv_voronoi_1d([-np.inf, 1.5, 2.5, np.inf],
                                 first_quantizer=1.)
        pt.assert_almost_equal(q, np.array([1., 2., 3.]))

    # TODO fails when empty?
