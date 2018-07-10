import numpy as np
import pytest

from scipy.stats import norm
from scipy.linalg import norm as linalg_norm
import scipy.integrate as integrate
from scipy.optimize import brute

import pymaat.testing as pt
import pymaat.quantutil as qutil
from pymaat.nputil import diag_view
from pymaat.nputil import printoptions
from pymaat.mathutil import dlogistic, ddlogistic

#########
# Stubs #
#########

class CoreStub(qutil.AbstractCore):

    def __init__(self):
        super().__init__(
                np.array([100,]*10),
                self._make_first_quant())

    def _make_first_quant(self):
        out = qutil.Quantization1D(np.array([0.]))
        out.set_probability(np.array([1.]))
        return out

    def _one_step_optimize(self, shape, previous):
        f = _QuantizerFactory1DStub(previous.probability)
        optimizer = qutil.Optimizer1D(f,  shape,
                robust=False,
                verbose=self.verbose)
        result = optimizer.optimize((-5.,5.))
        quantizer = np.ravel(result.value)
        out = QuantizationStub(quantizer[:,np.newaxis], None)
        out.set_probability(np.ravel(result.probability))
        return out

class QuantizationStub(qutil.AbstractQuantization):

    def _digitize(self, values):
        nquery = values.shape[0]
        shape_dim = len(self.shape)
        idx = np.empty((nquery,shape_dim), np.int)
        for (i,v) in zip(idx, values):
            it = np.nditer(self.quantizer, flags=['multi_index'])
            it.remove_axis(shape_dim)
            _distance = np.inf
            _idx = (0,) * shape_dim
            while not it.finished:
               q = self.quantizer[it.multi_index]
               d = (q-v)/self.weight
               d = linalg_norm(d)
               if d<_distance:
                   _distance = d
                   _idx = it.multi_index
               it.iternext()
            i[:] = _idx
        out = []
        for n in range(shape_dim):
            out.append(idx[:,n])
        return tuple(out)

class _QuantizerFactory1DStub:

    def __init__(self, prev_proba):
        self.prev_proba = prev_proba

    def make(self, value, norm=1.):
        return _Quantizer1DStub(value, self.prev_proba, norm)


class _Quantizer1DStub(qutil.AbstractQuantizer1D):

    """
        Gaussian quantizer implementation
    """

    def __init__(self, value, prev_proba, norm=1.):
        super().__init__(value, prev_proba, norm)
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


#########
# Tests #
#########


class TestAbstactQuantizationSimple1DCase:

    @pytest.fixture
    def quant(self):
        return QuantizationStub(np.array([[1.2],[2.4],[3.6]]), None)

    @pytest.fixture
    def bquant(self):
        return QuantizationStub(
                np.array([[1.2],[2.4],[3.6]]),
                np.array([[0.],[4.]]),
                )

    @pytest.fixture
    def wquant(self):
        out = QuantizationStub(
                np.array([[1.2],[2.4],[3.6]]),
                np.array([[0.],[4.]]),)
        out._set_weight(np.array([2.]))
        return out

    #################
    # Get quantized #
    #################

    def test__quantize(self, quant):
        idx = (np.array([0,2,1,1]),)
        out = quant._quantize(idx)
        pt.assert_almost_equal(out, np.array([[1.2],[3.6],[2.4],[2.4]]))

    def test_get_quantized(self, quant):
        out = quant.get_quantized(
                np.array([[1.21], [3.75], [2.5], [2.25]]))
        pt.assert_almost_equal(out, np.array([[1.2],[3.6],[2.4],[2.4]]))

    def test_get_quantized_with_bounds(self, bquant):
        out = bquant.get_quantized(
                np.array([[0.], [4.], [2.5], [-1.]]))
        pt.assert_almost_equal(out,
                np.array([[np.nan],[np.nan],[2.4],[np.nan]]),
                invalid='allow')

    ############
    # Distance #
    ############

    def test__distance(self, quant):
        at = np.array([[1.],[3.],[2.],[2.]])
        idx = (np.array([0,2,1,1]),)
        out = quant._distance(at, idx)
        expected = np.absolute(
                np.array([(1.-1.2),(3.-3.6),(2.-2.4),(2.-2.4)])
                )
        pt.assert_almost_equal(out, expected)

    def test_get_distance(self, quant):
        at = np.array([[1.],[3.],[2.],[2.]])
        out = quant.get_distance(at)
        expected = np.absolute(
                np.array([(1.-1.2),(3.-3.6),(2.-2.4),(2.-2.4)])
                )
        pt.assert_almost_equal(out, expected)

    def test_get_distance_with_bounds(self, bquant):
        at = np.array([[0.],[4.],[2.],[-1.]])
        out = bquant.get_distance(at)
        expected = np.absolute(
                np.array([np.nan,np.nan,(2.-2.4),np.nan])
                )
        pt.assert_almost_equal(out, expected, invalid='allow')

    def test_get_distance_weighted(self, wquant):
        at = np.array([[1.],[3.],[2.],[2.]])
        out = wquant.get_distance(at)
        expected = np.absolute(
                np.array([(1.-1.2),(3.-3.6),(2.-2.4),(2.-2.4)])*2.
                )
        pt.assert_almost_equal(out, expected)

    ###############
    # Probability #
    ###############

    @pytest.mark.parametrize("strict", [True, False])
    def test__estimate_and_set_probability(self, quant, strict):
        idx = (np.array([0,2,1,1]),)
        quant._estimate_and_set_probability(idx, strict)
        expected = np.array([1.,2.,1.])/4.
        pt.assert_almost_equal(quant.probability, expected)

    def test__estimate_and_set_probability_raise_unobs(self, quant):
        idx = (np.array([2,1,1]),)
        with pytest.raises(qutil.UnobservedState):
            # State 0 is unobserved!
            quant._estimate_and_set_probability(idx, True)

    def test_estimate_and_set_probability(self, quant):
        at = np.array([[1.],[3.4],[2.],[2.]])
        quant.estimate_and_set_probability(at)
        expected = np.array([1.,2.,1.])/4.
        pt.assert_almost_equal(quant.probability, expected)

    def test_estimate_and_set_probability_with_bounds(self, bquant):
        at = np.array([[0.],[3.4],[2.],[2.]])
        with pytest.raises(ValueError):
            bquant.estimate_and_set_probability(at)

    ##############
    # Distortion #
    ##############

    def test__estimate_and_set_distortion(self, quant):
        at = np.array([[1.],[3.],[2.],[2.]])
        idx = (np.array([0,2,1,1]),)
        quant._estimate_and_set_distortion(at, idx)
        out = quant.distortion
        expected = ((1.-1.2)**2.+(3.-3.6)**2.+(2.-2.4)**2.+(2.-2.4)**2.)/4.
        assert np.isclose(out, expected)

    def test_estimate_and_set_distortion(self, quant):
        at = np.array([[1.],[3.],[2.],[2.]])
        quant.estimate_and_set_distortion(at)
        out = quant.distortion
        expected = ((1.-1.2)**2.+(3.-3.6)**2.+(2.-2.4)**2.+(2.-2.4)**2.)/4.
        assert np.isclose(out, expected)

    def test_estimate_and_set_distortion_with_bounds(self, bquant):
        at = np.array([[0.],[3.],[2.],[2.]])
        with pytest.raises(ValueError):
            bquant.estimate_and_set_distortion(at)

    def test_estimate_and_set_distortion_weighted(self, wquant):
        at = np.array([[1.],[3.],[2.],[2.]])
        wquant.estimate_and_set_distortion(at)
        out = wquant.distortion
        expected = (
                ((1.-1.2)/2.)**2.
                +((3.-3.6)/2.)**2.
                +((2.-2.4)/2.)**2.
                +((2.-2.4)/2.)**2.)/4.

class TestAbstractQuantizationGeneral:

    @pytest.fixture(params = [(5,), (2,3), (2,2,2)])
    def shape(self, request):
        return request.param

    @pytest.fixture(params = [1,2,3,4])
    def ndim(self, request):
        return request.param

    @pytest.fixture
    def quantizer(self, shape, ndim):
        s = shape+(ndim,)
        out = np.arange(np.prod(s))
        out.shape = s
        out = np.array(out, np.float)
        return out

    @pytest.fixture
    def quant(self, quantizer):
        return QuantizationStub(quantizer, None)

    @pytest.fixture
    def bounds(self, quantizer, ndim):
        bounds = np.empty((2, ndim))
        for n in range(ndim):
            bounds[0,n] = np.amin(quantizer[...,n])-1.
            bounds[1,n] = np.amax(quantizer[...,n])+1.
        return bounds

    @pytest.fixture
    def bquant(self, quantizer, bounds):
        return QuantizationStub(quantizer, bounds)

    ########
    # Init #
    ########

    def test_init_ndim(self, quant, ndim):
        assert quant.ndim == ndim

    def test_init_shape(self, shape, quant):
        assert quant.shape == shape

    def test_init_size(self, quant, shape):
        assert quant.size == np.prod(shape)

    def test_init_bounds(self, bquant, bounds):
        pt.assert_almost_equal(bquant.bounds, bounds)

    def test_init_quantizer(self, quant, quantizer):
        pt.assert_almost_equal(quant.quantizer, quantizer)

    def test_init_proba_dist(self, quant):
        assert quant.probability is None
        assert quant.distortion is None

    # Bounds setter

    def test_default_bounds(self, quant):
        expected = np.empty((2, quant.ndim))
        expected[0] = -np.inf
        expected[1] = np.inf
        pt.assert_equal(quant.bounds, expected)

    def test_wrong_bounds_shape_raises_value_error(self, ndim, quant):
        bounds = np.empty((3, ndim+4))
        bounds[0] = 0.
        bounds[1] = 1.
        with pytest.raises(ValueError):
            quant._set_bounds(bounds)

    def test_bounds_not_strictly_increasing_raises_value_error(
            self, ndim, quant):
        bounds = np.zeros((2, ndim))
        with pytest.raises(ValueError):
            quant._set_bounds(bounds)

    # Quantizer setter

    def test_wrong_quantizer_shape_raises_value_error(self, quant):
        with pytest.raises(ValueError):
            quant._set_quantizer(np.zeros(quant.shape))
        with pytest.raises(ValueError):
            quant._set_quantizer(np.zeros((quant.size,)))

    @pytest.mark.parametrize("wrong", [np.nan, np.inf])
    def test_invalid_quantizer_raises_value_error(
            self, quant, quantizer, wrong):
        quantizer[0] = wrong
        with pytest.raises(ValueError):
            quant._set_quantizer(quantizer)

    @pytest.mark.parametrize("upper", [0,1])
    def test_quantizer_at_bounds_raises_value_error(
            self, ndim, bquant, bounds, quantizer, upper):
        for n in range(ndim):
            q = quantizer.copy()
            q[...,n] = bounds[upper,n]
            with pytest.raises(ValueError):
                bquant._set_quantizer(q)

    def test_quantizer_below_bounds_raises_value_error(
            self, ndim, bquant, bounds, quantizer):
        for n in range(ndim):
            q = quantizer.copy()
            q[...,n] = bounds[0,n]-0.5
            with pytest.raises(ValueError):
                bquant._set_quantizer(q)

    def test_quantizer_above_bounds_raises_value_error(
            self, ndim, bquant, bounds, quantizer):
        for n in range(ndim):
            q = quantizer.copy()
            q[...,n] = bounds[1,n]+0.5
            with pytest.raises(ValueError):
                bquant._set_quantizer(q)

    # Weight setter

    def test_wrong_weight_size_raises_value_error(self, ndim, quant):
        quant._set_weight(np.ones((ndim,)))
        with pytest.raises(ValueError):
            quant._set_weight(np.ones((ndim+1,)))

    @pytest.mark.parametrize("wrong", [np.nan, np.inf])
    def test_invalid_weight_raises_value_error(self, ndim, quant, wrong):
        w = np.ones((ndim,))
        w[0] = wrong
        with pytest.raises(ValueError):
            quant._set_weight(w)

    def test_negative_weight_raises_value_error(self, ndim, quant):
        w = np.ones((ndim,))
        w[0] = -1.
        with pytest.raises(ValueError):
            quant._set_weight(w)

    def test__set_weight(self, ndim, quant):
        quant._set_weight(2.*np.ones((ndim,1,1)))
        pt.assert_almost_equal(quant.weight, 2.*np.ones((1,ndim)))

    # Others

    @pytest.mark.parametrize("method_name", [
                'estimate_and_set_probability',
                'estimate_and_set_distortion',
                'get_quantized',
                'get_distance'
                ])
    def test_query_at_wrong_shape_raises_value_error(
            self, ndim, quant, method_name):
        fcn = getattr(quant, method_name)
        with pytest.raises(ValueError):
            fcn(np.zeros((3,ndim,1)))

    #################
    # Get Quantized #
    #################

    def test_quantize_at_quantizer(self, quant):
        pt.assert_almost_equal(
                quant.get_quantized(quant._ravel()),
                quant._ravel())

    @pytest.mark.parametrize("upper", [0,1])
    def test_get_quantized_at_bounds_is_nan(
            self, ndim, bquant, bounds, upper):
        at = bquant._ravel()
        for n in range(ndim):
            at = bquant._ravel().copy()
            at[0,n] = bounds[upper,n]
            quantized = bquant.get_quantized(at)
            pt.assert_invalid(quantized[0])

    def test_get_quantized_below_bounds_is_nan(
            self, ndim, bquant, bounds):
        at = bquant._ravel()
        for n in range(ndim):
            at = bquant._ravel().copy()
            at[0,n] = bounds[0,n]-0.5
            quantized = bquant.get_quantized(at)
            pt.assert_invalid(quantized[0])
            pt.assert_valid(quantized[1:])

    def test_get_quantized_above_bounds_is_nan(
            self, ndim, bquant, bounds):
        at = bquant._ravel()
        for n in range(ndim):
            at = bquant._ravel().copy()
            at[0,n] = bounds[1,n]+0.5
            quantized = bquant.get_quantized(at)
            pt.assert_invalid(quantized[0])
            pt.assert_valid(quantized[1:])

    ################
    # Get distance #
    ################

    def test_get_distance_at_quantizer(self, quant):
        pt.assert_almost_equal(quant.get_distance(quant._ravel()),
                np.zeros((quant.size,)))

    @pytest.mark.parametrize("upper", [0,1])
    def test_get_distance_at_bounds_is_nan(
            self, ndim, bquant, bounds, upper):
        at = bquant._ravel()
        for n in range(ndim):
            at = bquant._ravel().copy()
            at[0,n] = bounds[upper,n]
            d = bquant.get_distance(at)
            pt.assert_invalid(d[0])

    def test_get_distance_below_bounds_is_nan(
            self, ndim, bquant, bounds):
        at = bquant._ravel()
        for n in range(ndim):
            at = bquant._ravel().copy()
            at[0,n] = bounds[0,n]-0.5
            d = bquant.get_distance(at)
            pt.assert_invalid(d[0])
            pt.assert_valid(d[1:])

    def test_get_distance_above_bounds_is_nan(
            self, ndim, bquant, bounds):
        at = bquant._ravel()
        for n in range(ndim):
            at = bquant._ravel().copy()
            at[0,n] = bounds[1,n]+0.5
            d = bquant.get_distance(at)
            pt.assert_invalid(d[0])
            pt.assert_valid(d[1:])


    ############
    # Estimate #
    ############

    def test_estimate_and_set_probability_at_quantizer(
            self, quant):
        quant.estimate_and_set_probability(quant._ravel())
        pt.assert_almost_equal(quant.probability,
                1./quant.size,
                shape='broad')

    def test_estimate_and_set_probability_raises_unobs_by_default(
            self, quant):
        with pytest.raises(qutil.UnobservedState):
            quant.estimate_and_set_probability(quant._ravel()[:-1,:])

    def test_estimate_and_set_probability_strict_flag(self, quant):
        quant.estimate_and_set_probability(quant._ravel()[:-1,:],
                strict=False)
        with pytest.raises(qutil.UnobservedState):
            quant.estimate_and_set_probability(quant._ravel()[:-1,:],
                    strict=True)

    def test_estimate_distortion_at_quantizer(self, quant):
        at = quant._ravel()
        quant.estimate_and_set_distortion(at)
        assert np.isclose(quant.distortion, 0.)

    def test_estimate_outside_raises_value_error(self, bounds, bquant):
        v = bquant._ravel()
        v[0,0] = bounds[0,0]
        with pytest.raises(ValueError):
            bquant.estimate_and_set_probability(v)
        with pytest.raises(ValueError):
            bquant.estimate_and_set_distortion(v)


    #############
    # Utilities #
    #############

    # To voronoi quantization...

    def test_to_voronoi(self, quant):
        vq = quant.to_voronoi()
        assert isinstance(vq, qutil.VoronoiQuantization)

    def test_to_voronoi_keeps_bounds(self, quant):
        vq = quant.to_voronoi()
        pt.assert_almost_equal(quant.bounds, vq.bounds)

    def test_to_voronoi_keeps_bounds(self, quant):
        vq = quant.to_voronoi()
        pt.assert_almost_equal(quant.weight, vq.weight)

    def test_to_voronoi_makes_copy(self, quant, quantizer):
        vq = quant.to_voronoi()
        vq.quantizer[0] = np.nan
        pt.assert_almost_equal(quant.quantizer, quantizer)

    def test_ravel(self, shape, ndim, quant, quantizer):
        s = shape+(ndim,)
        expected = np.arange(np.prod(s))
        expected.shape = (np.prod(shape),ndim)
        expected = np.array(expected, np.float)
        pt.assert_almost_equal(
                quant._ravel(),
                expected
                )

    # Probability setter...

    @pytest.fixture
    def probability(self, shape):
        return np.ones(shape)/np.prod(shape)

    def test_proba(self, probability, quant):
        quant.set_probability(probability)
        pt.assert_almost_equal(probability, quant.probability)

    def test_wrong_proba_shape_raises_value_error(self, shape, quant):
        probability = np.zeros(shape+(3,2))
        with pytest.raises(ValueError):
            quant.set_probability(probability)

    @pytest.mark.parametrize("wrong", [np.nan, np.inf])
    def test_invalid_proba_raises_value_error(
            self, probability, quant, wrong):
        probability[0] = wrong
        with pytest.raises(ValueError):
            quant.set_probability(probability)

    def test_proba_does_not_sum_to_one_raises_value_error(
            self, quant, probability):
        probability[0] *= 2.
        with pytest.raises(ValueError):
            quant.set_probability(probability)

    def test_proba_is_null_raises_value_error(self, quant, probability):
        probability[1] += probability[0]
        probability[0] = 0.
        with pytest.raises(ValueError):
            quant.set_probability(probability)

    # Distortion setter

    def test_set_distortion(self, quant):
        quant.set_distortion(1.)
        assert np.isclose(quant.distortion, 1.)

    @pytest.mark.parametrize("wrong", [-1., np.nan, np.array([1.,2.])])
    def test_set_invalid_distortion_raises_value_error(self, quant, wrong):
        with pytest.raises(ValueError):
            quant.set_distortion(wrong)


class TestVoronoiQuantization:

    def test_wrong_quantizer_shape_raises_value_error(self):
        quantizer = np.ones((1,2,3,))
        with pytest.raises(ValueError):
            qutil.VoronoiQuantization(quantizer)

    @pytest.fixture(params=[1,2,3,10,100])
    def nquant(self, request):
        return request.param

    @pytest.fixture(params=[1,2,3,4])
    def ndim(self, request):
        return request.param

    @pytest.fixture
    def quantizer(self, nquant, ndim):
        np.random.seed(907831247)
        return np.random.uniform(
                low=1., high=2., size=(nquant,ndim))

    @pytest.fixture
    def quant(self, quantizer):
        return qutil.VoronoiQuantization(quantizer)

    @pytest.fixture(params=[10,17,100])
    def queries(self, request, quant):
        np.random.seed(907831247)
        ntest = request.param
        return np.random.uniform(
                low=-1., high=4., size=(ntest,quant.ndim))

    @pytest.fixture
    def distance(self, quant, queries):
        d = np.empty((queries.shape[0],quant.size))
        for n in range(queries.shape[0]):
            for nn in range(quant.size):
                # Compute Euclidean distance
                d[n,nn] = np.sum((queries[n] - quant.quantizer[nn])**2)**0.5
        return d

    def test_digitize_returns_tuple(self, quant, queries):
        assert isinstance(quant._digitize(queries), tuple)

    def test_digitize(self, quant, queries, distance):
        expected = np.argmin(distance, axis=1)
        out = quant._digitize(queries)
        out = out[0]
        pt.assert_equal(out, expected)

    def test_get_distance(self, quant, queries, distance):
        distance = np.amin(distance, axis=1)
        pt.assert_almost_equal(quant.get_distance(queries), distance)

    # Some redundant tests...

    def test_get_quantized_at_quantizer(self, quantizer, quant):
        pt.assert_almost_equal(
                quant.get_quantized(quantizer),
                quantizer)

    def test_get_quantized(self, quant, queries, distance):
        idx = np.argmin(distance, axis=1)
        expected = quant.quantizer[idx]
        pt.assert_almost_equal(quant.get_quantized(queries), expected)


class TestQuantization1D:

    @pytest.fixture(params = [(3,1), (1,3), (3,)])
    def quantizer(self, request):
        out = np.array([0.5, 1., 1.5])
        out.shape = request.param
        return out

    @pytest.fixture
    def quant(self, quantizer):
        return qutil.Quantization1D(quantizer, bounds=[0., 2.])

    def test_shape(self, quant):
        assert quant.shape == (3,)

    def test_ndim(self, quant):
        assert quant.ndim == 1

    def test_size(self, quant):
        assert quant.size == 3

    def test_quantizer(self, quant):
        pt.assert_almost_equal(
                quant.quantizer,
                np.array([[0.5],[1.],[1.5]])
                )

    def test_bounds(self, quant):
        pt.assert_almost_equal(quant.bounds, np.array([[0.],[2.]]))

    def test_digitize_returns_ndim_tuple(self, quant):
        out = quant._digitize(np.zeros((3,1)))
        assert isinstance(out, tuple)
        assert len(out) == 1

    def test_digitize(self, quant):
        idx = quant._digitize(
            np.array([[0.45], [0.55], [1.20], [1.99], [0.01], [0.92]]))
        idx = idx[0]
        pt.assert_equal(np.array([0,0,1,2,0,1]), idx)

    # Test internals...

    def test_voronoi(self, quant):
        pt.assert_almost_equal(quant._voronoi,
                np.array([0., 0.75, 1.25, 2.]))

def test_format_2d_bounds():
    bounds = qutil._format_2d_bounds(None, None)
    pt.assert_equal(
            bounds,
            np.array([[-np.inf, -np.inf], [np.inf, np.inf]])
            )

    bounds = qutil._format_2d_bounds([0.,2.], None)
    pt.assert_equal(
            bounds,
            np.array([[0., -np.inf], [2., np.inf]])
            )

    bounds = qutil._format_2d_bounds(None, [3., 4.])
    pt.assert_equal(
            bounds,
            np.array([[-np.inf, 3.], [np.inf, 4.]])
            )

class TestGridQuantization2D:

    @pytest.fixture
    def value1(self):
        return np.array([0.5, 1., 1.5])

    @pytest.fixture
    def value2(self):
        return np.array([2.5, 3., 4.])

    @pytest.fixture
    def quant(self, value1, value2):
        return qutil.GridQuantization2D(
                value1, value2, bounds1=[0.25,5.], bounds2=[0.,10.])

    def test_shape(self, quant):
        assert quant.shape == (3,3)

    def test_ndim(self, quant):
        assert quant.ndim == 2

    def test_size(self, quant):
        assert quant.size == 9

    def test_quantizer(self, quant):
        expected = np.empty((3,3,2))
        expected[0,:,0] = 0.5
        expected[1,:,0] = 1.
        expected[2,:,0] = 1.5
        expected[:,0,1] = 2.5
        expected[:,1,1] = 3.
        expected[:,2,1] = 4.
        pt.assert_almost_equal(
                quant.quantizer,
                expected
                )

    def test_bounds(self, quant):
        expected = np.array([[0.25, 0.],[5.,10.]])
        pt.assert_almost_equal(quant.bounds, expected)

    def test_digitize_returns_ndim_tuple(self, quant):
        out = quant._digitize(np.zeros((3,2)))
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_digitize(self, quant):
        idx = quant._digitize(
            np.array([[0.45, 4.05],
                [0.55, 2.95],
                [1.20, 2.25],
                [1.99, 3.99],
                [0.01, 2.85],
                [0.92, 2.25]]))
        idx1 = idx[0]
        idx2 = idx[1]
        pt.assert_equal(np.array([0,0,1,2,0,1]), idx1)
        pt.assert_equal(np.array([2,1,0,2,1,0]), idx2)

    # Test internals

    def test_voronoi1(self, quant):
        pt.assert_almost_equal(
                quant._voronoi1, np.array([0.25,0.75,1.25,5.]))

    def test_voronoi2(self, quant):
        pt.assert_almost_equal(
                quant._voronoi2, np.array([0.,2.75,3.5,10.]))

    # Some redundant tests...

    def test_quantize_ok(self, quant):
        result = quant.get_quantized(np.array([[0.51,3.75],
            [0.85, 4.25],
            [2.25, 2.65],
            ]))
        expected = np.array([[0.5, 4.],
            [1.,4.],
            [1.5, 2.5],
            ])
        pt.assert_almost_equal(result, expected)

    def test_quantize_invalid_lower_first_dim(self, quant):
        result = quant.get_quantized(np.array([
            [0.15,5.],
            [0.,5.],
            ]))
        pt.assert_invalid(result)

    def test_quantize_invalid_upper_first_dim(self, quant):
        result = quant.get_quantized(np.array([
            [10., 4.25],
            [6., 2.5]]))
        pt.assert_invalid(result)

    def test_quantize_invalid_lower_second_dim(self, quant):
        result = quant.get_quantized(np.array([
            [3.25, -10.],
            [4.25, -0.25]]))
        pt.assert_invalid(result)

    def test_quantize_invalid_upper_second_dim(self, quant):
        result = quant.get_quantized(np.array([
            [3.25, 11.25],
            [4.25, 100.]]))
        pt.assert_invalid(result)


class TestConditionalQuantization2D:

    @pytest.fixture
    def value1(self):
        return np.array([0.5, 1., 1.5])

    @pytest.fixture
    def value2(self):
        return np.array([[2.5, 3., 4.], [5., 6., 7.], [2., 2.5, 3.]])

    @pytest.fixture
    def quant(self, value1, value2):
        return qutil.ConditionalQuantization2D(
                value1, value2, bounds1=[0.25,5.], bounds2=[0.,10.])

    def test_check_inconsistent_shape_1(self):
        value1 = np.array([0.5, 1., 1.5])
        value2 = np.array([[2.5, 3., 4.], [5., 6., 7.]])
        with pytest.raises(ValueError):
            q = qutil.ConditionalQuantization2D(value1, value2)

    def test_check_inconsistent_shape_2(self):
        value1 = np.array([0.5, 1., 1.5])
        value2 = np.array([2.5, 3., 4.])
        with pytest.raises(ValueError):
            q = qutil.ConditionalQuantization2D(value1, value2)

    def test_shape(self, quant):
        assert quant.shape == (3,3)

    def test_ndim(self, quant):
        assert quant.ndim == 2

    def test_size(self, quant):
        assert quant.size == 9

    def test_quantizer(self, quant, value2):
        expected = np.empty((3,3,2))
        expected[0,:,0] = 0.5
        expected[1,:,0] = 1.
        expected[2,:,0] = 1.5
        expected[:,:,1] = value2
        pt.assert_almost_equal(
                quant.quantizer,
                expected
                )

    def test_bounds(self, quant):
        expected = np.array([[0.25, 0.],[5.,10.]])
        pt.assert_almost_equal(quant.bounds, expected)

    def test_digitize_returns_ndim_tuple(self, quant):
        out = quant._digitize(np.zeros((3,2)))
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_digitize(self, quant):
        idx = quant._digitize(
            np.array([[0.45, 4.05],
                [0.55, 2.95],
                [1.20, 6.05],
                [1.99, 2.5],
                [0.01, 2.85],
                [0.92, 5.05]]))
        idx1 = idx[0]
        idx2 = idx[1]
        pt.assert_equal(np.array([0,0,1,2,0,1]), idx1)
        pt.assert_equal(np.array([2,1,1,1,1,0]), idx2)

    # Test internals...

    def test_voronoi1(self, quant):
        pt.assert_almost_equal(
                quant._voronoi1,
                np.array([0.25,0.75,1.25,5.])
                )

    def test_voronoi2(self, quant):
        pt.assert_almost_equal(
                quant._voronoi2,
                np.array([
                    [0.,2.75,3.5,10.],
                    [0.,5.5,6.5,10.],
                    [0.,2.25,2.75,10.],
                    ]))

    # Some redundant tests...

    def test_quantize_ok(self, quant):
        result = quant.get_quantized(np.array([[0.51,3.75],
            [0.85, 4.25],
            [2.25, 2.65],
            ]))
        expected = np.array([[0.5, 4.],
            [1.,5.],
            [1.5, 2.5],
            ])
        pt.assert_almost_equal(result, expected)

    def test_quantize_invalid_lower_first_dim(self, quant):
        result = quant.get_quantized(np.array([
            [0.15,5.],
            [0.,5.],
            ]))
        pt.assert_invalid(result)

    def test_quantize_invalid_upper_first_dim(self, quant):
        result = quant.get_quantized(np.array([
            [10., 4.25],
            [6., 2.5]]))
        pt.assert_invalid(result)

    def test_quantize_invalid_lower_second_dim(self, quant):
        result = quant.get_quantized(np.array([
            [3.25, -10.],
            [4.25, -0.25]]))
        pt.assert_invalid(result)

    def test_quantize_invalid_upper_second_dim(self, quant):
        result = quant.get_quantized(np.array([
            [3.25, 11.25],
            [4.25, 100.]]))
        pt.assert_invalid(result)

class TestMarkovQuantization:

    @pytest.fixture(params = [(5,), (2,3), (2,2,2)])
    def shape(self, request):
        return request.param

    @pytest.fixture(params = [1,2,3])
    def ndim(self, request):
        return request.param

    @pytest.fixture
    def prev_shape(self, shape):
        return shape

    @pytest.fixture(params = [1,2])
    def prev_ndim(self, request):
        return request.param

    @pytest.fixture
    def quantizer(self, shape, ndim):
        s = shape+(ndim,)
        out = np.arange(np.prod(s))
        out.shape = s
        return 1.5*out

    @pytest.fixture
    def previous_quantizer(self, prev_shape, prev_ndim):
        s = prev_shape+(prev_ndim,)
        out = np.arange(np.prod(s))
        out.shape = s
        return 0.5*out

    @pytest.fixture
    def bounds(self, quantizer, ndim):
        bounds = np.empty((2, ndim))
        for n in range(ndim):
            bounds[0,n] = np.amin(quantizer[...,n])-1.
            bounds[1,n] = np.amax(quantizer[...,n])+1.
        return bounds

    @pytest.fixture
    def current(self, quantizer, bounds):
        return QuantizationStub(quantizer, bounds)

    @pytest.fixture
    def previous(self, previous_quantizer):
        return QuantizationStub(previous_quantizer, None)

    def test_invalid_quant_raises_value_error(self):
        with pytest.raises(ValueError):
            qutil.MarkovQuantizationDecorator(1.,1.)

    @pytest.fixture
    def quant(self, current, previous):
        return qutil.MarkovQuantizationDecorator(current, previous)

    def test_shape(self, quant, shape):
        assert quant.shape == shape

    def test_ndim(self, quant, ndim):
        assert quant.ndim == ndim

    def test_size(self, quant, shape):
        assert quant.size == np.prod(shape)

    def test_prev_shape(self, quant, prev_shape):
        assert quant.previous.shape == prev_shape

    def test_prev_ndim(self, quant, prev_ndim):
        assert quant.previous.ndim == prev_ndim

    def test_prev_size(self, quant, prev_shape):
        assert quant.previous.size == np.prod(prev_shape)

    def test_quantizer(self, quant, quantizer):
        pt.assert_almost_equal(
                quant.quantizer,
                quantizer)

    def test_bounds(self, quant, bounds):
        pt.assert_almost_equal(quant.bounds, bounds)

    def test_digitize(self, current, quant, shape, ndim):
        idx = quant._digitize(np.zeros((3,ndim)))
        expected = current._digitize(np.zeros((3,ndim)))
        for n in range(len(shape)):
            pt.assert_equal(idx[n], expected[n])

    def test_estimate_and_set_distortion(self, current, quant):
        quant.estimate_and_set_distortion(current._ravel())
        dist1 = quant.distortion
        current.estimate_and_set_distortion(current._ravel())
        dist2 = current.distortion
        assert np.isclose(dist1, dist2)

    def test_set_distortion(self, quant):
        quant.set_distortion(1.)
        assert np.isclose(quant.distortion, 1.)
        assert np.isclose(quant.current.distortion, 1.)

    def test_set_probability(self, quant):
        p = np.ones(quant.shape)/quant.size
        quant.set_probability(p)
        pt.assert_almost_equal(quant.probability, p)
        pt.assert_almost_equal(quant.current.probability, p)

    # Estimate all probabilities

    def test_estimate_and_set_reverts_to_super(self, current, quant):
        quant.estimate_and_set_probability(current._ravel())
        proba1 = quant.probability.copy()
        current.estimate_and_set_probability(current._ravel())
        proba2 = current.probability.copy()
        pt.assert_almost_equal(proba1, proba2)

    def test_estimate_and_set_all_probability_strict(
            self, quant, shape, prev_shape):
        idx = tuple(np.indices(shape))
        for i in idx:  # ravel all
            i.shape = -1
            i[0] = i[1] # First state never observed!
        prev_idx = tuple(np.indices(prev_shape))
        for i in prev_idx:  # ravel all
            i.shape = -1
        quant._estimate_and_set_all_probabilities(
                idx, prev_idx, False)  # Ok if not strict
        with pytest.raises(qutil.UnobservedState):
            quant._estimate_and_set_all_probabilities(
                    idx, prev_idx, True)

    def test_estimate_and_set_all_proba(
            self, current, previous, quant, shape, prev_shape):
        idx = tuple(np.indices(shape))
        for i in idx:  # ravel all
            i.shape = -1
        prev_idx = tuple(np.indices(prev_shape))
        for i in prev_idx:  # ravel all
            i.shape = -1
        quant._estimate_and_set_all_probabilities(idx, prev_idx, True)
        expected_proba = 1./np.prod(shape) * np.ones(shape)
        pt.assert_almost_equal(quant.probability, expected_proba)
        pt.assert_almost_equal(current.probability, expected_proba)
        pt.assert_almost_equal(quant.previous.probability, expected_proba)
        pt.assert_almost_equal(previous.probability, expected_proba)
        expected_trans = self._get_diagonal_trans(prev_shape, shape)
        pt.assert_almost_equal(quant.transition_probability, expected_trans)

    def test_estimate_at_wrong_shape_raises_value_error(
            self, ndim, prev_ndim, quant):
        with pytest.raises(ValueError):
            quant.estimate_and_set_probability(np.zeros((3,ndim,1)))
        with pytest.raises(ValueError):
            quant.estimate_and_set_probability(np.zeros((3,ndim)),
                    np.zeros((3,prev_ndim,1)))
        with pytest.raises(ValueError):
            quant.estimate_and_set_probability(np.zeros((3,ndim)),
                    np.zeros((2,prev_ndim)))

    def test_estimate_and_set_probability_at_quantizer_is_1_over_N(
            self, quant, prev_shape, shape):
        quant.estimate_and_set_probability(quant._ravel(),
                quant.previous._ravel())
        expected_trans = self._get_diagonal_trans(prev_shape, shape)
        pt.assert_almost_equal(
                quant.transition_probability,
                expected_trans
                )

    def _get_diagonal_trans(self, prev_shape, shape):
        trans = np.zeros(prev_shape + shape)
        it = np.nditer(np.empty(prev_shape), flags=['multi_index'])
        while not it.finished:
            trans[it.multi_index*2] = 1.
            it.iternext()
        return trans

    # TODO simple 1d to 1d scenario

    # Transition probability setter

    @pytest.fixture
    def trans(self, prev_shape, shape):
        np.random.seed(907831247)
        trans = np.random.uniform(low=0., high=1., size=prev_shape+shape)
        self._normalize_trans(trans, prev_shape, shape)
        return trans

    def _normalize_trans(self, trans, prev_shape, shape):
        norm_factor = np.sum(trans, axis = tuple(range(-len(shape),0)))
        norm_factor.shape = prev_shape + (1,)*len(shape)
        trans /= norm_factor

    def test_set_transition_probability(self, quant, trans):
        quant.set_transition_probability(trans)
        pt.assert_almost_equal(quant.transition_probability, trans)

    def test_set_wrong_shape_trans_raises_value_error(self, quant, trans):
        with pytest.raises(ValueError):
            quant.set_transition_probability(np.ravel(trans))

    @pytest.mark.parametrize("wrong", [np.nan, np.inf])
    def test_set_invalid_trans_raises_value_error(self, quant, trans, wrong):
        trans[0] = wrong
        with pytest.raises(ValueError):
            quant.set_transition_probability(trans)

    def test_unorm_trans_raises_value_error(self, quant, trans):
        trans[0] += 0.1
        with pytest.raises(ValueError):
            quant.set_transition_probability(trans)

    def test_null_trans_is_ok(self, quant, trans, prev_shape, shape):
        trans[(0,)*trans.ndim] = 0.
        self._normalize_trans(trans, prev_shape, shape)
        quant.set_transition_probability(trans)

    def test_negative_trans_raises_value_error(
            self, quant, trans, prev_shape, shape):
        trans[(0,)*trans.ndim] = -0.1
        self._normalize_trans(trans, prev_shape, shape)
        with pytest.raises(ValueError):
            quant.set_transition_probability(trans)


class TestAbstractQuantizer1D:

    @pytest.fixture(params=[1,4,3],
                    ids=['prev_size(1)',
                         'prev_size(4)',
                         'prev_size(3)',
                         ])
    def prev_size(self, request):
        return request.param

    @pytest.fixture
    def prev_proba(self, prev_size):
        np.random.seed(907831247)
        p = np.random.uniform(
            low=0.05,
            high=1,
            size=(prev_size,))
        p /= np.sum(p)  # Normalize
        return p

    @pytest.fixture
    def factory(self, prev_proba):
        return _QuantizerFactory1DStub(prev_proba)

    @pytest.fixture(params=[1, 2, 3, 5],
                    ids=['size(1)',
                         'size(2)',
                         'size(3)',
                         'size(5)' ])
    def size(self, request):
        return request.param

    @pytest.fixture
    def value(self, size):
        np.random.seed(907831247)
        value = np.random.normal(size=(size,))
        value.sort()
        return value

    ###############################
    # Testing test implementation #
    ###############################

    @pytest.mark.skip
    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_integral(self, factory, prev_size, size, value, order):
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
        expected.shape = (1,size)
        expected = np.broadcast_to(expected, (prev_size, size))
        # Value
        quantizer = factory.make(value)
        value = quantizer._integral[order]
        # Assertion
        pt.assert_almost_equal(value, expected)

    @pytest.mark.skip
    @pytest.mark.parametrize("order", [0, 1])
    def test_delta(self, factory, value, order):
        qutil._assert_valid_delta_at(factory, value, order)

    #########
    # Tests #
    #########

    def test_bounds_default_to_pm_inf(self, factory):
        assert factory.make(np.array([1.])).bounds == [-np.inf, np.inf]

    def test_size(self, factory, value):
        q = factory.make(value)
        assert q.size == value.size

    def test_value_is_broadcast_ready(self, factory):
        value = np.array([-1., 1.])
        q = factory.make(value)
        expected = value.copy()
        expected.shape = (1,2)
        pt.assert_almost_equal(q.value, expected)

    def test_voronoi_is_broadcast_ready(self, factory):
        value = np.array([-1., 1.])
        q = factory.make(value)
        expected = np.array([-np.inf, 0., np.inf])
        expected.shape = (1,3)
        pt.assert_almost_equal(q.voronoi, expected)

    @pytest.fixture(params = [1., 0.5])
    def norm(self, request):
        return request.param

    @pytest.fixture
    def quantizer(self, factory, value, norm):
        return factory.make(value, norm)

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

    def test_proba_and_trans_proba(self, factory, quantizer, prev_size):
        value = quantizer.probability
        expected = np.zeros_like(value)
        for (pp, tp) in zip(
                factory.prev_proba, quantizer.transition_probability):
            expected += pp * tp
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
        expected.shape = (1, quantizer.size)
        expected = np.broadcast_to(
                expected, (quantizer.prev_size, quantizer.size))
        value = quantizer._distortion_elements
        pt.assert_almost_equal(value, expected, rtol=1e-2)

    def test_distortion(self, factory, quantizer, norm):
        # See below for _distortion_elements test
        distortion = np.sum(quantizer._distortion_elements, axis=-1)
        expected = 0.
        for (pp,d) in zip(factory.prev_proba, distortion):
            expected += pp*d
        pt.assert_almost_equal(expected/norm, quantizer.distortion)

    def test_gradient(self, factory, quantizer, norm):
        pt.assert_gradient_at(quantizer.gradient,
                              np.ravel(quantizer.value),
                              function=lambda x: factory.make(x, norm).distortion)

    def test_hessian(self, factory, quantizer, norm):
        pt.assert_hessian_at(quantizer.hessian,
                             np.ravel(quantizer.value),
                             gradient=lambda x: factory.make(x, norm).gradient)

    def test_hessian_is_symmetric(self, quantizer):
        pt.assert_almost_equal(
            quantizer.hessian, np.transpose(quantizer.hessian))

    def test_conditional_expectation(self, quantizer):
        def _numerical_expectation(value):
            def do(low, high, value):
                def f(z): return z*norm.pdf(z)
                return (integrate.quad(f, low, high)[0]
                        /(norm.cdf(high)-norm.cdf(low)))
            # Perform integration for each value
            vor = qutil.voronoi_1d(np.ravel(value))
            out = np.empty((value.size,))
            for j in range(value.size):
                out[j] = do(vor[j], vor[j+1], value[j])
            return out
        expected = _numerical_expectation(np.ravel(quantizer.value))
        value = quantizer.conditional_expectation
        pt.assert_almost_equal(value, expected, rtol=1e-6)

    def test_conditional_expectation_stays_sorted(self, quantizer):
        assert np.all(
                np.diff(quantizer.conditional_expectation)>0.
                )

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
        x[0] *= 1.0001
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


    def test_invert(self, optim_factory, shape):
        np.random.seed(12312)
        sb = optim_factory.search_bounds
        values = np.random.uniform(low=sb[0], high=sb[1], size=shape)
        values.sort()
        x = optim_factory.invert(values)
        q = optim_factory.make(x)
        pt.assert_almost_equal(q.value, values)


class TestOptimQuantizer1D:

    @pytest.fixture(params=[1,2,5,10],
                    ids=['size(1)',
                        'size(2)',
                        'size(5)',
                        'size(10)',
                        ])
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
    def optim_quantizer(self, size, optim_factory):
        np.random.seed(1231)
        return optim_factory.make(np.random.normal(size=size))

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
                               np.log(optim_quantizer.quantizer.distortion))

    def test_theta(self, size, optim_quantizer):
        x = optim_quantizer.x
        for i in range(1,size):
            expected = -0.5
            for ii in range(0,i+1):
                expected += np.exp(x[ii])/(size+1)
            assert np.isclose(expected, optim_quantizer.theta[i])

    def test_jacobian_analytic(self, size, optim_quantizer):
        expected = np.zeros((size,size))
        off = optim_quantizer.OFFSET
        x = optim_quantizer.x
        sb = optim_quantizer.search_bounds
        span = sb[1] - sb[0]
        theta = optim_quantizer.theta
        for m in range(size):
            for n in range(size):
                expected[m,n] = (
                        np.double(m>=n)*span
                        *off*np.exp(x[n])/(size+1)
                        *dlogistic(off*theta[m]))
        pt.assert_almost_equal(expected, optim_quantizer.jacobian)

    def test_jacobian(self, optim_factory, optim_quantizer):
        def f(x): return optim_factory.make(x).value
        pt.assert_jacobian_at(
            optim_quantizer.jacobian,
            optim_quantizer.x,
            function=f)

    def test_hessian_analytic(self, size, optim_quantizer):
        sb = optim_quantizer.search_bounds
        span = sb[1] - sb[0]
        jac = optim_quantizer.jacobian
        dist = optim_quantizer.quantizer.distortion
        hess = optim_quantizer.quantizer.hessian
        grad = optim_quantizer.quantizer.gradient
        theta = optim_quantizer.theta
        off = optim_quantizer.OFFSET
        x = optim_quantizer.x
        df = off*np.exp(x)/(x.size+1)
        ddf = df[np.newaxis,:] * df[:,np.newaxis]
        df = np.diag(df)
        expected = jac.T.dot(hess).dot(jac)
        for (i, g) in enumerate(grad):
            expected[:i+1, :i+1] += \
                    span*g*df[:i+1,:i+1]*dlogistic(off*theta[i])
            expected[:i+1, :i+1] += \
                    span*g*ddf[:i+1,:i+1]*ddlogistic(off*theta[i])
        # Log - transformation
        expected /= dist
        expected -= (optim_quantizer.gradient[:,np.newaxis]
                *optim_quantizer.gradient[np.newaxis,:])
        pt.assert_almost_equal(expected, optim_quantizer.hessian)

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
            niter,
            niter_success,
            verbose):
        return qutil.Optimizer1D(factory, size,
                robust=robust,
                maxiter=maxiter,
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


    def test_niter(self, niter, optimizer):
        assert optimizer.niter == niter

    def test_niter_success(self, niter_success, optimizer):
        assert optimizer.niter_success == niter_success

    def test_verbose(self, verbose, optimizer):
        assert optimizer.verbose == verbose

    def test_optimize(self, search_bounds, optimizer, solution):
        opt = optimizer.optimize(search_bounds)
        assert isinstance(opt,  _Quantizer1DStub)
        pt.assert_almost_equal(opt.value, solution.value, rtol=1e-4)

    # Test internals...

    def test_free_memory_after_optimize(self, search_bounds, optimizer):
        optimizer.optimize(search_bounds)
        assert optimizer.factory._cache._Cache__currsize == 0

    def test_trust_ncg_options(self, optimizer):
        options = optimizer._format_trust_ncg_options()
        assert options['maxiter'] == optimizer.maxiter


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


class TestStochasticOptim:

    @pytest.fixture
    def weight(self):
        return np.ones((1,))

    @pytest.fixture(scope='class', params=[2,10])
    def size(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def solution(self, size):
        factory = _QuantizerFactory1DStub(np.array([1.]))
        opt = qutil.Optimizer1D(factory, size)
        result = opt.optimize((-5.,5.))
        quantizer = np.ravel(result.value)
        quantizer.shape = (size,1)
        return quantizer

    @pytest.fixture(scope='class')
    def simulations(self):
        nsim = 10000000
        np.random.seed(9312234)
        out = np.random.normal(size=nsim)
        out.shape = (nsim,1)
        return out

    @pytest.fixture(scope='class')
    def starting(self, size):
        np.random.seed(8923478)
        out = np.random.normal(size=size)
        out.shape = (size,1)
        return out

    @pytest.mark.slow
    def test_converge(self, size, starting, simulations, solution):
        out = qutil.VoronoiQuantization.stochastic_optimization(
                starting, simulations, nlloyd=10, split_step=1)
        result = np.sort(out.quantizer[:,0])
        expected = solution[:,0]
        pt.assert_almost_equal(result, expected, rtol=0.02)
