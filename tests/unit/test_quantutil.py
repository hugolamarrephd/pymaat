import numpy as np
import pytest

import pymaat.testing as pt
import pymaat.quantutil as qutil

class TestVoronoi1D:

    @pytest.fixture(params=[1,2])
    def ndim(self, request):
        return request.param

    @pytest.fixture(params=[0,1])
    def axis(self, request, ndim):
        return min(request.param, ndim-1)

    @pytest.fixture(params=[
        (np.array([1.]), np.array([])),
        (np.array([1.,2.,3.]),np.array([1.5,2.5])),
        (np.array([1.,3.,4.,10.]),np.array([2.,3.5,7.])),
        ])
    def setup(self, request):
        """
        (quantizer, voronoi (without bounds))
        """
        return request.param

    @pytest.fixture(params=[(-np.inf,np.inf),(0.0,np.inf)],
            ids=['unbounded','positive'])
    def bounds(self, request):
        return request.param

    @pytest.fixture
    def quantizer(self, setup, ndim, axis):
        values = setup[0]
        size = values.size
        shape =  np.full((ndim,), 1, dtype=np.int)
        shape[axis] = size
        return np.reshape(values, tuple(shape))

    @pytest.fixture
    def voronoi_with_bounds(self, setup, ndim, axis, bounds):
        values = setup[1]
        size = values.size
        if size>0:
            # Append bounds to both ends
            out = np.empty((size+2,))
            out[0] = bounds[0]
            out[-1] = bounds[-1]
            out[1:-1] = values
        else:
            out = np.array(bounds)
        shape =  np.full((ndim,), 1, dtype=np.int)
        shape[axis] = size
        shape[axis] += 2
        return np.reshape(out, tuple(shape))

    def test_with_bounds(self,
            axis, quantizer, bounds, voronoi_with_bounds):
        v = qutil.voronoi_1d(quantizer, axis=axis,
            lb=bounds[0], ub=bounds[-1])
        pt.assert_almost_equal(v, voronoi_with_bounds)

    def test_supports_non_array(self):
        v = qutil.voronoi_1d([1.,2.,3.])
        pt.assert_almost_equal(v, np.array([-np.inf,1.5,2.5,np.inf]))

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
        if axis==0:
            first_quantizer=quantizer[0]
        elif axis==1:
            first_quantizer=quantizer.T[0]
        q = qutil.inv_voronoi_1d(voronoi_with_bounds,
                first_quantizer=first_quantizer,
                axis=axis, with_bounds=True)
        pt.assert_almost_equal(q, quantizer)

    def test_inv_supports_non_array(self):
        q = qutil.inv_voronoi_1d([-np.inf,1.5,2.5,np.inf],
                first_quantizer=1.)
        pt.assert_almost_equal(q, np.array([1.,2.,3.]))

    # TODO fails when empty?
