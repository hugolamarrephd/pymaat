import numpy as np
import pytest

import pymaat.testing as pt
import pymaat.mathutil as pmu

class TestVoronoi1D:
    # TODO: test tensors

    def test_vector(self):
        v = pmu.voronoi_1d(np.array([1.,2.,3.]))
        pt.assert_almost_equal(v, np.array([-np.inf,1.5,2.5,np.inf]))

    def test_supports_non_array(self):
        v = pmu.voronoi_1d([1.,2.,3.])
        pt.assert_almost_equal(v, np.array([-np.inf,1.5,2.5,np.inf]))

    def test_vector_positive(self):
        v = pmu.voronoi_1d(np.array([1.,2.,3.]), lb=0.)
        pt.assert_almost_equal(v, np.array([0.,1.5,2.5,np.inf]))

    def test_vector_negative(self):
        v = pmu.voronoi_1d(np.array([-3.,-2.,-1.]), ub=0.)
        pt.assert_almost_equal(v, np.array([-np.inf,-2.5,-1.5,0.]))

    def test_vector_when_N_is_one(self):
        v = pmu.voronoi_1d(np.array([1.]), lb=0., ub=2.)
        pt.assert_almost_equal(v, np.array([0.,2.]))

    def test_vector_when_N_is_zero(self):
        with pytest.raises(ValueError):
            pmu.voronoi_1d(np.array([]))

    # Supports matrices
    def test_matrices_columns(self):
        a_matrix = np.array([[1.,3.],[2.,4.]])
        v = pmu.voronoi_1d(a_matrix, axis=0)
        expected = np.array([[-np.inf,-np.inf],
                [1.5, 3.5],
                [np.inf, np.inf]])
        pt.assert_almost_equal(v, expected)
        pt.assert_equal(a_matrix, [[1.,3.],[2.,4.]])

    def test_matrices_rows(self):
        a_matrix = np.array([[1.,2.],[3.,4.]])
        v = pmu.voronoi_1d(a_matrix, axis=1)
        expected = np.array([[-np.inf, 1.5, np.inf],
                [-np.inf, 3.5, np.inf]])
        pt.assert_almost_equal(v, expected)
        pt.assert_equal(a_matrix, [[1.,2.],[3.,4.]])

    def test_matrices_columns_when_N_is_one(self):
        v = pmu.voronoi_1d(np.array([[1.,1.5]]),
            lb=0., ub=2., axis=0)
        pt.assert_almost_equal(v, np.array([[0.,0.],[2.,2.]]))

    def test_matrices_rows_when_N_is_one(self):
        v = pmu.voronoi_1d(np.array([[1.],[1.5]]),
            lb=0., ub=2., axis=1)
        pt.assert_almost_equal(v, np.array([[0.,2.],[0.,2.]]))

    def test_matrices_when_N_is_zero(self):
        with pytest.raises(ValueError):
            pmu.voronoi_1d(np.array([[]]), axis=0)
        with pytest.raises(ValueError):
            pmu.voronoi_1d(np.array([[]]), axis=1)
