import pytest
import numpy as np

import pymaat.testing as pt

from pymaat.garch.quant import PriceVarianceGrid, PriceVarianceConditonal
from pymaat.garch.quant import VarianceOnly


class TestPriceVarianceCore:

    def test_shapes_from_nper(self, model, variance_scale):
        core = PriceVariance(
            model,
            variance_scale,
            price_size=100,
            variance_size=10,
            nper=21)
        assert core._shapes == [(100, 10)]*21

    def test_shapes_no_nper_from_scalar_size(self, model, variance_scale):
        core = PriceVariance(
            model,
            variance_scale,
            price_size=100,
            variance_size=10)
        assert core._shapes == [(100, 10)]

    def test_shapes_no_nper(self, model, variance_scale):
        core = PriceVariance(
            model,
            variance_scale,
            price_size=np.array([10, 20, 100]),
            variance_size=np.array([1, 5, 10]))
        assert core._shapes == [(10, 1), (20, 5), (100, 10)]

    def test_defaults(self, model, variance_scale):
        core = PriceVariance(model, variance_scale)
        pt.assert_equal(core._shapes, [(100, 10)])
        pt.assert_almost_equal(core._first_quantization.value1,
                               np.array([1.]))
        pt.assert_almost_equal(core._first_quantization.value2,
                               np.array([[variance_scale]]))
        pt.assert_almost_equal(core._first_quantization.probability,
                               np.array([[1.]]))

    def test_first_quant(self, model, variance_scale):
        fp = np.array([0.9, 1., 1.2])
        fv = variance_scale*np.array([
            [0.8, 1.2],
            [1., 1.025],
            [1.05, 1.2],
        ])
        fprob = np.array([
            [0.1, 0.1],
            [0.1, 0.1],
            [0.2, 0.4],
        ])
        core = PriceVariance(model,
                             first_price=fp,
                             first_variance=fv,
                             first_probability=fprob)
        pt.assert_almost_equal(core._first_quantization.value1, fp)
        pt.assert_almost_equal(core._first_quantization.value2, fv)
        pt.assert_almost_equal(core._first_quantization.probability, fprob)

    def test_quielty_normalize_first_proba(self, model, variance_scale):
        fp = np.array([0.9, 1., 1.2])
        fv = variance_scale*np.array([
            [0.8, 1.2],
            [1., 1.025],
            [1.05, 1.2],
        ])
        fprob = 9.75*np.array([
            [0.1, 0.1],
            [0.1, 0.1],
            [0.2, 0.4],
        ])
        core = PriceVariance(model,
                             first_price=fp,
                             first_variance=fv,
                             first_probability=fprob)
        pt.assert_almost_equal(core._first_quantization.probability,
                               np.array([
                                   [0.1, 0.1],
                                   [0.1, 0.1],
                                   [0.2, 0.4],
                               ]))

    # TODO: test one step optimize!


class TestVarianceOnlyCore:

    def test_shapes_from_nper(
            self, model, variance_scale):
        core = VarianceOnly(model, variance_scale, size=10, nper=21)
        pt.assert_equal(core._shapes, np.ones((21,))*10)

    def test_shapes_no_nper_from_scalar_size(
            self, model, variance_scale):
        core = VarianceOnly(model, variance_scale, size=10)
        pt.assert_equal(core._shapes, np.array(10))

    def test_shapes_no_nper(
            self, model, variance_scale):
        core = VarianceOnly(model, variance_scale, size=np.array([1, 2, 3, 4]))
        pt.assert_equal(core._shapes, np.array([1, 2, 3, 4]))

    def test_defaults(self, model, variance_scale):
        core = VarianceOnly(model, variance_scale)
        pt.assert_equal(core._shapes, np.array(100))
        pt.assert_almost_equal(core._first_quantization.value,
                               np.array([variance_scale]))
        pt.assert_almost_equal(core._first_quantization.probability,
                               np.array([1.]))

    def test_first_quant(self, model):
        core = VarianceOnly(
            model,
            first_variance=np.array([1., 2., 3.]),
            first_probability=np.array([0.5, 0.25, 0.25])
        )
        pt.assert_almost_equal(core._first_quantization.value,
                               np.array([1., 2., 3.]))
        pt.assert_almost_equal(core._first_quantization.probability,
                               np.array([0.5, 0.25, 0.25]))

    def test_quietly_normalize_first_proba(self, model):
        core = VarianceOnly(
            model,
            first_variance=np.array([1., 2., 3.]),
            first_probability=10.*np.array([0.5, 0.25, 0.25])
        )
        pt.assert_almost_equal(core._first_quantization.probability,
                               np.array([0.5, 0.25, 0.25]))

    # TODO: test one step optimize!
