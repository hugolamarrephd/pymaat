import pytest
import numpy as np

import pymaat.garch
import pymaat.garch.quant

NSIMUL = 10
SIZE = 10
VARIANCE_SCALE = 0.18**2./252.
MODEL = pymaat.garch.Garch(
        mu=2.01,
        omega=9.75e-20,
        alpha=4.54e-6,
        beta=0.79,
        gamma=196.21)
np.random.seed(1234567) # This line is critical to ensure test consistency
VARIANCES, _ = MODEL.one_step_generate(
        np.random.normal(size=(SIZE,NSIMUL)),
        VARIANCE_SCALE)

class TestMarginalVariance:

    # def test_initialize(self):
    #     grid, proba, trans = self.quant._initialize(first_variance)
    #     # Shapes
    #     self.assert_equal(grid.shape, (self.nper+1, self.nquant))
    #     self.assert_equal(proba.shape, (self.nper+1, self.nquant))
    #     self.assert_equal(trans.shape, (self.nper, self.nquant, self.nquant))
    #     # First grid
    #     self.assert_equal(grid[0,1:]==first_variance, True)

    # def get_optim_grid_from(self, init):
    #     (success, optim) = self.quant._one_step_quantize(self.state,
    #             init=init)
    #     assert success
    #     return optim

    # def test_one_step_quantize_converges_to_same_from_diff_init(self):
    #     optim1 = self.get_optim_grid_from(
    #         np.array([0.57234,-0.324234,3.4132,-2.123]))
    #     optim2 = self.get_optim_grid_from(
    #         np.array([-1.423,0.0234,-3.234,5.324]))
    #     optim3 = self.get_optim_grid_from(
    #         np.array([-0.0234,0.,0.002,1.234]))
    #     optim4 = self.get_optim_grid_from(
    #         np.array([-10,2.32,5.324,10.234]))
    #     self.assert_almost_equal(optim1, optim2)
    #     self.assert_almost_equal(optim2, optim3)
    #     self.assert_almost_equal(optim3, optim4)

    # def test_quantize_returns_valid_quantization(self):
    #     q = self.quant.quantize(0.18**2./252.)
    #     self.assert_true(type(q)==pymaat.garch.quant.Quantization)
    #     for t in range(1,self.nper+1):
    #         self.assert_almost_equal(
    #                 q.transition_probabilities[t-1].sum(axis=1), 1.)
    #         self.assert_almost_equal(
    #                 q.probabilities[t-1].dot(q.transition_probabilities[t-1]),
    #                 q.probabilities[t])
    #         self.assert_almost_equal(q.probabilities[t].sum(), 1.)
    #         self.assert_true(q.values[t]>0.)
    #         self.assert_true(np.diff(q.values[t])>0.)

pytest.main(__file__)
