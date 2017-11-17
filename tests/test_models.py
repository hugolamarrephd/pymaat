import unittest

import numpy as np

import pymaat.models as p


class TestGarchFilters(unittest.TestCase):
    def setUp(self):
        self.model = p.Garch()

    def test_filter_len2_tuple(self):
        out = self.model.filter(NORM_RAND,1e-7)
        self.assertIsInstance(out,tuple)
        self.assertEqual(len(out),2)

    def test_filter_size(self):
        out = self.model.filter(NORM_RAND,1e-7)
        s = NORM_RAND.shape
        self.assertEqual(out[0].shape,(s[0]+1,)+s[1:]) # variance
        self.assertEqual(out[1].shape,s) # innovations

    def test_filter_positive_variance(self):
        (h,*_) = self.model.filter(NORM_RAND,1e-7)
        np.testing.assert_array_equal(h>0,True)

    def test_filter_throws_exception_when_non_positive_variance(self):
        with self.assertRaises(ValueError):
            self.model.filter(NORM_RAND,-1)
        with self.assertRaises(ValueError):
            self.model.filter(NORM_RAND,0)

    def test_one_step_filter_len2_tuple(self):
        out = self.model.one_step_filter(NORM_RAND, GAM_RAND)
        self.assertIsInstance(out,tuple)
        self.assertEqual(len(out),2)

    def test_one_step_filter_consistent_size(self):
        out = self.model.one_step_filter(NORM_RAND, GAM_RAND)
        self.assertEqual(out[0].size, NORM_RAND.size)
        self.assertEqual(out[1].size, NORM_RAND.size)

    def test_one_step_filter_positive_variance(self):
        next_variances, *_ = self.model.one_step_filter(NORM_RAND, GAM_RAND)
        np.testing.assert_array_equal(next_variances>0,True)

    def test_one_step_simulate_len2_tuple(self):
        out = self.model.one_step_simulate(NORM_RAND, GAM_RAND)
        self.assertIsInstance(out,tuple)
        self.assertEqual(len(out),2)

    def test_one_step_simulate_consistent_size(self):
        out = self.model.one_step_simulate(NORM_RAND, GAM_RAND)
        self.assertEqual(out[0].size, NORM_RAND.size)
        self.assertEqual(out[1].size, NORM_RAND.size)

    def test_one_step_simulate_positive_variance(self):
        next_variances, *_ = self.model.one_step_simulate(NORM_RAND, GAM_RAND)
        np.testing.assert_array_equal(next_variances>0,True)

    def test_one_step_simulate_revertsto_filter(self):
        innovations = NORM_RAND
        current_variances = GAM_RAND
        sim_nvar, sim_returns = self.model.one_step_simulate(
            innovations,
            current_variances
        )
        filter_nvar, filter_innov = self.model.one_step_filter(
            sim_returns,
            current_variances
        )
        np.testing.assert_allclose(sim_nvar,filter_nvar)
        np.testing.assert_allclose(innovations,filter_innov)

    def test_neg_log_like_at_few_values(self):
        nll = self.model.negative_log_likelihood(1,1)
        self.assertAlmostEqual(nll,0.5)
        nll = self.model.negative_log_likelihood(0,np.exp(1))
        self.assertAlmostEqual(nll,0.5)



# Hard-code random sample to avoid test randomness
NORM_RAND = np.array([  1.24756770e+00,   2.08914936e+00,  -1.27108517e+00,
    9.24889588e-02,  -2.99486572e+00,   6.63851905e-01,
    1.99709826e+00,  -1.19870207e+00,  -5.08366828e-02,
    1.69526956e+00,  -7.98547759e-01,   1.03875675e+00,
    2.25280331e+00,  -3.88462408e-01,  -3.52430621e-01,
    2.99320489e-01,   1.39046021e-01,   1.08950341e+00,
    -8.59475371e-01,  -1.90737253e+00,   5.75872836e-01,
     1.63256775e+00,   8.73898123e-01,   3.45734054e-01,
     8.40884289e-01,   1.57190132e-02,   1.28333861e+00,
    -1.00755543e+00,   4.14665423e-02,   1.28900163e+00,
    -1.57401313e-01,   1.88485095e-02,  -1.07505006e+00,
    -9.22017617e-01,   1.66735834e+00,  -1.71799140e-01,
    -1.40131697e+00,   1.82635715e-01,  -8.57049607e-01,
     1.77979441e+00,   5.75947019e-01,   5.61500175e-01,
    -7.27851489e-01,  -1.01416794e+00,  -1.36878081e-01,
    -1.52275415e-01,  -1.93818774e-01,  -9.00729190e-01,
     3.68599560e-01,   4.76592952e-01,  -6.91982797e-01,
    -5.44016364e-01,   1.35430051e-01,  -4.36299354e-01,
     1.50220293e-01,  -1.63618994e-01,   2.16469417e+00,
     9.77923098e-01,   1.40195188e+00,   2.21147191e-01,
    -1.87959787e+00,  -1.09714118e-01,   3.92840047e-01,
     1.52597899e+00,   3.97661820e-01,   1.58063539e+00,
    -1.78618050e+00,  -1.58593779e-03,   3.01414171e-01,
    -1.77420403e-01,   8.77473667e-02,  -1.96126937e+00,
    -1.62278024e-01,   1.37337028e+00,   2.14470720e+00,
    -1.95148237e-01,   5.97364250e-01,   2.52435538e-01,
     6.82272583e-01,   3.34175337e-01,  -9.43967083e-01,
     1.06867823e+00,   3.35115587e-01,  -2.09461113e-01,
     2.40275475e+00,  -4.80403315e-01,   3.37966594e-01,
     9.33910158e-01,   6.31857422e-01,  -2.06248844e-01,
    -1.32890099e+00,   6.06298897e-01,   7.94792833e-01,
    -1.02040302e+00,   1.87372330e+00,   3.17392346e-01,
    -1.49266192e-01,   9.07922623e-01,  -4.30244060e-01,
    -9.16789507e-01])

GAM_RAND = np.array([  6.63975142e-01,   3.71325713e-01,   2.53582889e-01,
    2.22130249e+00,   1.51084218e-01,   5.09197754e-01,
    2.03723994e+00,   1.79911538e+00,   1.25810450e-01,
    3.61231684e-01,   1.48801360e-01,   5.79185431e-01,
    5.70774882e-02,   1.70161117e-01,   1.94311311e-01,
    2.82084295e-01,   3.78866053e+00,   6.42739247e-01,
    5.65659499e-02,   2.41782632e-01,   8.61487180e-01,
    3.99470968e-01,   4.15261744e-01,   8.84963260e-04,
    1.91341159e+00,   1.05887825e+00,   8.62162874e-01,
    3.15780374e-03,   3.58233691e-01,   4.43784333e-02,
    5.04928598e+00,   1.49859295e-01,   8.81756599e-01,
    1.38236593e+00,   9.36651240e-01,   1.15030619e-01,
    6.29524158e-02,   7.77507463e-01,   2.53345618e-01,
    7.10992681e-01,   4.24851815e-01,   2.78275428e-01,
    3.02120168e-01,   2.51910420e+00,   9.14999770e-01,
    3.13114032e+00,   5.84429189e+00,   8.64073072e-01,
    1.31352514e-01,   1.67567325e-01,   1.45310823e-01,
    7.24637864e-01,   6.32539520e-01,   2.88729565e-01,
    1.08753036e-01,   5.29744517e-01,   8.58559262e-01,
    9.40136092e-01,   9.05376519e-01,   1.13208009e-01,
    4.63924587e+00,   2.45102470e-01,   1.04570973e+00,
    4.53022904e+00,   9.71037042e-01,   3.38598305e+00,
    1.44280305e+00,   3.14878709e-01,   6.10256720e-01,
    9.38642200e-01,   2.41285232e+00,   2.10886942e-02,
    9.54472940e-01,   1.43308762e+00,   3.54300368e-02,
    3.39800422e+00,   1.31227604e+00,   6.19759788e-02,
    1.26040143e-01,   2.33726475e+00,   5.92150516e-01,
    8.99138356e-02,   3.72322230e-01,   2.39297476e-01,
    3.65711971e-03,   5.71818303e-01,   8.66844065e-01,
    6.15089711e-01,   6.74700311e-01,   1.10897769e+00,
    1.78757703e-01,   1.13328851e+00,   7.46080404e-01,
    1.33518328e+00,   1.27194919e-02,   1.91418688e+00,
    3.46539716e+00,   3.49583230e+00,   1.16530687e+00,
    9.88330735e-01])


