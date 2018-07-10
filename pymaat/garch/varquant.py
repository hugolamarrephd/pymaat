import numpy as np
from scipy.special import ndtr

import pymaat.quantutil as qutil
from pymaat.garch.format import variance_formatter
from pymaat.util import lazy_property
import pymaat.testing as pt
from pymaat.nputil import printoptions

import time

class Factory:

    def __init__(self, model, prev_proba, prev_value,
                 low_z=None, high_z=None, z_proba=None):
        self.model = model
        low_z, high_z, z_proba = self._import_z_bounds(
            low_z, high_z, z_proba, prev_proba.shape)
        if not (prev_proba.size == prev_value.size == low_z.size
                == high_z.size == z_proba.size):
            raise ValueError("Previous quantization size mismatch")
        self.norm = np.sum(prev_proba*z_proba)
        if self.norm <= 0.:
            raise qutil.UnobservedState
        self.prev_proba = prev_proba
        self.prev_value = np.ravel(prev_value)
        self.low_z = np.ravel(low_z)
        self.high_z = np.ravel(high_z)

    def make(self, value):
        return Quantizer(value,
                          self.prev_proba,
                          self.model,
                          self.prev_value,
                          self.low_z,
                          self.high_z,
                          self.norm)

    def get_all_singularities(self):
        lb = self.model.get_lowest_one_step_variance(
                self.prev_value,
                self.low_z,
                self.high_z
                )
        ub = self.model.get_highest_one_step_variance(
                self.prev_value,
                self.low_z,
                self.high_z
                )
        out = np.concatenate((np.ravel(lb), np.ravel(ub)))
        out = out[np.isfinite(out)]
        return out


    def get_search_bounds(self, crop):
        lz = np.clip(self.low_z, -crop, crop)
        hz = np.clip(self.high_z, -crop, crop)
        valid = lz<hz
        if not np.any(valid):
            crop = 10.  # Override user and deal with inf only
            lz = np.clip(self.low_z, -crop, crop)
            hz = np.clip(self.high_z, -crop, crop)
            valid = lz<hz
        pv = self.prev_value[valid]
        lz = lz[valid]
        hz = hz[valid]
        lb = self.model.get_lowest_one_step_variance(pv, lz, hz)
        ub = self.model.get_highest_one_step_variance(pv, lz, hz)
        return np.array([np.amin(lb), np.amax(ub)])


    def _import_z_bounds(self, lb, ub, p, s):
        if lb is None or ub is None:
            # Default root bounds
            lb = np.full(s, -np.inf)
            ub = np.full(s, np.inf)
            p = np.ones(s)
        elif p is None:
            # Compute transition probability
            # By-pass if known for efficiency
            p = ndtr(ub) - ndtr(lb)
        # Check root bounds consistency
        if np.any(ub < lb):
            raise ValueError("Invalid root bounds (UB lower than LB)")
        else:
            return (lb, ub, p)


class Quantizer(qutil.AbstractQuantizer1D):

    bounds = [0., np.inf]

    def __init__(self, value, prev_proba, model, prev_value, low_z, high_z,
            norm=1.):
        super().__init__(value, prev_proba, norm)
        self.model = model
        self.prev_value = prev_value[..., np.newaxis]
        self.low_z = low_z[..., np.newaxis]
        self.high_z = high_z[..., np.newaxis]

        _s = self.prev_value.shape[:-1] + (self.size,)
        _ss = self.prev_value.shape[:-1] + (self.size+1,)
        self._integral = (np.empty(_s), np.empty(_s), np.empty(_s))
        self._delta = (np.empty(_ss), np.empty(_ss))
        self.model.cimpl.integrate(
                np.ravel(self.prev_value),
                np.ravel(self.low_z),
                np.ravel(self.high_z),
                np.ravel(self.voronoi),
                self._integral[0], self._integral[1], self._integral[2],
                self._delta[0], self._delta[1]
                )

    @lazy_property
    def _roots(self):
        return self.model._real_roots(self.prev_value, self.voronoi)
