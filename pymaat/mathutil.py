from collections import namedtuple
import numpy as np
from scipy.stats import norm
from math import sqrt, pi
from pymaat.mathutil_c import normcdf as normcdf_c
from pymaat.mathutil_c import normpdf as normpdf_c
from pymaat.nputil import atleast_1d


def round_to_int(x, base=1, fcn=round):
    return int(base * fcn(float(x)/base))


@atleast_1d
def normcdf(x):
    out = np.empty_like(x)
    normcdf_c(x, out)
    return out

@atleast_1d
def normpdf(x):
    out = np.empty_like(x)
    normpdf_c(x, out)
    return out

@atleast_1d
def norminv(p):
    # TODO improve performance here
    return norm.ppf(p)

#####################
# Special functions #
#####################

# Logistic

def logistic(x):
    return 1./(1.+np.exp(-x))

def dlogistic(x):
    return logistic(x)*(1.-logistic(x))

def ddlogistic(x):
    return dlogistic(x)*(1.-2.*logistic(x))

def ilogistic(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(x/(1.-x))
