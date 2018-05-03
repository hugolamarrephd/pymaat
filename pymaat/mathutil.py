from collections import namedtuple
import numpy as np
from math import sqrt, pi


def round_to_int(x, base=1, fcn=round):
    return int(base * fcn(float(x)/base))


"""
 Interface
 values: list of length nper+1 containing np.arrays of shape (size[t],)
    with elements of length ndim representing stochastic process values
    (as a vector)
 probabilities: list of length nper+1 containing np.arrays
    of shape (size[t],) containing np.double representing
   probabilities
 transition_probabilities: list of length nper containing np.arrays
    of shape (size[t], size[t+1]) containing np.double representing
    transition probabilities
"""

MarkovChain = namedtuple('MarkovChain',
                         ['nper',
                          'ndim',
                          'sizes',
                          'values',
                          'probabilities',
                          'transition_probabilities'])


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

# Fast error function

def fast_erfc(x):
    x = x.astype(np.float32)
    neg = x < 0.
    z = np.absolute(x)
    t = 1.0/(1.0+0.5*z)
    out = t.copy()
    out *= 0.17087277
    out += -0.82215223
    out *= t
    out += 1.48851587
    out *= t
    out += -1.13520398
    out *= t
    out += 0.27886807
    out *= t
    out += -0.18628806
    out *= t
    out += 0.09678418
    out *= t
    out += 0.37409196
    out *= t
    out += 1.00002368
    out *= t
    out += -1.26551223
    out += -z*z
    out = np.exp(out)
    out *= t
    out[neg] *= -1.
    out[neg] += 2.
    return out


def fast_erf(x):  # Error function
    neg = x < 0.
    z = np.absolute(x)
    out = 1.-fast_erfc(z)
    out[neg] *= -1.
    return out



# Standard Normal Distribution

def fast_normcdf(x):
    return 0.5*(1.+fast_erf(x/sqrt(2.)))

def normpdf(x):
    return np.exp(-0.5*(x*x))/sqrt(2.*pi)
