from collections import namedtuple
import numpy as np
from math import sqrt, pi


def round_to_int(x, base=1, fcn=round):
    return int(base * fcn(float(x)/base))


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
