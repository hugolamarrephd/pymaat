from collections import namedtuple
import numpy as np

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

