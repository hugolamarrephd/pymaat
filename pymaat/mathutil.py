from collections import namedtuple
import numpy as np

from pymaat.nputil import workon_axis, diag_view

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

###########
# Voronoi #
###########


@workon_axis
def voronoi_1d(quantizer, lb=-np.inf, ub=np.inf):
    if quantizer.size == 0:
        raise ValueError
    shape = list(quantizer.shape)
    shape[0] += 1
    voronoi = np.empty(shape)
    voronoi[0] = lb
    voronoi[1:-1] = quantizer[:-1] + 0.5*np.diff(quantizer, n=1, axis=0)
    voronoi[-1] = ub
    return voronoi


@workon_axis
def inv_voronoi_1d(voronoi, *, first_quantizer=None, with_bounds=True):
    if voronoi.ndim > 2:
        raise ValueError("Does not support dimension greater than 2")
    if np.any(np.diff(voronoi, axis=0)<=0.):
        raise ValueError("Not strictly increasing Voronoi")
    if with_bounds:
        voronoi = voronoi[1:-1]  # Crop bounds, otherwise do nothing

    s = voronoi.shape
    broadcastable = (s[0],)+(1,)*(len(s)-1)
    # First, build (-1,+1) alternating vector
    alt_vector = np.empty((s[0],))
    alt_vector[::2] = 1.
    alt_vector[1::2] = -1.

    # Preliminary checks
    b = _get_first_quantizer_bounds(s, broadcastable, voronoi, alt_vector)
    if np.any(b[0]>=b[1]):
        raise ValueError("Has no inverse")
    if first_quantizer is None:
        if np.all(np.isfinite(b)):
            first_quantizer = 0.5 * (b[0] + b[1])
        else:
            raise ValueError("Could not infer first quantizer")
    elif np.any(first_quantizer>=b[1]) or np.any(first_quantizer<=b[0]):
        raise ValueError("Invalid first quantizer")

    # Initialize output
    inverse = np.empty((s[0]+1,)+s[1:])
    inverse[0] = first_quantizer  # May broadcast here
    # Solve using numpy matrix multiplication
    alt_matrix = np.empty((s[0], s[0]))
    for i in range(s[0]):
        diag_view(alt_matrix, k=-i)[:] = alt_vector[i]
    if voronoi.size > 0:
        inverse[1:] = 2.*np.dot(np.tril(alt_matrix), voronoi)
    # Correct for first element of quantizer
    inverse[1:] -= (np.reshape(alt_vector, broadcastable)
                    * inverse[np.newaxis, 0])
    assert np.all(np.diff(inverse, axis=0)>0.)
    return inverse


def _get_first_quantizer_bounds(s, broadcastable, voronoi, alt_vector):
    lb = []; ub = []
    term = np.cumsum(
            np.reshape(alt_vector, broadcastable)*voronoi,
            axis=0
            )

    for (i,v) in enumerate(voronoi):
        if i==0:
            ub.append(v)
        else:
            if i%2 == 0:
                ub.append(v-2.*term[i-1])
            else:
                lb.append(-v+2.*term[i-1])

    if len(lb) == 0:
        lb = -np.inf
    else:
        lb = np.max(np.array(lb), axis=0)

    if len(ub) == 0:
        ub = np.inf
    else:
        ub = np.min(np.array(ub), axis=0)

    return lb, ub
