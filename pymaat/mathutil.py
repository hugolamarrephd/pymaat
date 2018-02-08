import numpy as np

from pymaat.nputil import workon_axis, diag_view

###########
# Voronoi #
###########

@workon_axis
def voronoi_1d(quantizer, *, lb=-np.inf, ub=np.inf):
    if quantizer.size==0:
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
    if voronoi.ndim>2:
        raise ValueError("Does not support dimension greater than 2")
    if with_bounds:
        voronoi = voronoi[1:-1] # Crop bounds, otherwise do nothing
    if first_quantizer is None:
        first_quantizer = voronoi[0] - 0.5*(voronoi[1]-voronoi[0])
    s = voronoi.shape
    # Initialize output
    inverse = np.empty((s[0]+1,)+s[1:])
    inverse[0] = first_quantizer # May broadcast here
    # Solve using numpy matrix multiplication
    # First, build (-1,+1) alternating vector and matrice
    alt_vector = np.empty((s[0],))
    alt_vector[::2]=1.;  alt_vector[1::2]=-1.
    alt_matrix = np.empty((s[0],s[0]))
    for i in range(s[0]):
        diag_view(alt_matrix, k=-i)[:] = alt_vector[i]
    if voronoi.size>0:
        inverse[1:] = 2.*np.dot(np.tril(alt_matrix), voronoi)
    # Correct for first element of quantizer
    broadcastable = (s[0],)+(1,)*(len(s)-1)
    inverse[1:] -= (np.reshape(alt_vector, broadcastable)
            *inverse[np.newaxis,0])
    return inverse
