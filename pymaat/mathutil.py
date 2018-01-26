import numpy as np

def voronoi_1d(quantizer, axis=0, lb=-np.inf, ub=np.inf):
    quantizer = np.asarray(quantizer)
    if quantizer.size==0:
        raise ValueError
    quantizer = np.swapaxes(quantizer,0,axis)
    shape = list(quantizer.shape)
    shape[0] += 1
    voronoi = np.empty(shape)
    voronoi[0] = lb
    voronoi[1:-1] = quantizer[:-1] + 0.5*np.diff(quantizer, n=1, axis=0)
    voronoi[-1] = ub
    voronoi = np.swapaxes(voronoi,0,axis)
    return voronoi
