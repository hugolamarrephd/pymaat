import numpy as np
import math
from pymaat.mathutil import round_to_int

def get_lim(*args,
        bounds=[-np.inf, np.inf],
        base=None,
        precision=None,
        margin=0.,
        formatter=lambda x:x):
    args = [np.ravel(a) for a in args]
    x = np.concatenate(args)
    x = formatter(x)
    ptp = np.ptp(x)
    lim = [np.amin(x)-ptp*margin, np.amax(x)+ptp*margin]
    if base is not None:
        lim[0] = round_to_int(lim[0], base=base, fcn=math.floor)
        lim[1] = round_to_int(lim[1], base=base, fcn=math.ceil)
    elif precision is not None:
        lim[0] = round_to_int(
                lim[0]*10.**precision,
                fcn=math.floor
                )*10.**-precision
        lim[1] = round_to_int(
                lim[1]*10.**precision,
                fcn=math.ceil
                )*10.**-precision
    lim[0] = max(bounds[0], lim[0])
    lim[1] = min(bounds[1], lim[1])
    return np.array(lim)
