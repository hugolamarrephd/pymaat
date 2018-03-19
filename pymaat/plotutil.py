import numpy as np
import math
from pymaat.mathutil import round_to_int

def get_lim(values, lb=None, ub=None, base=5, margin=0.1):
    lowest = round_to_int(np.amin(values)*(1.-margin),
            base=base, fcn=math.floor)
    highest = round_to_int(np.amax(values)*(1.+margin),
            base=base, fcn=math.ceil)
    if lb is not None:
        lowest = max(lb, lowest)
    if ub is not None:
        highest = min(ub, highest)
    return lowest, highest
