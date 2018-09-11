import numpy as np
from pymaat.nputil import diag_view

def inv_tridiagonal(A):
    d = diag_view(A)
    a = diag_view(A, 1)
    b = diag_view(A, -1)
    ab = a*b
    n = d.size
    # Pre-computations
    #   (1) Alpha
    alpha = np.empty((n+1,))
    alpha[0] = 1.
    alpha[1] = d[0]
    if n>2:
        for i in range(2, n+1):
            alpha[i] = d[i-1]*alpha[i-1] - ab[i-2]*alpha[i-2]
    #   (2) Beta
    beta = np.empty((n+1,))
    beta[0] = alpha[-1]
    beta[n] = 1.
    beta[n-1] = d[-1]
    if n>2:
        for i in range(n-2, 0, -1):
            beta[i] = d[i]*beta[i+1] - ab[i]*beta[i+2]
    # Invertible?
    if np.any(alpha==0.) or np.any(beta==0.):
        raise ValueError
    # Compute inverse
    T = np.empty_like(A)
    #   (1) Diagonal terms
    for i in range(0, n):
        denom = d[i]
        if i > 0:
            denom -= ab[i-1]*alpha[i-1]/alpha[i]
        if i < n-1:
            denom -= ab[i]*beta[i+2]/beta[i+1]
        T[i,i] = 1./denom
    D = diag_view(T)
    #   (2) Off-diagonal terms
    for i in range(n-1):
        cuma = np.cumprod(a[i:])
        cumb = np.cumprod(b[i:])
        # Alternating signs
        sign = np.empty_like(cuma)
        sign[::2] = -1.
        sign[1::2] = 1.
        #   (2.1) Upper diagonal terms
        T[i,i+1:] = sign*cuma*alpha[i]/alpha[i+1:-1]*D[i+1:]
        #   (2.1) Lower diagonal terms
        T[i+1:,i] = sign*cumb*beta[i+2:]/beta[i+1]*D[i]
    return T
