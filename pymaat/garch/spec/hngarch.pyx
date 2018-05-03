# cython: linetrace=True
# cython: binding=True

import numpy as np

from pymaat.garch.model import AbstractOneLagGarch
from pymaat.garch.spec.ret import CentralRiskPremium
from pymaat.nputil import flat_view, forced_elbyel

from cython cimport boundscheck, wraparound

class HestonNandiGarch(AbstractOneLagGarch):

    def __init__(self, mu, omega, alpha, beta, gamma):
        super().__init__(CentralRiskPremium(mu))
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if (1 - self.beta - self.alpha*self.gamma**2 <= 0):
            raise ValueError


    def _equation(self, innovations, variances, volatilities):
        """
         ` h_{t+1} = omega + beta*h_{t} + alpha*(z_t-gamma*sqrt(h_{t}))^2`
        """
        return (self.omega + self.beta*variances
                + self.alpha*np.power(innovations-self.gamma*volatilities,2))

    def _real_roots(self, variances, next_variances):
        discr = (next_variances-self.omega-self.beta*variances)/self.alpha
        const = self.gamma * np.sqrt(variances)
        sqrtdiscr = np.maximum(discr, 0.)**0.5  # Real part only
        return [const + (pm * sqrtdiscr) for pm in [-1., 1.]]

    def _real_roots_unsigned_derivative(self, variances, next_variances):
        discr = next_variances-self.omega-self.beta*variances
        sqrtdiscr = (self.alpha*np.maximum(discr, 0.))**0.5  # Real part only
        out = np.zeros_like(sqrtdiscr)
        out[sqrtdiscr>0.] = 0.5 / sqrtdiscr[sqrtdiscr>0.]
        return out

    def _first_order_integral_factors(self, variances, innovations):
        # PDF
        with np.errstate(invalid='ignore'):
            # Infinite innovations are silently ignored here
            pdf_factor = 2*self.gamma*np.sqrt(variances)-innovations
            pdf_factor *= self.alpha
        # CDF
        cdf_factor = (self.omega + self.alpha
                + (self.beta+self.alpha*self.gamma**2) * variances)
        return (pdf_factor, cdf_factor)

    def _second_order_integral_factors(self, variances, innovations):
        # Preliminary computations
        z = innovations
        gv = self.gamma*variances**0.5
        gv2 = gv**2.
        minh = self.beta*variances + self.omega
        # PDF
        with np.errstate(invalid='ignore'):
            # Infinite innovations are silently ignored here
            z2 = z*z
            pdf_factor = 2.*gv2*(2.*gv-3.*z) + 4.*gv*(z2+2.) - z*(z2+3.)
            pdf_factor *= self.alpha
            pdf_factor +=  2.*(2.*gv-z)*minh
            pdf_factor *= self.alpha
        # CDF
        cdf_factor = self.alpha**2.*(gv2*(gv2+6.)+3.)
        cdf_factor += 2.*self.alpha*(gv2+1.)*minh
        cdf_factor += minh**2.
        return (pdf_factor, cdf_factor)

##########
# CYTHON #
##########

# @boundscheck(False)
# @wraparound(False)
# cpdef c_one_step_roots_unsigned_derivative(
#         double omega, double alpha, double beta,
#         double[:] variances,
#         double[:] next_variances,
#         double[:] result):

#     cdef:
#         unsigned int i, n
#     n = variances.size
#     for i in range(n):
#         result[i] = 0.5*(alpha*(
#                 next_variances[i]-omega-beta*variances[i]))**-0.5
