import numpy as np

from pymaat.garch.model import AbstractOneLagGarch

class HestonNandiGarch(AbstractOneLagGarch):

    def __init__(self, mu, omega, alpha, beta, gamma):
        self.mu = mu
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if (1 - self.beta - self.alpha*self.gamma**2 <= 0):
            raise ValueError

    # Return Specification

    def _one_step_return_equation(self, innovations, variances, volatilities):
        """
            `r_t = (mu - 0.5)*h_{t} + sqrt(h_{t}) * z_{t}`
        """
        return (self.mu-0.5)*variances + volatilities*innovations

    def _one_step_innovation_equation(self, returns, variances, volatilities):
        return (returns-(self.mu-0.5)*variances)/volatilities

    # Variance Specification

    def _one_step_equation(self, innovations, variances, volatilities):
        """
         ` h_{t+1} = omega + beta*h_{t} + alpha*(z_t-gamma*sqrt(h_{t}))^2`
        """
        return (self.omega + self.beta*variances
                + self.alpha*np.power(innovations-self.gamma*volatilities,2))

    def _get_one_step_roots(self, variances, next_variances):
        discr = (next_variances-self.omega-self.beta*variances)/self.alpha
        const = self.gamma * np.sqrt(variances)
        with np.errstate(invalid='ignore'):
            roots = [const + (pm * discr**0.5) for pm in [-1., 1.]]
        return roots

    def _get_one_step_roots_unsigned_derivative(self,
            variances, next_variances):
        factor = self.alpha*(next_variances-self.omega-self.beta*variances)
        with np.errstate(divide='ignore', invalid='ignore'):
            return 0.5*factor**-0.5

    def _get_one_step_first_order_expectation_factors(self,
            variances, innovations):
        # PDF
        with np.errstate(invalid='ignore'):
            # Infinite innovations are silently ignored here
            pdf_factor = 2*self.gamma*np.sqrt(variances)-innovations
            pdf_factor *= self.alpha
        # CDF
        cdf_factor = (self.omega + self.alpha
                + (self.beta+self.alpha*self.gamma**2) * variances)
        return (pdf_factor, cdf_factor)

    def _get_one_step_second_order_expectation_factors(self,
            variances, innovations):
        # Preliminary computations
        gamma_vol = self.gamma*variances**0.5
        gamma_vol_squared = gamma_vol**2.
        betah_plus_omega = self.beta*variances + self.omega
        # PDF
        with np.errstate(invalid='ignore'):
            # Infinite innovations are silently ignored here
            innovations_squared = innovations**2.
            pdf_factor = (self.alpha
                        *(2.*gamma_vol_squared*(2.*gamma_vol-3.*innovations)
                        + 4.*gamma_vol*(innovations_squared+2.)
                        - innovations*(innovations_squared+3.))
                        + 2.*(2.*gamma_vol-innovations)*betah_plus_omega)
            pdf_factor *= self.alpha
        # CDF
        cdf_factor = (self.alpha**2.
                *(gamma_vol_squared*(gamma_vol_squared+6.)+3.)
                + 2.*self.alpha*(gamma_vol_squared+1.)*betah_plus_omega
                + betah_plus_omega**2.)
        return (pdf_factor, cdf_factor)
