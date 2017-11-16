import numpy as np

class Garch():
    def __init__(self, mu=2, omega=1e-7, alpha=1e-7, beta=0.8, gamma=100):
        self.mu = mu
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def one_step_filter(self, returns, variances):
        volatility = np.sqrt(variances)
        innovations = (returns - (self.mu-0.5) * variances)/volatility
        next_variances = ( self.omega + self.beta * variances + self.alpha *
            np.power(innovations - self.gamma * volatility,2) )
        return (next_variances, innovations)

    def one_step_simulate(self, innovations, variances):
        volatility = np.sqrt(variances)
        returns = (self.mu-0.5) * variances + volatility * innovations
        next_variances = ( self.omega + self.beta * variances + self.alpha *
            np.power(innovations-self.gamma * volatility,2) )
        return (next_variances, returns)
