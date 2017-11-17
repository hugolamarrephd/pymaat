import numpy as np

class Garch():
    def __init__(self, mu=2, omega=1e-7, alpha=1e-7, beta=0.8, gamma=100):
        self.mu = mu
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def filter(self, return_timeseries, first_variance):
        self.check_variance(first_variance)
        return_timeseries = np.ascontiguousarray(return_timeseries)
        s = return_timeseries.shape
        innovations = np.empty(s, order='C')
        variance_timeseries = np.empty((s[0]+1,)+s[1:], order='C')
        variance_timeseries[0] = first_variance
        for (t,(r,h)) in enumerate(zip(return_timeseries, variance_timeseries)):
            (variance_timeseries[t+1], innovations[t]) = self.one_step_filter(r,h)
        return (variance_timeseries, innovations)


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

    def negative_log_likelihood(self, innovations, states):
        return 0.5 * (np.power(innovations, 2) + np.log(states))

    @staticmethod
    def check_variance(variances):
        if np.any(variances<=0):
            raise ValueError
