import numpy as np

class Garch():
    def __init__(self, mu, omega, alpha, beta, gamma):
        self.mu = mu
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if (1 - self.beta - self.alpha*self.gamma**2 <= 0):
            raise ValueError

    def filter(self, ret_ts, first_var):
        self._check_variance(first_var)

        innov_ts = np.empty_like(ret_ts)
        var_ts = self._init_var_ts(ret_ts, first_var)

        for (t,(r,h)) in enumerate(zip(ret_ts, var_ts)):
            (var_ts[t+1], innov_ts[t]) = self.one_step_filter(r,h)

        return (var_ts, innov_ts)

    def simulate(self, innov_ts, first_var):
        self._check_variance(first_var)

        ret_ts = np.empty_like(innov_ts)
        var_ts = self._init_var_ts(innov_ts, first_var)

        for (t,(z,h)) in enumerate(zip(innov_ts, var_ts)):
            (var_ts[t+1], ret_ts[t]) = self.one_step_simulate(z,h)

        return (var_ts, ret_ts)

    def one_step_filter(self, ret, var):
        vol = np.sqrt(var)
        innov = (ret - (self.mu-0.5) * var)/vol
        next_var = ( self.omega + self.beta * var + self.alpha *
            np.power(innov - self.gamma * vol,2) )
        return (next_var, innov)

    def one_step_simulate(self, innov, var):
        vol = np.sqrt(var)
        ret = (self.mu-0.5) * var + vol * innov
        next_var = ( self.omega + self.beta * var + self.alpha *
            np.power(innov-self.gamma * vol,2) )
        return (next_var, ret)

    def one_step_roots(self, var, next_var):
        if np.any(next_var<=self.get_lowest_next_variance(var)):
            raise ValueError
        d = np.sqrt((next_var - self.omega - self.beta*var) / self.alpha)
        a = self.gamma * np.sqrt(var)
        innov_left = a-d
        innov_right = a+d
        return (innov_left, innov_right)

    def get_lowest_next_variance(self, var):
        return self.omega + self.beta * var

    def negative_log_likelihood(self, innov, var):
        return 0.5 * (np.power(innov, 2) + np.log(var))

    @staticmethod
    def _check_variance(var):
        if np.any(var<=0):
            raise ValueError

    @staticmethod
    def _init_var_ts(like, first_var):
        var_shape = (like.shape[0]+1,) + like.shape[1:]
        var_ts = np.empty(var_shape)
        var_ts[0] = first_var
        return var_ts

class Quantizer():
    def __init__(self, model, init_innov, first_var):
        self.model = model

        if init_innov.ndim == 2:
            self.n_per = init_innov.shape[0]+1
            self.n_quant = init_innov.shape[1]
        else:
            raise ValueError

        (h,*_) = self.model.simulate(init_innov, first_var)
        self._set_gamma(h)
        #TODO set initial transition probas

    def quantize(self):
        pass

    def _set_gamma(self, gamma, t=None):
        if t is None:
            self.gamma = gamma
            self.gamma.sort()
            self.voronoi = self.gamma[:,:-1] + np.diff(self.gamma)
        else:
            self.gamma[t,:] = gamma
            self.gamma[t,:].sort()
            self.voronoi[t,:] = self.gamma[t,:-1] + np.diff(self.gamma[t,:])



