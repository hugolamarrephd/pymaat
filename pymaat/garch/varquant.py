from collections import namedtuple, defaultdict
from hashlib import sha1

import numpy as np
import scipy.optimize
from scipy.stats import norm

from pymaat.garch.quant import AbstractQuantization, MarkovChain
from pymaat.mathutil import voronoi_1d
from pymaat.util import lazy_property

class MarginalVariance(AbstractQuantization):

    def __init__(self, model, first_variance, size=100, first_probability=1.,
            nper=None):
        shape = self._get_shape(nper, size)
        first_quantizer = self._make_first_quantizer(
                first_variance,
                first_probability)
        super().__init__(model, shape, first_quantizer)

    def get_markov_chain(self):
        pass
        # sizes = []
        # values = []
        # probabilities = []
        # transition_probabilities = []
        # for q in self.all_quantizers:
        #     sizes.append(q.value.size)
        #     values.append(q.value)
        #     probabilities.append(q.probability)
        #     transition_probabilities.append(q.transition_probability)
        # return MarkovChain(nper=len(q), ndim=1,
        #         sizes=sizes,
        #         values=values,
        #         probabilities=probabilities,
        #         transition_probabilities=transition_probabilities)

    def quantize(self, t, values):
        pass

    @staticmethod
    def _make_first_quantizer(value, probability):
        # Set first values
        value = np.ravel(np.atleast_1d(value))
        # Set first probabilities
        probability = np.broadcast_to(probability, value.shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)
        return _QuantizerFactory.make_stub(value, probability)

    @staticmethod
    def _get_shape(nper, size):
        if nper is not None:
            shape = np.broadcast_to(size, (nper,))
        else:
            shape = np.atleast_1d(size)
        return shape

    def _one_step_optimize(self, size, previous):
        result = _QuantizerFactory(self.model, size, previous).optimize()
        return MarginalVariance._QuantizerFactory(
                value=result.value,
                probability=result.probability,
                transition_probability=result.transition_probability)


class _QuantizerFactory:

    def __init__(self, model, size, previous):
        self.model = model
        self.shape = (size,)
        self.size = size
        self.previous = previous
        self._cache = defaultdict()

    def optimize(self):
        # Perform optimization
        success, optim_x = self._perform_optimization()
        if not success:
            # Early failure if optimizer unsuccessful
            raise RuntimeError
        else:
            return self.make_unconstrained(optim_x)

    def make_unconstrained(self, x):
        x.flags.writeable = False
        key = sha1(x).hexdigest()
        quant = self._cache.get(key)
        if quant:
            return quant
        else:
            quant =_Unconstrained(self, x)
            self._cache[key] = quant
            return quant

    def make(self, value):
        return _Quantizer(self, value)

    @staticmethod
    def make_stub(value, probability):
        """
        Returns a minimal quantizer-like object which may be used
        as a previous quantizer when instantiating a subsequent
        factory.  It is most useful for initiating a recursion,
        ie for making the first time step quantizer
        """
        if np.any(np.isnan(value)) or np.any(~np.isfinite(value)):
            msg = 'Invalid value: NaNs or infinite'
            raise ValueError(msg)
        if np.any(value<=0.):
            msg = 'Invalid value: variance must be strictly positive'
            raise ValueError(msg)
        if np.abs(np.sum(probability)-1.)>1e-12 or np.any(probability<0.):
            msg = 'Probabilities must sum to one and be positive'
            raise ValueError(msg)
        fact = namedtuple( '_Quantizer',
            ['value', 'probability', 'size', 'shape'])
        return fact(value=value, probability=probability,
                size=value.size, shape=value.shape)

    def _perform_optimization(self, init=None):
        if init is None:
            init = np.zeros(self.size)
        func = lambda x: self.make_unconstrained(x).distortion
        jac = lambda x: self.make_unconstrained(x).gradient
        hess = lambda x: self.make_unconstrained(x).hessian
        minimizer_kwargs = {
                'method':'trust-ncg',
                'jac':jac,
                'hess':hess,
                'options':{'disp':False, 'maxiter':np.inf, 'gtol':1e-24}}
        sol = scipy.optimize.basinhopping(func, init,
                minimizer_kwargs=minimizer_kwargs)
        # Format output
        return True, sol.x

    # Utilities

    @lazy_property
    def min_variance(self):
        return self.model.get_lowest_one_step_variance(
                np.amin(self.previous.value))


class _Quantizer():

    def __init__(self, parent, value):
        if not isinstance(parent, _QuantizerFactory):
            msg = 'Quantizers must be instantiated from valid factory'
            raise ValueError(msg)
        else:
            self.model = parent.model
            self.size = parent.size
            self.shape = parent.shape
            self.previous = parent.previous
            self.min_variance = parent.min_variance

        value = np.asarray(value)
        if value.shape != self.shape:
            msg = 'Unexpected quantizer shape'
            raise ValueError(msg)
        elif ( np.any(np.isnan(value)) or np.any(~np.isfinite(value)) ):
            msg = 'Invalid quantizer'
            raise ValueError(msg)
        elif np.any(value<=self.min_variance):
            msg = ('Unexpected quantizer value(s): '
                    'must be strictly above minimum variance')
            raise ValueError(msg)
        elif np.any(np.diff(value)<=0.):
            msg = ('Unexpected quantizer value(s): '
                    'must be strictly increasing')
            raise ValueError(msg)
        else:
            self.value = value

    # Results
    @lazy_property
    def voronoi(self):
        """
        Returns a self.size+1 np.array representing the 1D voronoi tiles
        """
        return voronoi_1d(self.value, lb=0.)

    @property
    def probability(self):
        """
        Returns a self.size np.array containing the time-t probability
        """
        return self.previous.probability.dot(self.transition_probability)

    @property
    def transition_probability(self):
        """
        Returns a self.previous.size-by-self.size np.array
        containing the transition probabilities from time t-1 to time t
        """
        return self._integral[0]

    @property
    def distortion(self):
        """
        Returns a (scalar) distortion function to be optimized
        """
        elements = np.sum(self._distortion_elements, axis=1)
        return self.previous.probability.dot(elements)

    @property
    def gradient(self):
        """
        Returns self.size np.array representing the gradient
        of self.distortion
        """
        return -2. * self.previous.probability.dot(
                self._integral[1]
                - self._integral[0]
                *self.value[np.newaxis,:])

    @property
    def hessian(self):
        """
        Returns a self.size-by-self.size np.array containing the
        Hessian of self.distortion
        """
        def _get_diagonal():
            return -2. * self.previous.probability.dot(
                    self._integral_derivative[1]
                    -self._integral_derivative[0]
                    *self.value[np.newaxis,:]
                    -self._integral[0])
        def _get_off_diagonal():
            return -2. * self.previous.probability.dot(
                    self._integral_derivative_lagged[1]
                    -self._integral_derivative_lagged[0]
                    *self.value[np.newaxis,1:])
        # Build Hessian
        d = _get_diagonal()
        od = _get_off_diagonal()
        hess = np.diag(d) + np.diag(od, k=-1) + np.diag(od, k=1)
        return hess

    #############
    # Internals #
    #############

    @property
    def _distortion_elements(self):
        """
        Returns the un-weighted (self.previous.size-by-self.size)
        elements of the distortion to be summed, namely
        ```
            (\\mathcal{I}_{2,t}^{(ij)} - 2\\mathcal{I}_{1,t}^{(ij)} h_t^{(j)}
            + \\mathcal{I}_{0,t}^{(ij)} (h_t^{(j)})^2
        ```
        Rem. Output only has valid entries, ie free of NaNs
        """
        return (self._integral[2]
                - 2.*self._integral[1]*self.value[np.newaxis,:]
                + self._integral[0]*self.value[np.newaxis,:]**2.)

    @lazy_property
    def _integral(self):
        """
        Returns of len-3 list indexed by order and filled with
        self.previous.size-by-self.size arrays containing
        ```
            \\mathcal{I}_{order,t}^{(ij)}
        ```
        for all i,j.
        Rem. Output only has valid entries, ie free of NaNs
        """
        def _get_integral(order):
            integral = [self.model.one_step_expectation_until(
                    self.previous.value[:,np.newaxis],
                    self._roots[right], order=order,
                    pdf=self._pdf[right], cdf=self._cdf[right])
                    for right in [False, True]]
            out = integral[True] - integral[False]
            out = np.ma.filled(out, 0.)
            return np.diff(out, n=1, axis=1)
        return [_get_integral(order) for order in [0,1,2]]

    @property
    def _integral_derivative(self):
        """
        Returns a len-3 list indexed by order and filled
        previous.size-by-size arrays containing
        ```
            d\\mathcal{I}_{order,t}^{(ij)}/dh_t^{(j)}
        ```
        for all i,j.
        Rem. Output only has valid entries, ie free of NaNs
        """
        return [np.diff(self._delta[order], n=1, axis=1)
                for order in [0,1,2]]

    @property
    def _integral_derivative_lagged(self):
        """
        Returns a len-3 list indexed by order and filled with
        previous.size-by-size-1 arrays containing
        ```
            d\\mathcal{I}_{order,t}^{(ij)}/dh_t^{(j-1)}
        ```
        for all i and j>1.
        Rem. Output only has valid entries, ie free of NaNs
        """
        return [-self._delta[order][:,1:-1] for order in [0,1,2]]

    @lazy_property
    def _delta(self):
        """
        Returns a previous.size-by-size+1 array containing
        ```
            \\delta_{order,t}^{(ij\\pm)}
        ```
        for all i,j and a given order.
        Rem. Output only has valid entries, ie free of NaNs
        In particular, the limiting case where self.voronoi = +np.inf
        returns zero.
        """
        unsigned_root_derivative = \
                self.model.one_step_roots_unsigned_derivative(
                self.previous.value[:,np.newaxis],
                self.voronoi[np.newaxis,:])
        factor = (0.5*unsigned_root_derivative
                *(self._pdf[True]+self._pdf[False]))

        def _get_delta(order):
            out = factor*self.voronoi[np.newaxis,:]**order
            out = out.data
            #   When a root does not exist, delta is 0
            no_root = unsigned_root_derivative.mask
            out[no_root] = 0.
            out = np.ma.filled(out, 0.)
            # In the limiting case when voronoi is +np.inf (which
            #   always occur in the last column), delta is zero.
            out[:,-1] = 0.
            return out

        return [_get_delta(order) for order in [0,1,2]]

    @lazy_property
    def _pdf(self):
        """
        Returns a len-2 tuples containing (left, right) PDF of roots
        as a masked np array
        """
        def _get_pdf(roots):
            with np.errstate(invalid='ignore'):
                out = norm.pdf(roots.data)
                return np.ma.masked_array(out,
                        mask=roots.mask, copy=False)

        roots = self._roots
        return [_get_pdf(roots[right]) for right in [False, True]]

    @lazy_property
    def _cdf(self):
        """
        Returns a len-2 tuples containing (left, right) CDF roots
        as a masked np array
        """
        def _get_cdf(roots):
            with np.errstate(invalid='ignore'):
                out = norm.cdf(roots.data)
                return np.ma.masked_array(out,
                        mask=roots.mask, copy=False)

        roots = self._roots
        return [_get_cdf(roots[right]) for right in [False, True]]

    @lazy_property
    def _roots(self):
        """
        Returns a len-2 tuples containing (left, right) roots
        as a masked np array where the mask identifies non-existent
        roots
        """
        return self.model.one_step_roots(
                self.previous.value[:,np.newaxis],
                self.voronoi[np.newaxis,:])

class _Unconstrained(_Quantizer):

    def __init__(self, parent, x):
        if not isinstance(parent, _QuantizerFactory):
            msg = 'Quantizers must be instantiated from valid factory'
            raise ValueError(msg)
        x = np.asarray(x)
        self.x = x
        self._scaled_exp_x, value = self._change_of_variable(parent, x)
        # Let super catch invalid value...
        super().__init__(parent, value)

    @staticmethod
    def _change_of_variable(parent, x):
        scaled_exp_x = (np.exp(x)*parent.min_variance/(parent.size+1.))
        value = (parent.min_variance + np.cumsum(scaled_exp_x))
        return (scaled_exp_x, value)

    @property
    def gradient(self):
        """
        Returns the transformed gradient for the distortion function, ie
        dDistortion/ dx = dDistortion/dGamma * dGamma/dx
        """
        grad = super().gradient
        return grad.dot(self._jacobian)

    @property
    def hessian(self):
        """
        Returns the transformed Hessian for the distortion function
        """
        s = super()
        def get_first_term():
            jac = self._jacobian
            hess = s.hessian
            return jac.T.dot(hess).dot(jac)

        def get_second_term():
            inv_cum_grad = np.flipud(np.cumsum(np.flipud(s.gradient)))
            return np.diag(self._scaled_exp_x * inv_cum_grad)

        return get_first_term() + get_second_term()

    @property
    def _jacobian(self):
        """
        Returns the self.size-by-self.size np.array containing the jacobian
        of the change of variable, ie
        ```
        \\nabla_{ij} \\Gamma_{h,t}
        ```
        """
        mask = np.tril(np.ones((self.size, self.size)))
        return self._scaled_exp_x[np.newaxis,:]*mask
