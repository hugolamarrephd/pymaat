from functools import partial, wraps, lru_cache
from abc import ABC, abstractmethod
import collections

import numpy as np
import scipy.optimize
from scipy.stats import norm

from pymaat.nputil import printoptions
from pymaat.util import lazy_property
from pymaat.mathutil import voronoi_1d

class AbstractQuantization(ABC):

    def __init__(self, model, shape, first_quantizer):
        self.model = model
        self.shape = shape
        self.first_quantizer = first_quantizer

    def optimize(self, **kwargs):
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False
        current_quantizer = self.first_quantizer
        # Perform recursion
        self.all_quantizers = []
        for (t,s) in enumerate(self.shape):
            if verbose:
                header = "[t={0:d}] Searching for optimal quantizer".format(t)
                print("\n" + ("#"*len(header)) + "\n")
                print(header, end='')
            # Gather results from previous iteration...
            self.all_quantizers.append(current_quantizer)
            # Advance to next time step (computations done here...)
            current_quantizer = self._one_step_optimize(
                    s, current_quantizer, **kwargs)
            if verbose:
                print("\n[Optimal distortion]\n{0:.6e}".format(
                    current_quantizer.distortion))
                print("\n[Optimal quantizer (%/y)]")
                with printoptions(precision=2, suppress=True):
                    print(np.sqrt(252.*current_quantizer.value)*100.)
        # Handle last quantizer
        self.all_quantizers.append(current_quantizer)
        if verbose:
            print("\n" + ("#"*len(header)) + "\n")
        return self #for convenience

    @abstractmethod
    def quantize(self, t, values):
        pass

    @abstractmethod
    def _one_step_optimize(self, shape, previous):
        pass

class ConditionalOnVariance(AbstractQuantization):
    # Quantizes marginal variance first and then
    # quantizes price *conditional* on the optimal variance quantization

    _QuantizerFactory = collections.namedtuple(
            'ConditionalOnVarianceQuantizer',
            ['price_value',
            'variance_quantizer',
            'probability',
            'transition_probability',
            ])

    def __init__(self, model, first_variance,
            variance_size=100, price_size=100,
            first_price=1., first_probability=1.,
            nper=None):

        shape = self._get_shape(nper, variance_size, price_size)
        first_quantizer = self._make_first_quantizer(
                first_variance,
                first_price,
                first_probability)

        super(ConditionalOnVariance, self).__init__(
               model, shape, first_quantizer)

    # def get_markov_chain(self):
    #     pass
        # # TODO: optimize memory
        # sizes = []; values = []; probabilities=[]; transition_probabilities=[]
        # for q in self.all_quantizers:
        #     # Sizes
        #     sizes.append(q.price_value.size)
        #     # Values
        #     tmp_value = np.empty(q.price_value.shape,
        #             dtype=[('price',np.float_),('variance',np.float_)])
        #     tmp_value['price'] = q.price_value
        #     tmp_value['variance'] = q.variance_quantizer.value[:,np.newaxis]
        #     values.append(tmp_value.flatten())
        #     # Probabilities
        #     probabilities.append(q.probability.flatten())
        #     # Transition probabilities
        #     tmp_trans_proba = np.reshape(q.transition_probability,
        #             (sizes[-2], sizes[-1])) #TODO: Double check this line
        #     transition_probabilities.append(tmp_trans_proba)

        # return MarkovChain(nper=len(q), ndim=2, sizes=sizes, values=values,
        #         probabilities=probabilities,
        #         transition_probabilities=transition_probabilities)

    def quantize(self, t, values):
        pass

    # Internals
    @staticmethod
    def _get_shape(nper, variance_size, price_size):
        if nper is not None:
            variance_size = np.broadcast_to(variance_size, (nper,))
            price_size = np.broadcast_to(price_size, (nper,))
        else:
            variance_size = np.atleast_1d(variance_size)
            price_size = np.atleast_1d(price_size)
        shape = list(zip(variance_size, price_size))
        return shape

    @staticmethod
    def _make_first_quantizer(variance, price, probability):
        # Set first variance
        variance = np.ravel(np.atleast_1d(variance))
        variance_size = variance.size # (j)

        # Set first price
        price = np.atleast_1d(price)
        if price.ndim==1:
            price = price[:,np.newaxis]
        shape = (variance_size, price.shape[1])
        price_value = np.broadcast_to(price, shape)# (j,m)

        # Set first probabilities
        probability = np.broadcast_to(probability, shape) # (j,m)
        # ...and quietly normalize
        probability = probability/np.sum(probability)

        # Set first variance quantizer
        variance_probability = np.sum(probability, axis=1)
        variance_quantizer = MarginalVariance._make_first_quantizer(
                variance,
                variance_probability)

        return ConditionalOnVariance._QuantizerFactory(
                price_value=price_value,
                variance_quantizer=variance_quantizer,
                probability=probability,
                transition_probability=None)

    @staticmethod
    def _one_step_optimize(self, shape, previous):
        variance_size = shape[0]
        price_size = shape[1]

        # Get optimal variance quantization first...
        optimize_variance = _MarginalVarianceQuantizerFactory(
                self.model,
                variance_size,
                previous.variance_quantizer)
        variance_state = optimize_variance()

        # Perform asset price optimization for each next variances
        # TODO: Parallelize!
        all_price_states = [None]*variance_size
        for j in range(variance_size):
            optimize = _ConditionalOnVarianceElementOptimizer(
                    self.model,
                    price_size,
                    previous,
                    variance_state,
                    j)
            all_price_states[j] = optimize()

        # Join processes
        # Merge & format results
        shape = (variance_size, price_size)
        price_value = np.empty(shape)
        probability = np.empty(shape)
        transition_probability = np.empty(previous.shape+shape)
        for j in range(variance_size):
            price_value[j] =  all_price_states[j].value
            probability[j] =  all_price_states[j].probability
            transition_probability[:,:,j,:] = \
                 all_price_states[j].transition_probability

        variance_quantizer = MarginalVariance._QuantizerFactory(
                value = variance_state.value,
                probability = variance_state.probability,
                transition_probability = variance_state.transition_probability)

        return ConditionalOnVariance._QuantizerFactory(
                price_value=price_value,
                variance_quantizer=variance_quantizer,
                probability=probability,
                transition_probability=transition_probability)


###################################
# Conditional Quantizer Internals #
###################################

class _ConditionalOnVarianceElementOptimizer():

    def __init__(self, model, size, previous, variance_state, index):
        self.model = model
        self.size = size
        self.previous = previous
        self.variance_state = variance_state
        self.index = index
        assert (self.index>=0 and self.index<self.variance_state.parent.size)

    def __call__(self):
        success, optim_x = self._perform_optimization()
        if not success:
            # Early failure if optimizer unsuccessful
            raise RuntimeError
        else:
            return self.eval_at(optim_x)

    def _perform_optimization(self, init=None, try_number=0):
        # TODO return success==false when try_number reach MAX
        if init is None:
            init = np.zeros(self.size)

        # We let the minimization run until the predicted reduction
        #   in distortion hits zero (or less) by setting gtol=0 and
        #   maxiter=inf. We hence expect a status flag of 2.
        sol = scipy.optimize.minimize(
                lambda x: self.eval_at(x).distortion,
                init,
                method='trust-ncg',
                jac=lambda x: self.eval_at(x).transformed_gradient,
                hess=lambda x: self.eval_at(x).transformed_hessian,
                options={'disp':False, 'maxiter':np.inf, 'gtol':0.})

        success = sol.status==2 or sol.status==0

        # Format output
        return success, sol.x

    #TODO: profile caching (is it really necessary?)
    def eval_at(self, x):
        # Make sure array is hashable for lru_cache
        x.flags.writeable = False
        return _ConditionalOnVarianceElementState(self, x)

    @lazy_property
    def branch_probability(self): # 2 * (i)
        j = self.index
        pdf = self.variance_state.pdf
        norm_factor = (pdf[0][:,j]+pdf[1][:,j])
        out = [p[:,j]/norm_factor for p in pdf]
        # TODO: send to tests: assert np.all(out[0]+out[1]==1.)
        return out

    @lazy_property
    def conditional_prices(self):# returns ((i,l),(i,l))
        j = self.index
        returns = [self.model._one_step_return_equation(z[:,j],
                self.previous.variance_quantizer.value,
                self.previous_volatility)
                for z in self.variance_state.roots]
        out = [self.previous.price_value*np.exp(r[:,np.newaxis])
                for r in returns]
        return out

    @lazy_property
    def conditional_probability(self): # (i,l)
        j = self.index
        out = (self.variance_state.transition_probability[:,j,np.newaxis]*
                self.previous.probability)
        return out

    @lazy_property
    def previous_volatility(self): # returns (i)
        out = np.sqrt(self.previous.variance_quantizer.value)
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def max_price(self):
        return np.amax(self.conditional_prices[1])

    @lazy_property
    def min_price(self):
        return np.amin(self.conditional_prices[0])


class _ConditionalOnVarianceElementState():

    def __init__(self, parent, x):
        self.parent = parent
        self.x = x
        assert (x.ndim == 1 and self.x.shape == (self.parent.size,))

    @lazy_property
    def value(self):
        out = self.parent.min_price + np.cumsum(self.all_exp_x)
        assert np.all(np.isfinite(out))
        assert np.all(out>0)
        assert np.all(np.diff(out)>0)
        return out

    @lazy_property
    def all_exp_x(self):
        span = self.parent.max_price-self.parent.min_price
        return np.exp(self.x)*span/(self.parent.size+1)

    @property
    def probability(self):
        return self.hessian_diagonal/2.

    @property
    def transition_probability(self):
        out = ((self.weights[0] + self.weights[1]) # (i,l,m)
                / self.parent.previous.probability[:,:,np.newaxis])
        return out

    @property
    def distortion(self):
        out = (self.weights[0]
                * (self.parent.conditional_prices[0][:,:,np.newaxis]
                - self.value[np.newaxis,np.newaxis,:])**2.
            + self.weights[1]
                * (self.parent.conditional_prices[1][:,:,np.newaxis]
                - self.value[np.newaxis,np.newaxis,:])**2.) #(i,l,m)
        return out.sum()

    @property
    def transformed_gradient(self):
        out = self.gradient.dot(self.jacobian)
        assert not np.any(np.isnan(out))
        return out

    @property
    def gradient(self): # m
        out = -2.*(self.weights[0]
                * (self.parent.conditional_prices[0][:,:,np.newaxis]
                - self.value[np.newaxis,np.newaxis,:])
                + self.weights[1]
                * (self.parent.conditional_prices[1][:,:,np.newaxis]
                - self.value[np.newaxis,np.newaxis,:]))
        return np.ravel(out.sum(axis=(0,1))) # m

    @property
    def transformed_hessian(self):
        hessian = np.diag(self.hessian_diagonal)
        out = (self.jacobian.T.dot(hessian).dot(self.jacobian)
                + self.hessian_correction)
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def hessian_diagonal(self): # m
        out = 2.*(self.weights[0] + self.weights[1])
        return np.ravel(out.sum(axis=(0,1))) # m

    @property
    def jacobian(self):
        mask = np.tril(np.ones((self.parent.size, self.parent.size)))
        out = self.all_exp_x[np.newaxis,:]*mask
        assert not np.any(np.isnan(out))
        return out

    @property
    def hessian_correction(self):
        out = np.zeros((self.parent.size,self.parent.size))
        mask = np.empty(self.parent.size)
        for i in range(self.parent.size):
            mask[:] = 0.
            mask[:i+1] = 1.
            out += self.gradient[i]*np.diag(mask*self.all_exp_x)
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def weights(self): # 2*(i,l,m)
        cond_proba = self.parent.conditional_probability[:,:,np.newaxis]
        out = [cond_proba*p[:,np.newaxis,np.newaxis]*I
                for (p,I) in zip(
                    self.indicator,
                    self.parent.branch_probability)]
        return out

    @property
    def indicator(self):# 2 * (i,l,m)
        # TODO: more robust treatment of nans
        # TODO: optimize memory
        j = self.parent.index
        roots = self.roots # only compute once
        out = [np.logical_and(
                roots[:,:,:-1] < z[:,j,np.newaxis,np.newaxis],
                z[:,j,np.newaxis,np.newaxis] < roots[:,:,1:])
                for z in self.parent.variance_state.roots] # 2 * (i,l,m)
        return out

    @property
    def roots(self): # (i,l,m+1)
        ret = (self.voronoi[np.newaxis,np.newaxis,:]
                /self.parent.previous.price_value[:,:,np.newaxis])
        var = (self.parent.previous
                .variance_quantizer.value[:,np.newaxis,np.newaxis])
        vol = self.parent.previous_volatility[:,np.newaxis,np.newaxis]
        out = self.parent.model._one_step_return_root(ret, var, vol)
        assert not np.any(np.isnan(out))
        return out

    @property
    def voronoi(self):# returns (m+1)
        out = voronoi_1d(self.value, lb=0.)
        return out
