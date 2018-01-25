# All Marginal Garch Quantizers

from functools import partial, wraps, lru_cache
from abc import ABC, abstractmethod
import collections

import numpy as np
import scipy.optimize
from scipy.stats import norm

#TODO: move to semantic utilities
class lazy_property:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            # replace method by property...
            setattr(instance, self.func.__name__, value)
            return value

#TODO: move to math utilities
def get_voronoi_1d(quantizer, axis=0, lb=-np.inf, ub=np.inf):
    if quantizer.size==0:
        raise ValueError
    quantizer = np.swapaxes(quantizer,0,axis)
    shape = list(quantizer.shape)
    shape[0] += 1
    voronoi = np.empty(shape)
    voronoi[0] = lb
    voronoi[1:-1] = quantizer[:-1] + 0.5*np.diff(quantizer, n=1, axis=0)
    voronoi[-1] = ub
    voronoi = np.swapaxes(voronoi,0,axis)
    return voronoi

####################################################
MAX_VAR_QUANT_TRY = 5

#
# Interface
# values: list of length nper+1 containing np.arrays of shape (size[t],)
#    with elements of length ndim representing stochastic process values
#    (as a vector)
# probabilities: list of length nper+1 containing np.arrays
#    of shape (size[t],) containing np.double representing
#   probabilities
# transition_probabilities: list of length nper containing np.arrays
#    of shape (size[t], size[t+1]) containing np.double representing
#    transition probabilities
#
MarkovChain = collections.namedtuple('MarkovChain',
            ['nper', 'ndim',
            'sizes',
            'values',
            'probabilities',
            'transition_probabilities'])

class AbstractQuantization(ABC):

    def __init__(self, model, shape, first_quantizer):
        self.model = model
        self.shape = shape
        self.first_quantizer = first_quantizer

    def optimize(self):
        current_quantizer = self.first_quantizer
        # Perform recursion
        self.all_quantizers = []
        for s in self.shape:
            # Gather results from previous iteration...
            self.all_quantizers.append(current_quantizer)
            # Advance to next time step (computations done here...)
            current_quantizer = self._one_step_optimize(
                    self.model, s, current_quantizer)
        # Handle last quantizer
        self.all_quantizers.append(current_quantizer)
        return self #for convenience

    @abstractmethod
    def get_markov_chain(self):
        pass

    @abstractmethod
    def quantize(self, t, values):
        pass

    @abstractmethod
    def _one_step_optimize(self, shape, previous_quantizer):
        pass

class ConditionalOnVarianceQuantization(AbstractQuantization):
    # Quantizes marginal variance first and then
    # quantizes price *conditional* on the optimal variance quantization

    Quantizer = collections.namedtuple('ConditionalOnVarianceQuantizer',
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
        first_quantizer = self._get_first_quantizer(
                first_variance,
                first_price,
                first_probability)

        super(ConditionalOnVarianceQuantization, self).__init__(
               model, shape, first_quantizer)

    def get_markov_chain(self):
        # TODO: optimize memory
        sizes = []; values = []; probabilities=[]; transition_probabilities=[]
        for q in self.all_quantizers:
            # Sizes
            sizes.append(q.price_value.size)
            # Values
            tmp_value = np.empty(q.price_value.shape,
                    dtype=[('price',np.float_),('variance',np.float_)])
            tmp_value['price'] = q.price_value
            tmp_value['variance'] = q.variance_quantizer.value[:,np.newaxis]
            values.append(tmp_value.flatten())
            # Probabilities
            probabilities.append(q.probability.flatten())
            # Transition probabilities
            tmp_trans_proba = np.reshape(q.transition_probability,
                    (sizes[-2], sizes[-1])) #TODO: Double check this line
            transition_probabilities.append(tmp_trans_proba)

        return MarkovChain(nper=len(q), ndim=2, sizes=sizes, values=values,
                probabilities=probabilities,
                transition_probabilities=transition_probabilities)

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
    def _get_first_quantizer(variance, price, probability):
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
        variance_quantizer = MarginalVariance._get_first_quantizer(
                variance,
                variance_probability)

        return ConditionalOnVariance.Quantizer(
                price_value=price_value,
                variance_quantizer=variance_quantizer,
                probability=probability,
                transition_probability=None)

    @staticmethod
    def _one_step_optimize(self, shape, previous_quantizer):
        variance_size = shape[0]
        price_size = shape[1]

        # Get optimal variance quantization first...
        optimize_variance = _MarginalVarianceOptimizer(
                self.model,
                variance_size,
                previous_quantizer.variance_quantizer)
        variance_state = optimize_variance()

        # Perform asset price optimization for each next variances
        # TODO: Parallelize!
        all_price_states = [None]*variance_size
        for j in range(variance_size):
            optimize = _ConditionalOnVarianceElementOptimizer(
                    self.model, price_size, previous_quantizer, variance_state, j)
            all_price_states[j] = optimize()

        # Join processes
        # Merge & format results
        shape = (variance_size, price_size)
        price_value = np.empty(shape)
        probability = np.empty(shape)
        transition_probability = np.empty(previous_quantizer.shape+shape)
        for j in range(variance_size):
            price_value[j] =  all_price_states[j].value
            probability[j] =  all_price_states[j].probability
            transition_probability[:,:,j,:] = \
                 all_price_states[j].transition_probability

        variance_quantizer = MarginalVarianceQuantization.Quantizer(
                value = variance_state.value,
                probability = variance_state.probability,
                transition_probability = variance_state.transition_probability)

        return ConditionalOnVariance.Quantizer(
                price_value=price_value,
                variance_quantizer=variance_quantizer,
                probability=probability,
                transition_probability=transition_probability)


class MarginalVarianceQuantization():

    Quantizer = collections.namedtuple('MarginalVarianceQuantizer',
            ['value',
            'probability',
            'transition_probability'])

    def __init__(self, model, first_value,
            size=100, first_probability=1.,
            nper=None):
        shape = self._get_shape(nper, size)
        first_quantizer = self._get_first_quantizer(
                first_value, first_probability)
        super(MarginalVarianceQuantization, self).__init__(
                model, shape, first_quantizer)

    def get_markov_chain(self):
        sizes = []
        values = []
        probabilities = []
        transition_probabilities = []
        for q in self.all_quantizers:
            sizes.append(q.value.size)
            values.append(q.value)
            probabilities.append(q.probability)
            transition_probabilities.append(q.transition_probability)
        return MarkovChain(nper=len(q), ndim=1,
                sizes=sizes,
                values=values,
                probabilities=probabilities,
                transition_probabilities=transition_probabilities)

    def quantize(self, t, values):
        pass

    @staticmethod
    def _get_first_quantizer(value, probability):
        # Set first values
        value = np.ravel(np.atleast_1d(value))
        # Set first probabilities
        probability = np.broadcast_to(probability, value.shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)
        return MarginalVariance.Quantizer(
                value=value,
                probability=probability,
                transition_probability=None)

    @staticmethod
    def _get_shape(nper, size):
        if nper is not None:
            shape = np.broadcast_to(size, (nper,))
        else:
            shape = np.atleast_1d(size)
        return shape

    def _one_step_optimize(self, size, previous_quantizer):
        optimize = _MarginalVarianceOptimizer(self.model, size, previous_quantizer)
        result = optimize()
        return MarginalVariance.Quantizer(
                value=result.value,
                probability=result.probability,
                transition_probability=result.transition_probability)

###################################
# Conditional Quantizer Internals #
###################################

class _ConditionalOnVarianceElementOptimizer():

    def __init__(self, model, size, previous_quantizer, variance_state, index):
        self.model = model
        self.size = size
        self.previous_quantizer = previous_quantizer
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
                self.previous_quantizer.variance_quantizer.value,
                self.previous_quantizer_volatility)
                for z in self.variance_state.roots]
        out = [self.previous_quantizer.price_value*np.exp(r[:,np.newaxis])
                for r in returns]
        return out

    @lazy_property
    def conditional_probability(self): # (i,l)
        j = self.index
        out = (self.variance_state.transition_probability[:,j,np.newaxis]*
                self.previous_quantizer.probability)
        return out

    @lazy_property
    def previous_volatility(self): # returns (i)
        out = np.sqrt(self.previous_quantizer.variance_quantizer.value)
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
                / self.parent.previous_quantizer.probability[:,:,np.newaxis])
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
                /self.parent.previous_quantizer.price_value[:,:,np.newaxis])
        var = (self.parent.previous_quantizer
                .variance_quantizer.value[:,np.newaxis,np.newaxis])
        vol = self.parent.previous_volatility[:,np.newaxis,np.newaxis]
        out = self.parent.model._one_step_return_root(ret, var, vol)
        assert not np.any(np.isnan(out))
        return out

    @property
    def voronoi(self):# returns (m+1)
        out = get_voronoi_1d(self.value, lb=0.)
        return out


#########################################
# Marginal Variance Quantizer Internals #
#########################################


class _MarginalVarianceOptimizer():

    def __init__(self, model, size, previous_quantizer):
        self.model = model
        self.size = size
        self.previous_quantizer = previous_quantizer

    def __call__(self):
        # Perform optimization
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
        assert init.size==self.size

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

        ## Detect saddle points...
        #saddle = np.diag(sol.hess) == 0.
        #if np.any(saddle):
        #    if try_number < MAX_VAR_QUANT_TRY:
        #        #Re-initialize flat components only
        #        reinit_x = sol.x.copy()
        #        reinit_x[saddle] = 0.
        #        success, sol.x = self._perform_optimization(
        #                init=reinit_x, try_number=try_number+1)
        #    else:
        #        success = False

        # Format output
        return success, sol.x

    # TODO: is cache really necessary here
    # @lru_cache(maxsize=None)
    def eval_at(self, x):
        x.flags.writeable = False
        return _MarginalVarianceState(self, x)

    @lazy_property
    def min_variance (self):
        return self.model.get_lowest_one_step_variance(
                self.previous_quantizer.value[0])

class _MarginalVarianceState():

    def __init__(self, parent, x):
        self.parent = parent
        self.x = x
        assert self.x.ndim == 1 and self.x.shape == (self.parent.size,)

    # For optimizer
    @property
    def distortion(self):
        out = np.sum(self.integrals[2]
                - 2.*self.integrals[1]*self.value[np.newaxis,:]
                + self.integrals[0]*self.value[np.newaxis,:]**2., axis=1)
        out = self.parent.previous_quantizer.probability.dot(out)
        assert not np.isnan(out)
        return out

    @property
    def transformed_gradient(self):
        out = self.gradient.dot(self.jacobian)
        assert not np.any(np.isnan(out))
        return out

    @property
    def transformed_hessian(self):
        out = (self.jacobian.T.dot(self.hessian).dot(self.jacobian)
                + self._get_trans_hess_correction())
        assert not np.any(np.isnan(out))
        return out

    def _get_trans_hess_correction(self):
        out = np.zeros((self.parent.size,self.parent.size))
        mask = np.empty(self.parent.size)
        for i in range(self.parent.size):
            mask[:] = 0.
            mask[:i+1] = 1.
            out += self.gradient[i]*np.diag(mask*self.all_exp_x)
        assert not np.any(np.isnan(out))
        return out

    # Results

    @property
    def transition_probability(self):
        out = self.integrals[0]
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def probability(self):
        out = self.parent.previous_quantizer.probability.dot(self.transition_probability)
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def value(self):
        out = self.parent.min_variance + np.cumsum(self.all_exp_x)
        assert np.all(np.isfinite(out))
        assert np.all(out>0)
        assert np.all(np.diff(out)>0)
        return out

    @lazy_property
    def voronoi(self):
        out = get_voronoi_1d(self.value, lb=0.)
        assert out[0] == 0.
        assert out[1]>=self.parent.min_variance
        return out

    @lazy_property
    def roots(self):
        out = self.parent.model.one_step_roots(
                self.parent.previous_quantizer.value[:,np.newaxis],
                self.voronoi[np.newaxis,:])
        return out

    @property
    def gradient(self):
        out = -2. * self.parent.previous_quantizer.probability.dot(
                self.integrals[1]-self.integrals[0]*self.value[np.newaxis,:])
        assert not np.any(np.isnan(out))
        return out

    @property
    def hessian(self):
        diagonal = -2. * self.parent.previous_quantizer.probability.dot(
                self.integral_derivatives[0][1]
                -self.integral_derivatives[0][0]*self.value[np.newaxis,:]
                -self.integrals[0])
        off_diagonal = -2. * self.parent.previous_quantizer.probability.dot(
                self.integral_derivatives[1][1]
                -self.integral_derivatives[1][0]*self.value[np.newaxis,:])
        # Build hessian
        out = np.zeros((self.parent.size, self.parent.size))
        for j in range(self.parent.size):
            out[j,j] = diagonal[j]
            if j>0:
                out[j-1,j] = off_diagonal[j]
                out[j,j-1] =out[j-1,j] # make symmetric
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def integrals(self):
        return [self._get_integral(order=i) for i in range(3)]

    @lazy_property
    def pdf(self):
        return [norm.pdf(self.roots[i]) for i in range(2)]

    @lazy_property
    def cdf(self):
        return [norm.cdf(self.roots[i]) for i in range(2)]

    def _get_integral(self, *, order):
        integral_func = self.parent.model.one_step_expectation_until
        integral = [integral_func(self.parent.previous_quantizer.value[:,np.newaxis],
                self.roots[i], order=order, pdf=self.pdf[i], cdf=self.cdf[i])
                for i in range(2)]
        out = integral[1] - integral[0]
        out[np.isnan(out)] = 0
        out = np.diff(out, n=1, axis=1)
        return out

    @lazy_property
    def integral_derivatives(self):
        factor = self._get_factor_for_integral_derivative()
        dI = [self._get_integral_derivative(factor, order=i, lag=0)
                for i in range(2)]
        dI_lagged = [self._get_integral_derivative(factor, order=i, lag=-1)
                for i in range(2)]
        return dI, dI_lagged

    def _get_factor_for_integral_derivative(self):
        derivative_func = self.parent.model.one_step_roots_unsigned_derivative
        root_derivative = derivative_func(
                self.parent.previous_quantizer.value[:,np.newaxis],
                self.voronoi[np.newaxis,:])
        factor = (0.5 * root_derivative * (self.pdf[0]+self.pdf[1]))
        factor[np.isnan(factor)] = 0
        return factor

    def _get_integral_derivative(self, factor, order=0, lag=0):
        if order==0:
            d = factor
        elif order==1:
            d = factor * self.voronoi[np.newaxis,:]
        else:
            d = factor * self.voronoi[np.newaxis,:]**order
        d[np.isnan(d)] = 0
        if lag==-1:
            d = -d[:,:-1]
            d[:,0] = np.nan # first column is meaningless
        elif lag==0:
            d = np.diff(d, n=1, axis=1)
        elif lag==1:
            d = d[:,1:]
            d[:,-1] = np.nan # last column is meaningless
        else:
            assert False # should never happen
        return d

    @lazy_property
    def all_exp_x(self):
        return np.exp(self.x)*self.parent.min_variance/self.parent.size

    @lazy_property
    def jacobian(self):
        mask = np.tril(np.ones((self.parent.size, self.parent.size)))
        out = self.all_exp_x[np.newaxis,:]*mask
        assert not np.any(np.isnan(out))
        return out
