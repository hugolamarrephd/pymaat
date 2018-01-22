# All Marginal Garch Quantizers

from functools import partial, wraps, lru_cache
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

class AbstractQuantization():

    def optimize(self):
        current = self.first_quantizer
        # Perform recursion
        self.all_quantizers = []
        for s in self.shape:
            # Gather results from previous iteration...
            self.all_quantizers.append(current)
            # Advance to next time step (computations done here...)
            current = self._one_step_optimize_and_advance(model, s, current)
        # Handle last quantizer
        self.all_quantizers.append(current)

class ConditionalOnVarianceQuantization(AbstractQuantization):
    # Quantizes marginal variance first and then
    # quantizes price *conditional* on the optimal variance quantization

    Quantizer = collections.namedtuple('ConditionalOnVarianceQuantizer',
            ['shape',
            'value',
            'probability',
            'transition_probability',
            'variance'])

    def __init__(self, model, first_variance, *,
            variance_size=100, price_size=100,
            first_price=1., first_probability=1.,
            nper=None):

        self.model = model

        if nper is not None:
            variance_size = np.broadcast_to(variance_size, (nper,))
            price_size = np.broadcast_to(price_size, (nper,))
        else:
            variance_size = np.atleast_1d(variance_size)
            price_size = np.atleast_1d(price_size)

        self.shape = list(zip(variance_size, price_size))
        self.first_quantizer = self._get_first_quantizer(
                first_variance,
                first_price,
                first_probability)

    def get_markov_chain(self):
        # TODO: optimize memory
        sizes = []; values = []; probabilities=[]; transition_probabilities=[]
        for q in self.all_quantizers:
            # Sizes
            sizes.append(q.shape[0]*q.shape[1])
            # Values
            tmp_value = np.empty(q.shape, dtype=[('price',np.float_),
                ('variance',np.float_)])
            tmp_value['price'] = q.value
            tmp_value['variance'] = q.variance.value[:,np.newaxis]
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

    def quantize(self, t, variance, price):
        pass

    # Internals

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
        value = np.broadcast_to(price, shape)# (j,m)

        # Set first probabilities
        probability = np.broadcast_to(probability, shape) # (j,m)
        # ...and quietly normalize
        probability = probability/np.sum(probability)

        # Set first variance quantizer
        variance_probability = np.sum(probability, axis=1)
        variance = MarginalVariance._get_first_quantizer(
                variance,
                variance_probability)

        return ConditionalOnVariance.Quantizer(shape=shape,
            value=value,
            probability=probability,
            variance=variance)

    @staticmethod
    def _one_step_optimize_and_advance(model, shape, previous):
        variance_size = shape[0]
        price_size = shape[1]
        # Find optimal variance quantization first...
        variance = MarginalVariance._one_step_optimize_and_advance(
            model, variance_size, previous.variance)

        # Perform asset price optimization for each next variances
        # TODO: Parallelize!
        all_price_quantizers = [None]*variance_size
        for j in range(variance_size):
            optimize = _ConditionalOnVarianceElementOptimizer(
                    model, price_size, previous, variance, j)
            all_price_quantizers[j] = optimize()

        # Join processes and merge results
        shape = (variance_size, price_size)
        value = np.empty(shape)
        probability = np.empty(shape)
        transition_probability = np.empty(previous.shape+shape)
        for j in range(variance_size):
            value[j] =  all_price_quantizers[j].value
            probability[j] =  all_price_quantizers[j].probability
            transition_probability[:,:,j,:] = \
                 all_price_quantizers[j].transition_probability

        return ConditionalOnVariance.Quantizer(shape=shape,
            value=value,
            probability=probability,
            transition_probability=transition_probability,
            variance=variance)


class MarginalVarianceQuantization():

    Quantizer = collections.namedtuple('MarginalVarianceQuantizer',
            ['size',
            'shape',
            'value',
            'probability',
            'transition_probability',
            'roots',
            'pdf'])

    def __init__(self, model, first_value, *,
            size=100, first_probability=1.,
            nper=None):

        self.model = model

        if nper is not None:
            self.shape = np.broadcast_to(size, (nper,))
        else:
            self.shape = np.atleast_1d(size)

        self.first_quantizer = self._get_first_quantizer(
                first_value, first_probability)

    def get_markov_chain(self):
        sizes = []; values = []; probabilities=[]; transition_probabilities=[]
        for q in self.all_quantizers:
            sizes.append(q.size)
            values.append(q.value)
            probabilities.append(q.probability)
            transition_probabilities.append(q.transition_probability)
        return MarkovChain(nper=len(q), ndim=1, sizes=sizes, values=values,
                probabilities=probabilities,
                transition_probabilities=transition_probabilities)


    def quantize(self, t, variance, price):
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
                size=value.size,
                shape=value.shape,
                value=value,
                probability=probability)

    @staticmethod
    def _one_step_optimize_and_advance(model, size, previous):
        optimize = _MarginalVarianceOptimizer(model, size, previous)
        return optimize()

###################################
# Conditional Quantizer Internals #
###################################

class _ConditionalOnVarianceElementOptimizer():

    def __init__(self, model, size, previous, variance, index):
        self.model = model
        self.size = size
        self.previous = previous
        self.variance = variance
        self.index = index
        assert (self.index>=0 and self.index<self.variance.size)

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
    @lru_cache(maxsize=None)
    def eval_at(self, x):
        # Make sure array is hashable for lru_cache
        x.flags.writeable = False
        return _ConditionalOnVarianceElementState(self, x)

    @lazy_property
    def branch_probability(self): # 2 * (i)
        j = self.index
        pdf = self.variance.pdf
        norm_factor = (pdf[0][:,j]+pdf[1][:,j])
        out = [p[:,j]/norm_factor for p in pdf]
        # TODO: send to tests: assert np.all(out[0]+out[1]==1.)
        return out

    @lazy_property
    def conditional_prices(self):# returns ((i,l),(i,l))
        j = self.index
        returns = [self.model._one_step_return_equation(z[:,j],
                self.previous.variance.value,
                self.previous_volatility)
                for z in self.variance.roots]
        out = [self.previous.value*np.exp(r[:,np.newaxis])
                for r in returns]
        return out

    @lazy_property
    def conditional_probability(self): # (i,l)
        j = self.index
        out = (self.variance.transition_probability[:,j,np.newaxis]*
                self.previous.probability)
        return out

    @lazy_property
    def previous_volatility(self): # returns (i)
        out = np.sqrt(self.previous.variance.value)
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
                for z in self.parent.variance.roots] # 2 * (i,l,m)
        return out

    @property
    def roots(self): # (i,l,m+1)
        ret = (self.voronoi[np.newaxis,np.newaxis,:]
                /self.parent.previous.value[:,:,np.newaxis])
        var = self.parent.previous.variance.value[:,np.newaxis,np.newaxis]
        vol = self.parent.previous_volatility[:,np.newaxis,np.newaxis]
        out = self.parent.model._one_step_return_root(ret, var, vol)
        assert not np.any(np.isnan(out))
        return out

    @property
    def voronoi(self):# returns (m+1)
        out = get_voronoi_1d(self.value, lb=0.)
        return out


################################
# Variance Quantizer Internals #
################################


class _MarginalVarianceOptimizer():

    def __init__(self, model, size, previous):
        self.model = model
        self.size = size
        self.previous = previous

    def __call__(self):
        # Perform optimization
        success, optim_x = self._perform_optimization()
        if not success:
            # Early failure if optimizer unsuccessful
            raise RuntimeError
        else:
            result = self.eval_at(optim_x)
            return MarginalVariance.Quantizer(
                    size=result.size,
                    shape=result.shape,
                    value=result.value,
                    probability=result.probability,
                    transition_probability=result.transition_probability,
                    roots=result.roots,
                    pdf=result.pdf)

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

    @lru_cache(maxsize=None)
    def eval_at(self, x):
        x.flags.writeable = False
        return _MarginalVarianceState(
                self.model, self.size, self.previous, x)


class _MarginalVarianceState():

    def __init__(self, model, size, previous, x):
        self.model = model
        self.size = size
        self.previous = previous
        self.x = x
        assert self.x.ndim == 1 and self.x.shape == (self.size,)

    @lazy_property
    def h_min(self):
        return self.model.get_lowest_one_step_variance(
                self.previous.value[0])

    @lazy_property
    def distortion(self):
        out = np.sum(self.integrals[2]
                - 2.*self.integrals[1]*self._value
                + self.integrals[0]*self._value**2., axis=1)
        out = self.previous.probability.dot(out)
        assert not np.isnan(out)
        return out

    @lazy_property
    def transformed_gradient(self):
        out = self.gradient.dot(self.jacobian)
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def transformed_hessian(self):
        out = (self.jacobian.T.dot(self.hessian).dot(self.jacobian)
                + self.hessian_correction)
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def transition_probability(self):
        out = self.integrals[0]
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def probability(self):
        out = self.previous.probability.dot(self.transition_probability)
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def value(self):
        out = self.h_min + np.cumsum(self.all_exp_x)
        assert np.all(np.isfinite(out))
        assert np.all(out>0)
        assert np.all(np.diff(out)>0)
        return out

    @lazy_property
    def voronoi(self):
        out = get_voronoi_1d(self.value, lb=0.)
        assert out[0] == 0.
        assert out[1]>=self.h_min
        return out

    @lazy_property
    def roots(self):
        out = self.model.one_step_roots(self._prev_value, self._voronoi)
        return out

    @lazy_property
    def gradient(self):
        out = -2. * self.previous.probability.dot(
                self.integrals[1]-self.integrals[0]*self._value)
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def hessian(self):
        diagonal = -2. * self.previous.probability.dot(
                self.integral_derivatives[0][1]
                -self.integral_derivatives[0][0]*self._value
                -self.integrals[0])
        off_diagonal = -2. * self.previous.probability.dot(
                self.integral_derivatives[1][1]
                -self.integral_derivatives[1][0]*self._value)
        # Build hessian
        out = np.zeros((self.size, self.size))
        for j in range(self.size):
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
        integral_func = self.model.one_step_expectation_until
        integral = [integral_func(self._prev_value, self.roots[i],
                order=order, pdf=self.pdf[i], cdf=self.cdf[i])
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
        derivative_func = self.model.one_step_roots_unsigned_derivative
        root_derivative = derivative_func(self._prev_value, self._voronoi)
        factor = (0.5 * root_derivative * (self.pdf[0]+self.pdf[1]))
        factor[np.isnan(factor)] = 0
        return factor

    def _get_integral_derivative(self, factor, order=0, lag=0):
        if order==0:
            d = factor
        elif order==1:
            d = factor * self._voronoi
        else:
            d = factor * self._voronoi**order
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
        return np.exp(self.x)*self.h_min/self.size

    @lazy_property
    def jacobian(self):
        mask = np.tril(np.ones((self.size, self.size)))
        out = self.all_exp_x[np.newaxis,:]*mask
        assert not np.any(np.isnan(out))
        return out

    @lazy_property
    def hessian_correction(self):
        out = np.zeros((self.size,self.size))
        mask = np.empty(self.size)
        for i in range(self.size):
            mask[:] = 0.
            mask[:i+1] = 1.
            out += self.gradient[i]*np.diag(mask*self.all_exp_x)
        assert not np.any(np.isnan(out))
        return out

    # Broadcast-ready formatting
    # TODO: Memory inefficient to store this!
    @lazy_property
    def _prev_value(self):
        return self.previous.value[:,np.newaxis] #size-by-1

    @lazy_property
    def _value(self):
        return self.value[np.newaxis,:] #1-by-size

    @lazy_property
    def _voronoi(self):
        return self.voronoi[np.newaxis,:] #1-by-size
