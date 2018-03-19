import numpy as np

from pymaat.quantutil import AbstractQuantization
from pymaat.quantutil import AbstractFactory
from pymaat.quantutil import AbstractQuantizer
import pymaat.garch.format
from pymaat.nputil import printoptions
from pymaat.util import lazy_property


class Conditional(AbstractQuantization):

    price_formatter = pymaat.garch.format.base_price

    def __init__(self, model, first_variance,
                 variance_size=100, price_size=100,
                 first_price=1., first_probability=1.,
                 nper=None, freq='daily'):
        super().__init__(model)
        self.nper = nper
        self.variance_size = variance_size
        self.price_size = price_size
        self.first_variance = first_variance
        self.first_price = first_price
        self.first_probability = first_probability
        self.variance_formatter = (
            pymaat.garch.format.get_variance_formatter(freq))

    def quantize(self, t, values):
        pass

    # Internals
    def _get_all_shapes(self):
        if self.nper is not None:
            variance_size = np.broadcast_to(
                self.variance_size, (self.nper,))
            price_size = np.broadcast_to(
                self.price_size, (self.nper,))
        else:
            variance_size = np.ravel(self.variance_size)
            price_size = np.ravel(self.price_size)

        return zip(variance_size, price_size)

    def _make_first_quantizer(self):
        # Set first price (i)
        price = np.ravel(self.first_price)

        # Set first variance (il)
        variance = np.atleast_2d(self.first_variance)
        shape = (price.size, variance.shape[1])
        variance = np.broadcast_to(variance, shape)

        # Set first probabilities
        probability = np.broadcast_to(self.first_probability, shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)

        return self.make_stub(
            price_value=price,
            variance_value=variance,
            probability=probability)

    @staticmethod
    def make_stub(price_value, variance_value, probability):
        """
        Returns a quantizer-like object which may be used
            as a previous quantizer e.g.
            for initializing recursions and testing
        """

        msg = ''
        if np.any(~np.isfinite(price_value)):
            msg += 'Invalid prices: NaNs or infinite\n'
        if np.any(price_value <= 0.):
            msg += 'Invalid prices: must be strictly positive\n'
        if np.any(~np.isfinite(variance_value)):
            msg += 'Invalid variances: NaNs or infinite\n'
        if np.any(variance_value <= 0.):
            msg += 'Invalid variances: must be strictly positive\n'
        if probability.shape != variance_value.shape:
            msg += 'Invalid probabilities: size mismatch\n'
        if np.abs(np.sum(probability)-1.) > 1e-12:
            msg += 'Invalid probabilities: must sum to one\n'
        if np.any(probability < 0.):
            msg += 'Invalid probabilities: must be positive\n'
        if len(msg) > 0:
            raise ValueError(msg)

        return _Quantizer(price_value=price_value,
                          variance_value=variance_value,
                          probability=probability,
                          transition_probability=None,
                          shape=variance_value.shape)

    def _one_step_optimize(self, t, shape, previous, *, verbose=True):
        if verbose:
            header = "[t={0:d}] ".format(t)
            header += " Searching for optimal quantizer"
            sep = "\n" + ("#"*len(header)) + "\n"
            print(sep)
            print(header, end='')

        factory = _QuantizerFactory(self.model, shape, previous)
        quant = factory.optimize(verbose)

        if verbose:
            # Display results
            print("\n[Price distortion]\n{0:.6e}".format(
                quant.price_distortion))

            print("\n[Variance distortion]")
            with printoptions(precision=6):
                print(quant.variance_distortion)

            print("\n[Price quantizer (%/y)]")
            with printoptions(precision=2, suppress=True):
                print(self.price_formatter(quant.price_value))
            print(sep)

        return quant


_Quantizer = namedtuple('_Quantizer', [
    'price_value',
    'variance_value',
    'probability',
    'transition_probability',
    'shape'])


class _QuantizerFactory:

    def __init__(self, model, shape, previous):
        self.model = model
        self.shape = shape
        self.previous = previous

    def optimize(self, verbose):
        # (1) Optimize price quantizer
        price_factory = _MarginalPriceFactory(
            self.model,
            self.shape[0],
            self.previous)
        price_quant = price_factory.optimize(verbose)

        # (2) Optimize all conditional variance quantizers

        # (3) Merge results in a quantizer

        # variance_size = shape[0]
        # price_size = shape[1]

        # # Get optimal variance quantization first...

        # # Perform asset price optimization for each next variances
        # # TODO: Parallelize!
        # all_price_states = [None]*variance_size
        # for j in range(variance_size):
        #     optimize = _ConditionalOnVarianceElementOptimizer(
        #             self.model,
        #             price_size,
        #             previous,
        #             variance_state,
        #             j)
        #     all_price_states[j] = optimize()

        # # Join processes
        # # Merge & format results
        # shape = (variance_size, price_size)
        # price_value = np.empty(shape)
        # probability = np.empty(shape)
        # transition_probability = np.empty(previous.shape+shape)
        # for j in range(variance_size):
        #     price_value[j] =  all_price_states[j].value
        #     probability[j] =  all_price_states[j].probability
        #     transition_probability[:,:,j,:] = \
        #          all_price_states[j].transition_probability

        # variance_quantizer = MarginalVariance._QuantizerFactory(
        #         value = variance_state.value,
        #         probability = variance_state.probability,
        #         transition_probability = variance_state.transition_probability)

        # return ConditionalOnVariance._QuantizerFactory(
        #         price_value=price_value,
        #         variance_quantizer=variance_quantizer,
        #         probability=probability,
        #         transition_probability=transition_probability)
        pass



class _MarginalPriceFactory(AbstractFactory):

    def __init__(self, model, size, previous):
        super().__init__(
                model, (size,), previous,
                target=_MarginalPriceUnconstrained
                )

    def gradient_tolerance(self):
        return 1e-4 / (self.previous.shape[0]*self.shape[0]**2.)

class _ConditionalVarianceFactory(AbstractFactory):

    def __init__(self, model, size, previous,
            price_quantizer, price_index):
        super().__init__(
                model, (size,), previous,
                target=_ConditionalVarianceUnconstrained
                )
        self.price_quantizer = price_quantizer
        self.price_index = price_index

    def gradient_tolerance(self):
        return 1e-4 * self.min_variance**2. / (
                self.previous.shape[0]*self.shape[0]**2.)

    @lazy_property
    def min_variance(self):
        return self.model.get_lowest_one_step_variance(
            np.amin(self.previous.variance_value))



# class _MarginalPriceQuantizer(AbstractQuantizer):

#     def __init__(self, parent, value):
#         super().__init__(self, parent, value, lb=0., prevdim=2):

#     @lazy_property
#     def _integral(self):
#         """
#         Returns of len-3 list indexed by order and filled with
#         self.previous.size-by-self.size arrays containing
#         ```
#             \\mathcal{I}_{order,t}^{(ij)}
#         ```
#         for all i,j.
#         Rem. Output only has valid entries, ie free of NaNs
#         """
#         roots = self._roots
#         previous_value = self.previous.value[:, np.newaxis]
#         cdf = self._cdf
#         pdf = self._pdf

#         def _get_integral(order):
#             integral = [self.model.one_step_expectation_until(
#                 previous_value, roots[right], order=order,
#                 _pdf=pdf[right], _cdf=cdf[right])
#                 for right in [False, True]]
#             out = integral[True] - integral[False]
#             out[self._no_roots] = 0.0
#             return np.diff(out, n=1, axis=1)
#         return [_get_integral(order) for order in [0, 1, 2]]

#     @lazy_property
#     def _delta(self):
#         """
#         Returns a previous.size-by-size+1 array containing
#         ```
#             \\delta_{t}^{(ij\\pm)}
#         ```
#         for all i,j.
#         Rem. Output only has valid entries, ie free of NaNs, but may be
#             infinite (highly unlikely)
#         In particular,
#             (1) When no root exists (eg. voronoi==0), delta is zero
#             (2) When voronoi==+np.inf, delta is zero
#             (3) When voronoi is at a root singularity, delta is np.inf
#         """
#         unsigned_root_derivative = \
#             self.model.one_step_roots_unsigned_derivative(
#                 self.previous.value[:, np.newaxis],
#                 self.voronoi[np.newaxis, :])
#         limit_index = unsigned_root_derivative == np.inf
#         if np.any(limit_index):
#             warnings.warn(
#                 "Voronoi tiles at singularity detected",
#                 UserWarning
#             )
#         out = (0.5*unsigned_root_derivative
#                * (self._pdf[True]+self._pdf[False]))
#         # Limit cases
#         # (1)  When a root does not exist, delta is 0
#         out[self._no_roots] = 0.
#         # (2) When voronoi is +np.inf (which always occur
#         #   at the last column), delta is zero.
#         out[:, -1] = 0.
#         # (3) When roots approach the singularity, delta is np.inf
#         out[limit_index] = np.inf
#         return out

#     @lazy_property
#     def _pdf(self):
#         """
#         Returns a len-2 tuples containing (left, right) PDF of roots
#         """
#         roots = self._roots
#         with np.errstate(invalid='ignore'):
#             return [norm.pdf(roots[right]) for right in [False, True]]

#     @lazy_property
#     def _cdf(self):
#         """
#         Returns a len-2 tuples containing (left, right) CDF roots
#         """
#         roots = self._roots
#         with np.errstate(invalid='ignore'):
#             return [norm.cdf(roots[right]) for right in [False, True]]

#     @lazy_property
#     def _roots(self):
#         """
#         Returns a len-2 tuples containing (left, right) roots
#         """
#         return self.model.one_step_roots(
#             self.previous.value[:, np.newaxis],
#             self.voronoi[np.newaxis, :])

#     @lazy_property
#     def _no_roots(self):
#         return np.isnan(self._roots[0])


# class _MarginalPriceUnconstrained(_MarginalPriceQuantizer):
#     pass

#@############################# OLDDD
# class _ConditionalOnVarianceElementOptimizer():

#    def __init__(self, model, size, previous, variance_state, index):
#        self.model = model
#        self.size = size
#        self.previous = previous
#        self.variance_state = variance_state
#        self.index = index
#        assert (self.index>=0 and self.index<self.variance_state.parent.size)

#    def __call__(self):
#        success, optim_x = self._perform_optimization()
#        if not success:
#            # Early failure if optimizer unsuccessful
#            raise RuntimeError
#        else:
#            return self.eval_at(optim_x)

#    def _perform_optimization(self, init=None, try_number=0):
#        # TODO return success==false when try_number reach MAX
#        if init is None:
#            init = np.zeros(self.size)

#        # We let the minimization run until the predicted reduction
#        #   in distortion hits zero (or less) by setting gtol=0 and
#        #   maxiter=inf. We hence expect a status flag of 2.
#        sol = scipy.optimize.minimize(
#                lambda x: self.eval_at(x).distortion,
#                init,
#                method='trust-ncg',
#                jac=lambda x: self.eval_at(x).transformed_gradient,
#                hess=lambda x: self.eval_at(x).transformed_hessian,
#                options={'disp':False, 'maxiter':np.inf, 'gtol':0.})

#        success = sol.status==2 or sol.status==0

#        # Format output
#        return success, sol.x

#    #TODO: profile caching (is it really necessary?)
#    def eval_at(self, x):
#        # Make sure array is hashable for lru_cache
#        x.flags.writeable = False
#        return _ConditionalOnVarianceElementState(self, x)

#    @lazy_property
#    def branch_probability(self): # 2 * (i)
#        j = self.index
#        pdf = self.variance_state.pdf
#        norm_factor = (pdf[0][:,j]+pdf[1][:,j])
#        out = [p[:,j]/norm_factor for p in pdf]
#        # TODO: send to tests: assert np.all(out[0]+out[1]==1.)
#        return out

#    @lazy_property
#    def conditional_prices(self):# returns ((i,l),(i,l))
#        j = self.index
#        returns = [self.model._one_step_return_equation(z[:,j],
#                self.previous.variance_quantizer.value,
#                self.previous_volatility)
#                for z in self.variance_state.roots]
#        out = [self.previous.price_value*np.exp(r[:,np.newaxis])
#                for r in returns]
#        return out

#    @lazy_property
#    def conditional_probability(self): # (i,l)
#        j = self.index
#        out = (self.variance_state.transition_probability[:,j,np.newaxis]*
#                self.previous.probability)
#        return out

#    @lazy_property
#    def previous_volatility(self): # returns (i)
#        out = np.sqrt(self.previous.variance_quantizer.value)
#        assert not np.any(np.isnan(out))
#        return out

#    @lazy_property
#    def max_price(self):
#        return np.amax(self.conditional_prices[1])

#    @lazy_property
#    def min_price(self):
#        return np.amin(self.conditional_prices[0])


# class _ConditionalOnVarianceElementState():

#    def __init__(self, parent, x):
#        self.parent = parent
#        self.x = x
#        assert (x.ndim == 1 and self.x.shape == (self.parent.size,))

#    @lazy_property
#    def value(self):
#        out = self.parent.min_price + np.cumsum(self.all_exp_x)
#        assert np.all(np.isfinite(out))
#        assert np.all(out>0)
#        assert np.all(np.diff(out)>0)
#        return out

#    @lazy_property
#    def all_exp_x(self):
#        span = self.parent.max_price-self.parent.min_price
#        return np.exp(self.x)*span/(self.parent.size+1)

#    @property
#    def probability(self):
#        return self.hessian_diagonal/2.

#    @property
#    def transition_probability(self):
#        out = ((self.weights[0] + self.weights[1]) # (i,l,m)
#                / self.parent.previous.probability[:,:,np.newaxis])
#        return out

#    @property
#    def distortion(self):
#        out = (self.weights[0]
#                * (self.parent.conditional_prices[0][:,:,np.newaxis]
#                - self.value[np.newaxis,np.newaxis,:])**2.
#            + self.weights[1]
#                * (self.parent.conditional_prices[1][:,:,np.newaxis]
#                - self.value[np.newaxis,np.newaxis,:])**2.) #(i,l,m)
#        return out.sum()

#    @property
#    def transformed_gradient(self):
#        out = self.gradient.dot(self.jacobian)
#        assert not np.any(np.isnan(out))
#        return out

#    @property
#    def gradient(self): # m
#        out = -2.*(self.weights[0]
#                * (self.parent.conditional_prices[0][:,:,np.newaxis]
#                - self.value[np.newaxis,np.newaxis,:])
#                + self.weights[1]
#                * (self.parent.conditional_prices[1][:,:,np.newaxis]
#                - self.value[np.newaxis,np.newaxis,:]))
#        return np.ravel(out.sum(axis=(0,1))) # m

#    @property
#    def transformed_hessian(self):
#        hessian = np.diag(self.hessian_diagonal)
#        out = (self.jacobian.T.dot(hessian).dot(self.jacobian)
#                + self.hessian_correction)
#        assert not np.any(np.isnan(out))
#        return out

#    @lazy_property
#    def hessian_diagonal(self): # m
#        out = 2.*(self.weights[0] + self.weights[1])
#        return np.ravel(out.sum(axis=(0,1))) # m

#    @property
#    def jacobian(self):
#        mask = np.tril(np.ones((self.parent.size, self.parent.size)))
#        out = self.all_exp_x[np.newaxis,:]*mask
#        assert not np.any(np.isnan(out))
#        return out

#    @property
#    def hessian_correction(self):
#        out = np.zeros((self.parent.size,self.parent.size))
#        mask = np.empty(self.parent.size)
#        for i in range(self.parent.size):
#            mask[:] = 0.
#            mask[:i+1] = 1.
#            out += self.gradient[i]*np.diag(mask*self.all_exp_x)
#        assert not np.any(np.isnan(out))
#        return out

#    @lazy_property
#    def weights(self): # 2*(i,l,m)
#        cond_proba = self.parent.conditional_probability[:,:,np.newaxis]
#        out = [cond_proba*p[:,np.newaxis,np.newaxis]*I
#                for (p,I) in zip(
#                    self.indicator,
#                    self.parent.branch_probability)]
#        return out

#    @property
#    def indicator(self):# 2 * (i,l,m)
#        # TODO: more robust treatment of nans
#        # TODO: optimize memory
#        j = self.parent.index
#        roots = self.roots # only compute once
#        out = [np.logical_and(
#                roots[:,:,:-1] < z[:,j,np.newaxis,np.newaxis],
#                z[:,j,np.newaxis,np.newaxis] < roots[:,:,1:])
#                for z in self.parent.variance_state.roots] # 2 * (i,l,m)
#        return out

#    @property
#    def roots(self): # (i,l,m+1)
#        ret = (self.voronoi[np.newaxis,np.newaxis,:]
#                /self.parent.previous.price_value[:,:,np.newaxis])
#        var = (self.parent.previous
#                .variance_quantizer.value[:,np.newaxis,np.newaxis])
#        vol = self.parent.previous_volatility[:,np.newaxis,np.newaxis]
#        out = self.parent.model._one_step_return_root(ret, var, vol)
#        assert not np.any(np.isnan(out))
#        return out

#    @property
#    def voronoi(self):# returns (m+1)
#        out = voronoi_1d(self.value, lb=0.)
#        return out
