from collections import namedtuple
from hashlib import sha1
import warnings
from cachetools import LRUCache

import numpy as np
from scipy.linalg import norm as infnorm
from scipy.optimize import basinhopping
from scipy.stats import norm, truncnorm, uniform
import matplotlib.pyplot as plot

from pymaat.garch.quant import AbstractQuantization
from pymaat.mathutil import voronoi_1d
from pymaat.util import lazy_property
from pymaat.nputil import diag_view, printoptions


class MarginalVariance(AbstractQuantization):

    def __init__(
        self, model, first_variance, size=100, first_probability=1.,
            nper=None):
        shape = self._init_shape(nper, size)
        first_quantizer = self._make_first_quantizer(
            first_variance,
            first_probability)
        super().__init__(model, shape, first_quantizer)

   # def get_markov_chain(self):
   #      sizes = []
   #      values = []
   #      probabilities = []
   #      transition_probabilities = []
   #      for q in self.all_quantizers:
   #          sizes.append(q.value.size)
   #          values.append(q.value)
   #          probabilities.append(q.probability)
   #          transition_probabilities.append(q.transition_probability)
   #      return MarkovChain(nper=len(q), ndim=1,
   #              sizes=sizes,
   #              values=values,
   #              probabilities=probabilities,
   #              transition_probabilities=transition_probabilities)

    def visualize_all(self):
        # cmap = matplotlib.cm.get_cmap('gnuplot2')
        # cmap = matplotlib.colors.LinearSegmentedColormap(
        #     'gnuplot2_reverse', matplotlib.cm.revcmap(cmap._segmentdata))
        fig, ax = plt.subplots()

        for (t, q) in enumerate(self.all_quantizers):
            x = np.broadcast_to(t+1, q.shape)
            y = np.sqrt(252.*q.value)*100.
            c = q.probability*100.
            # Add scatter
            cax = plt.scatter(x.flatten(), y.flatten(), c=c.flatten(),
                              marker=",", cmap=cmap, s=ss)

        plt.xlabel(r'Trading Days ($t$)')
        plt.ylabel(r'Annualized Volatility (%)')
        plt.xlim(1, t+1)
        plt.xticks([1, 5, 10, 15, 20])
        plt.yticks([10, 20, 30, 40])

        # Add colorbar
        cbar = fig.colorbar(cax)
        cbar.ax.set_title(r'$p_{h}^{(i)}$(%)', fontsize=fs)

        fig.set_size_inches(4, 3)
        ax.grid(True, linestyle='dashed', alpha=0.5, axis='y')
        plt.tight_layout()
        # plt.savefig('marginal_variance.eps', format='eps', dpi=1000)

    def visualize_transition(self, t=1):
        if t == 0:
            raise ValueError
        x = np.sqrt(252.*mc.values[t])*100.
        selected = np.logical_and(x > 10., x < 30.)
        x = x[selected, np.newaxis]
        y = np.sqrt(252.*mc.values[t+1])*100.
        y = y[np.newaxis, selected]
        c = mc.transition_probabilities[t]*100.
        c = c[selected][:, selected]
        x, y = np.broadcast_arrays(x, y)

        fig, ax = plt.subplots()

        # Add scatter
        cax = plt.scatter(x.flatten(), y.flatten(), c=c.flatten(),
                          marker=",", cmap=cmap,
                          s=ss, linewidths=0.15)

        plt.xlabel(r'Annualized Volatility (%) at t=10', fontsize=fs)
        plt.ylabel(r'Annualized Volatility (%) at t=11', fontsize=fs)
        plt.xticks([10, 15, 20, 25, 30], fontsize=fs)
        plt.yticks([10, 15, 20, 25, 30], fontsize=fs)
        plt.xlim(10, 30)
        plt.ylim(10, 30)
        plt.tight_layout()

        # Add colorbar
        cbar = fig.colorbar(cax)
        cbar.ax.set_title(r'$p_{h}^{(ij)}$(%)', fontsize=fs)

        fig.set_size_inches(4, 3)
        ax.grid(True, linestyle='dashed', alpha=0.5)
        plt.tight_layout()
        # plt.savefig('marginal_variance_transition.eps', format='eps', dpi=1000)

    def quantize(self, t, values):
        pass

    @staticmethod
    def _init_shape(nper, size):
        if nper is not None:
            shape = np.broadcast_to(size, (nper,))
        else:
            shape = np.atleast_1d(size)
        return shape

    @staticmethod
    def _make_first_quantizer(value, probability):
        # Set first values
        value = np.ravel(np.atleast_1d(value))
        # Set first probabilities
        probability = np.broadcast_to(probability, value.shape)
        # ...and quietly normalize
        probability = probability/np.sum(probability)
        return _QuantizerFactory.make_stub(value, probability)

    def _one_step_optimize(
            self, size, previous, *, init=None):
        return _QuantizerFactory(
            self.model, size, previous).optimize(init)


class _QuantizerFactory:

    def __init__(self, model, size, previous):
        self.model = model
        self.shape = (size,)
        self.size = size
        self.previous = previous
        self._cache = LRUCache(maxsize=30)
        if np.any(self.previous.probability <= 0.):
            raise ValueError(
                "Previous probabilities must be strictly positive")

    def optimize(self, *, niter=1000, seed=None, init=None, verbose=True):
        self._T = self.min_variance**2./(self.previous.size*self.size**2.)
        self._tol = self._T*1e-6
        if init is None:
            init = np.zeros(self.size)

        def func(x): return self.make_unconstrained(x).distortion
        self.take_step = _OptimizerTakeStep(self, seed)
        # Set up convex optimizer...

        def jac(x): return self.make_unconstrained(x).gradient

        def hess(x): return self.make_unconstrained(x).hessian
        options = {'maxiter': 1000, 'disp': False, 'gtol': self._tol}
        minimizer_kwargs = {
            'method': 'trust-ncg',
            'jac': jac,
            'hess': hess,
            'options': options
        }
        if verbose:
            callback = self._print_callback
        else:
            def callback(x, f, a): return None
        try:
            opt = basinhopping(
                func,
                init,
                niter=niter,
                T=self._T,
                take_step=self.take_step,
                disp=False,
                minimizer_kwargs=minimizer_kwargs,
                callback=callback,
                niter_success=100
            )
        except _AtSingularity as exc:
            # Optimizer met a singularity for indices idx
            init[exc.idx] += 1e-2
            if verbose:
                print("Singularity failure: re-booting optimizer\n")
            return self.optimize(
                niter=niter, seed=seed, init=init, verbose=verbose)
        return self.make_unconstrained(opt.x)

    def _print_callback(self, x, f, accept):
        q = self.make_unconstrained(x)

        # Initialization
        if not hasattr(self, '_distortion'):
            self._distortion = q.distortion
            self._value = q.value
            self._niter = 0

        if q.distortion < self._distortion:
            objective = 100.*(q.distortion/self._distortion-1.)
            msg = "\n[#{0:d}] ".format(self._niter)
            msg += "New Global Optimum"
            msg += " {0:+.2f}% from global {1:.4e}\n".format(
                objective, self._distortion)
            msg += "\tDistortion: {0:.32e}\n".format(q.distortion)
            msg += "\tStepsize: {0:.2f}\n".format(self.take_step.stepsize)
            msg += "\tGradient (tol): {0:.4e} ({1:.4e})\n".format(
                infnorm(q.gradient), self._tol)
            change = 100.*np.amax(np.absolute(q.value/self._value-1.))
            msg += "\tDelta-Quantizer: {0:.2f}%\n".format(change)
            msg += "\tQuantizer(%\y):"
            print(msg)
            with printoptions(precision=2):
                print("\t{}".format(np.sqrt(252.*q.value)*100.))
            self._distortion = q.distortion
            self._value = q.value

        self._niter += 1

    def make_unconstrained(self, x):
        x.flags.writeable = False
        key = sha1(x).hexdigest()
        if key not in self._cache:
            quant = _Unconstrained(self, x)
            self._cache[key] = quant
        return self._cache.get(key)

    def make(self, value):
        return _Quantizer(self, value)

    @staticmethod
    def make_stub(value, probability):
        """
        Returns a minimal quantizer-like object which may be used
            as a previous quantizer when instantiating a subsequent
            factory.
        Rem. It is used for initializing recursions and testing
        """
        if np.any(np.isnan(value)) or np.any(~np.isfinite(value)):
            msg = 'Invalid value: NaNs or infinite'
            raise ValueError(msg)
        if np.any(value <= 0.):
            msg = 'Invalid value: variance must be strictly positive'
            raise ValueError(msg)
        if (
            np.abs(np.sum(probability)-1.) > 1e-12
            or np.any(probability < 0.)
        ):
            msg = 'Probabilities must sum to one and be positive'
            raise ValueError(msg)
        fact = namedtuple('_Quantizer',
                          ['value', 'probability', 'size', 'shape'])
        return fact(value=value, probability=probability,
                    size=value.size, shape=value.shape)

    # Utilities

    @lazy_property
    def min_variance(self):
        return self.model.get_lowest_one_step_variance(
            np.amin(self.previous.value))

    @lazy_property
    def singularities(self):
        return np.sort(
            self.model.get_lowest_one_step_variance(
                self.previous.value))


class _OptimizerTakeStep():

    def __init__(self, parent, seed=None):
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()
        self.parent = parent
        self.stepsize = 1.
        if self.parent.previous.size > 1:
            self.scale = np.mean(np.diff(self.parent.singularities))
        else:
            raise ValueError

    def __call__(self, x):
        _, old_value = _Unconstrained._change_of_variable(self.parent, x)
        new_value = old_value.copy()
        new_value = self._make_forward_pass(new_value)
        new_x = _Unconstrained._inv_change_of_variable(
            self.parent, new_value)
        # Bound x to avoid numerical instability
        return np.clip(new_x, -10., 20.)

    def _make_forward_pass(self, value):
        lb = self.parent.singularities[0]
        ub = value[-1]
        for i in range(value.size):
            loc = value[i]
            scale = self.stepsize*self.scale
            a_ = (lb-loc)/scale
            b_ = (ub-loc)/scale
            tmp = truncnorm.rvs(
                a_, b_,
                loc=loc, scale=scale,
                random_state=self.random_state)
            if not (tmp > lb and tmp < ub):
                # Fail-safe if numerical instability arises
                value[i:] = uniform(
                    lb, ub,
                    size=value[i:].shape,
                    random_state=self.random_state)
                msg = "Take-step failsafe: numerical instability "
                msg += " [stepsize={.2f}]".format(self.stepsize)
                warnings.warn(msg, UserWarning)
                break
            else:
                lb = value[i] = tmp
        return value


class _Quantizer():

    def __init__(self, parent, value):
        if not isinstance(parent, _QuantizerFactory):
            msg = 'Quantizers must be instantiated from valid factory'
            raise ValueError(msg)
        else:
            self.parent = parent
            self.model = parent.model
            self.size = parent.size
            self.shape = parent.shape
            self.previous = parent.previous

        value = np.asarray(value)
        if value.shape != self.shape:
            msg = 'Unexpected quantizer shape'
            raise ValueError(msg)
        elif (np.any(np.isnan(value)) or np.any(~np.isfinite(value))):
            msg = 'Invalid quantizer (NaN or inf)'
            raise ValueError(msg)
        elif np.any(np.diff(value) <= 0.):
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
        out = -2. * self.previous.probability.dot(
            self._integral[1]
            - self._integral[0]
            * self.value[np.newaxis, :])
        return out

    @property
    def hessian(self):
        """
        Returns a self.size-by-self.size np.array containing the
        Hessian of self.distortion
        """
        with np.errstate(invalid='ignore'):
            # (minus, plus)
            terms = ((self.value - self.voronoi[:-1])*self._delta[:, :-1],
                     (self.voronoi[1:] - self.value)*self._delta[:, 1:])
            # Catch inf*zero indetermination at voronoi=np.inf...
            terms[1][:, -1] = 0.

        # Build Hessian
        hess = np.zeros((self.size, self.size))
        main_diagonal = -2. * self.previous.probability.dot(
            terms[0] + terms[1] - self._integral[0])
        diag_view(hess, k=0)[:] = main_diagonal
        if self.size > 1:
            off_diagonal = -2. * self.previous.probability.dot(
                terms[0][:, 1:])
            diag_view(hess, k=1)[:] = off_diagonal
            diag_view(hess, k=-1)[:] = off_diagonal
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
                - 2.*self._integral[1]*self.value[np.newaxis, :]
                + self._integral[0]*self.value[np.newaxis, :]**2.)

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
        BOTTLENECK
        """
        roots = self._roots
        previous_value = self.previous.value[:, np.newaxis]
        cdf = self._cdf
        pdf = self._pdf

        def _get_integral(order):
            integral = [self.model.one_step_expectation_until(
                previous_value, roots[right], order=order,
                _pdf=pdf[right], _cdf=cdf[right])
                for right in [False, True]]
            out = integral[True] - integral[False]
            out[self._no_roots] = 0.0
            return np.diff(out, n=1, axis=1)
        return [_get_integral(order) for order in [0, 1, 2]]

    @lazy_property
    def _delta(self):
        """
        Returns a previous.size-by-size+1 array containing
        ```
            \\delta_{t}^{(ij\\pm)}
        ```
        for all i,j.
        Rem. Output only has valid entries, ie free of NaNs, but may be
            infinite (highly unlikely)
        In particular,
            (1) When no root exists (eg. voronoi==0), delta is zero
            (2) When voronoi==+np.inf, delta is zero
            (3) When voronoi is at a root singularity, delta is np.inf
        """
        unsigned_root_derivative = \
            self.model.one_step_roots_unsigned_derivative(
                self.previous.value[:, np.newaxis],
                self.voronoi[np.newaxis, :])
        limit_index = unsigned_root_derivative == np.inf
        if np.any(limit_index):
            warnings.warn(
                "Voronoi tiles at singularity detected",
                UserWarning
            )
        out = (0.5*unsigned_root_derivative
               * (self._pdf[True]+self._pdf[False]))
        # Limit cases
        # (1)  When a root does not exist, delta is 0
        out[self._no_roots] = 0.
        # (2) When voronoi is +np.inf (which always occur
        #   at the last column), delta is zero.
        out[:, -1] = 0.
        # (3) When roots approach the singularity, delta is np.inf
        out[limit_index] = np.inf
        return out

    @lazy_property
    def _pdf(self):
        """
        Returns a len-2 tuples containing (left, right) PDF of roots
        """
        roots = self._roots
        with np.errstate(invalid='ignore'):
            return [norm.pdf(roots[right]) for right in [False, True]]

    @lazy_property
    def _cdf(self):
        """
        Returns a len-2 tuples containing (left, right) CDF roots
        """
        roots = self._roots
        with np.errstate(invalid='ignore'):
            return [norm.cdf(roots[right]) for right in [False, True]]

    @lazy_property
    def _roots(self):
        """
        Returns a len-2 tuples containing (left, right) roots
        """
        return self.model.one_step_roots(
            self.previous.value[:, np.newaxis],
            self.voronoi[np.newaxis, :])

    @lazy_property
    def _no_roots(self):
        return np.isnan(self._roots[0])


class _AtSingularity(Exception):
    def __init__(self, idx):
        self.idx = np.unique(idx)


class _Unconstrained(_Quantizer):

    def __init__(self, parent, x):
        if not isinstance(parent, _QuantizerFactory):
            msg = 'Quantizers must be instantiated from valid factory'
            raise ValueError(msg)
        x = np.asarray(x)
        self.x = x
        self.min_variance = parent.min_variance
        self._scaled_exp_x, value = self._change_of_variable(parent, x)
        # Let super catch invalid value...
        super().__init__(parent, value)

    @staticmethod
    def _change_of_variable(parent, x):
        scaled_exp_x = np.exp(x)*parent.min_variance / (parent.size+1.)
        value = (parent.min_variance + np.cumsum(scaled_exp_x))
        return (scaled_exp_x, value)

    @staticmethod
    def _inv_change_of_variable(parent, value):
        padded_values = np.insert(value, 0, parent.min_variance)
        expx = (
            (parent.size+1.)*np.diff(padded_values)/parent.min_variance
        )
        return np.log(expx)

    @property
    def gradient(self):
        """
        Returns the transformed gradient for the distortion function, ie
        dDistortion/ dx = dDistortion/dGamma * dGamma/dx
        """
        grad = super().gradient
        out = grad.dot(self._jacobian)
        return out

    @property
    def hessian(self):
        """
        Returns the transformed Hessian for the distortion function
        """
        s = super()

        def get_first_term():
            jac = self._jacobian
            hess = s.hessian
            at_singularity = np.logical_not(np.isfinite(hess))
            if np.any(at_singularity):
                w = np.where(at_singularity)
                raise _AtSingularity(w[1])
            return jac.T.dot(hess).dot(jac)

        def get_second_term():
            inv_cum_grad = np.flipud(np.cumsum(np.flipud(s.gradient)))
            return np.diag(self._scaled_exp_x * inv_cum_grad)

        hess = get_first_term() + get_second_term()
        return hess

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
        return self._scaled_exp_x[np.newaxis, :]*mask
