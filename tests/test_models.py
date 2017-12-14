import unittest
from functools import partial

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

import pymaat.testing
import pymaat.models

VAR_LEVEL = pymaat.models.VAR_LEVEL
VOL_LEVEL = np.sqrt(pymaat.models.VAR_LEVEL)
RET_EXP = 0.06/252.

class TestTimeseriesGarchFixture():

    __innovations = np.array(
            [[[ 0.31559906, -1.80551443, -0.49468605, -1.04842416],
            [ 1.47092134, -0.18375408,  0.61439572,  0.64938352]],

           [[-1.62316804, -0.76478879, -0.75826613,  0.17750069],
            [ 0.82250737,  0.10701289, -1.15691368,  1.05573836]],

           [[-0.22774868,  0.74240245,  1.11521215,  1.10464703],
            [-0.56968483,  0.54138324, -0.69508153, -0.14254879]]])

    __returns = __innovations * VOL_LEVEL + RET_EXP

    __variances = VAR_LEVEL  * np.array(
            [[[ 3.40880883,  1.50217308,  1.11125856,  0.14462167],
            [ 0.08321133,  0.36507093,  1.52133949,  1.42666615]],

           [[ 0.78940018,  0.96257061,  0.3341445 ,  0.59402684],
            [ 1.67863795,  1.21765875,  1.50332767,  1.03224149]],

           [[ 0.78179542,  1.32126548,  1.64830936,  1.59253612],
            [ 0.21397881,  0.01596482,  2.48993567,  1.3034988 ]]])

    __first_variance = __variances[0]

    __shape = __innovations.shape

    def timeseries_filter(self,returns=__returns):
        return self.model.timeseries_filter(
                returns,
                self.__first_variance)

    def timeseries_simulate(self, innovations=__innovations):
        return self.model.timeseries_simulate(
                innovations,
                self.__first_variance)

    def test_simulate_revertsto_filter(self):
        simulated_variances, simulated_returns = self.timeseries_simulate()
        filtered_variances, filtered_innovations = \
                self.timeseries_filter(returns=simulated_returns)
        self.assert_almost_equal(simulated_variances, filtered_variances)
        self.assert_almost_equal(self.__innovations, filtered_innovations)

    def test_filter_initialize_variance(self):
        filtered_variance, _ = self.timeseries_filter()
        self.assert_almost_equal(filtered_variance[0],
                self.__first_variance)

    def test_filter_simulate_variance(self):
        simulated_variance, _ = self.timeseries_simulate()
        self.assert_almost_equal(simulated_variance[0],
                self.__first_variance)

    def test_filter_size(self):
        next_variances, innovations = self.timeseries_filter()
        self.assert_equal(
                next_variances.shape,
                (self.__shape[0]+1,) + self.__shape[1:])
        self.assert_equal(innovations.shape, self.__shape)

    def test_filter_positive_variance(self):
        variances,_ = self.timeseries_filter()
        self.assert_equal(variances>0, True)

    def test_filter_throws_exception_when_non_positive_variance(self):
        with self.assertRaises(ValueError):
            self.model.timeseries_filter(np.nan, -1)
        with self.assertRaises(ValueError):
            self.model.timeseries_filter(np.nan, 0)

    def test_simulate_size(self):
        variances, returns = self.timeseries_simulate()
        self.assert_equal(variances.shape,
                (self.__shape[0]+1,) + self.__shape[1:])
        self.assert_equal(returns.shape, self.__shape)

    def test_simulate_positive_variance(self):
        variances,_ = self.timeseries_simulate()
        self.assert_equal(variances>0, True)

    def test_simulate_throws_exception_when_non_positive_variance(self):
        with self.assertRaises(ValueError):
            self.model.timeseries_simulate(np.nan, -1)
        with self.assertRaises(ValueError):
            self.model.timeseries_simulate(np.nan, 0)

class TestOneStepGarchFixture():

    __innovations = np.array(
            [[[ 0.36841639, -0.57224794,  0.25019277, -1.24721141],
            [ 1.35128646, -0.92096528, -0.92572966, -0.98178342],
            [-0.84366922, -1.28207108, -1.53467485, -0.22874378]],

            [[-1.05050049, -1.31158806, -1.17961533,  0.05363092],
            [-1.67960499,  0.02364286,  0.28870878,  0.01871441],
            [ 1.71135472,  0.7167683 , -0.93715744, -0.8482234 ]]])

    __returns = __innovations * VOL_LEVEL + RET_EXP

    __variances =  VAR_LEVEL * np.array(
            [[[ 0.3217341 ,  0.00868056,  1.22722262,  0.58144423],
            [ 0.54204103,  3.68888772,  0.77486123,  0.95989976],
            [ 0.90888372,  0.07370864,  2.91764007,  2.21766175]],

            [[ 0.7019393 ,  2.38621637,  1.24599797,  1.17731278],
            [ 0.81830769,  0.26109424,  0.2980641 ,  0.85817885],
            [ 0.93831248,  0.17303652,  0.45056502,  2.22328546]]])

    __shape = __innovations.shape

    __innovations_broadcast = np.array([[-0.70687441],
            [ 0.72613442],
            [-2.20412955],
            [-0.27571779],
            [ 1.38167603],
            [-0.96744496],
            [-0.05243877],
            [-0.41254891],
            [-1.27985463],
            [ 0.12262462]])

    __variances_broadcast =  VAR_LEVEL * np.array([[ 2.70588279,  0.96680575,
            0.09027765, 1.23017577,  1.14468471]])

    __shape_broadcast = (__innovations_broadcast.shape[0],
            __variances_broadcast.shape[1])

    # I/O

    def get_one_step_funcs(self):
        return [self.model.one_step_filter,
               self.model.one_step_simulate,
               self.model.one_step_has_roots,
               self.model.one_step_roots,
               self.model.one_step_roots_unsigned_derivative,
               self.model.one_step_expectation_until]

    def test_all_funcs_support_scalar(self):
        allowed_primitives = (bool, float)
        for f in self.get_one_step_funcs():
        # Actual inputs do not matter here
        # Only testing shapes...
            out = f(np.nan, np.nan)
            if isinstance(out, tuple):
                for o in out:
                    msg_ = 'Func was {}, Type was: {}'.format(f, type(o))
                    self.assert_true(type(o) in allowed_primitives,
                            msg=msg_)
            else:
                msg_ = 'Func was {}, Type was: {}'.format(f, type(out))
                self.assert_true(type(out) in allowed_primitives,
                            msg=msg_)

    def test_all_funcs_support_array(self):
        for f in self.get_one_step_funcs():
        # Actual inputs do not matter here
        # Only testing shapes...
            msg_ = 'Func was {}'.format(f)
            out = f(self.__innovations, self.__variances)
            if isinstance(out, tuple):
                for o in out:
                    self.assert_equal(o.shape, self.__shape, msg=msg_)
            else:
                self.assert_equal(out.shape, self.__shape, msg=msg_)

    def test_all_one_step_funcs_support_broadcasting(self):
        for f in self.get_one_step_funcs():
            # Actual inputs do not matter here
            # Only testing shapes...
            msg_ = 'Func was {}'.format(f)
            out = f(self.__innovations_broadcast,
                    self.__variances_broadcast)
            if isinstance(out, tuple):
                for o in out:
                    self.assert_equal(o.shape, self.__shape_broadcast,
                            msg=msg_)
            else:
                self.assert_equal(out.shape, self.__shape_broadcast, msg=msg_)

    # One-step filter

    def one_step_filter(self, returns=__returns):
        return self.model.one_step_filter(returns, self.__variances)

    def one_step_simulate(self, innovations=__innovations):
        return self.model.one_step_simulate(innovations, self.__variances)

    def test_one_step_simulate_revertsto_filter(self):
        simulated_variances, simulated_returns = self.one_step_simulate()
        filtered_variances, filtered_innovations = self.one_step_filter(
                returns=simulated_returns)
        self.assert_almost_equal(simulated_variances, filtered_variances)
        self.assert_almost_equal(self.__innovations, filtered_innovations)

    def test_one_step_filter_positive_variance(self):
        next_variances, _ = self.one_step_filter()
        self.assert_equal(next_variances>0, True)

    def test_one_step_simulate_positive_variance(self):
        next_variances, _ = self.one_step_simulate()
        self.assert_equal(next_variances>0, True)

    # One-step innovation roots

    def get_lowest_one_step_variance(self):
        return self.model._get_lowest_one_step_variance(self.__variances)

    def get_valid_one_step(self):
        next_variances = self.__variances
        variances = np.min(next_variances)*0.9
        return (variances, next_variances)

    def get_invalid_one_step(self):
        variances = self.__variances
        next_variances = 0.9*self.model._get_lowest_one_step_variance(
                self.__variances)
        return (variances, next_variances)

    def test_one_step_roots_when_valid(self):
        (variances, next_variances) = self.get_valid_one_step()
        left_roots, right_roots = self.model.one_step_roots(
                variances,
                next_variances)
        for (z,s) in zip((left_roots, right_roots), ('left', 'right')):
            next_variances_solved, _ = self.model.one_step_simulate(
                z, variances)
            self.assert_almost_equal(next_variances_solved, next_variances,
                    msg='Invalid ' + s + ' roots.')

    def test_one_step_roots_left_right_order(self):
        (variances, next_variances) = self.get_valid_one_step()
        left_roots, right_roots = self.model.one_step_roots(
                variances,
                next_variances)
        self.assert_equal(left_roots<right_roots, True)

    def test_roots_nan_when_invalid(self):
        (variances, next_variances) = self.get_invalid_one_step()
        left_roots, right_roots = self.model.one_step_roots(
                self.__variances, next_variances)
        self.assert_equal(left_roots, np.nan)
        self.assert_equal(right_roots, np.nan)

    def test_same_roots_at_singularity(self):
        variances_at_singularity = self.get_lowest_one_step_variance()
        left_roots, right_roots = self.model.one_step_roots(
                self.__variances, variances_at_singularity)
        self.assert_almost_equal(left_roots, right_roots)

    def test_roots_at_inf_are_pm_inf(self):
        [left_root, right_root] = self.model.one_step_roots(
                self.__variances, np.inf)
        self.assert_equal(left_root, -np.inf)
        self.assert_equal(right_root, np.inf)

    def test_roots_at_zero_are_nan(self):
        [left_root, right_root] = self.model.one_step_roots(
                self.__variances, 0)
        self.assert_equal(left_root, np.nan)
        self.assert_equal(right_root, np.nan)

    def test_roots_derivatives_when_valid(self):
        (variances, at) = self.get_valid_one_step()
        def roots_func(next_variances):
            _, z = self.model.one_step_roots(
                    variances,
                    next_variances)
            return z
        def roots_der(next_variances):
            dz = self.model.one_step_roots_unsigned_derivative(
                    variances,
                    next_variances)
            return dz
        self.assert_derivative_at(roots_der, roots_func, at,
                mode='forward', rtol=1e-4)

    def test_roots_derivative_is_positive(self):
        (variances, next_variances) = self.get_valid_one_step()
        der = self.model.one_step_roots_unsigned_derivative(
                variances,
                next_variances)
        self.assert_true(der>0, True)

    def test_roots_derivatives_nan_when_invalid(self):
        (variances, next_variances) = self.get_invalid_one_step()
        der = self.model.one_step_roots_unsigned_derivative(
                variances, next_variances)
        self.assert_equal(der, np.nan)

    def test_roots_derivatives_is_nan_at_singularity(self):
        variances_at_singularity = self.get_lowest_one_step_variance()
        der = self.model.one_step_roots_unsigned_derivative(
                self.__variances, variances_at_singularity)
        self.assert_equal(der, np.nan)

    def test_roots_derivatives_at_inf_are_null(self):
        der = self.model.one_step_roots_unsigned_derivative(
                self.__variances, np.inf)
        self.assert_equal(der, 0.)

    def test_roots_derivatives_at_zero_are_nan(self):
        der = self.model.one_step_roots_unsigned_derivative(
                self.__variances, 0)
        self.assert_equal(der, np.nan)

    # One-step variance integration
    __test_expectation_values = np.array(
            [0.123,-0.542,-1.3421,1.452,-3.412,5.234])
    def test_one_step_expectation_until(self):
        def func_to_integrate(z):
            (h, _) = self.model.one_step_simulate(z, VAR_LEVEL)
            return h * norm.pdf(z)
        integral = partial(self.model.one_step_expectation_until, VAR_LEVEL)
        self.assert_integral_until(integral, func_to_integrate,
                self.__test_expectation_values, rtol=1e-1)

    def test_one_step_expectation_is_integral_until(self):
        integral = partial(self.model.one_step_expectation_until, VAR_LEVEL)
        self.assert_is_integral_until(integral)

    def test_one_step_expectation_squared_until(self):
        def func_to_integrate(z):
            (h, _) = self.model.one_step_simulate(z, VAR_LEVEL)
            return h**2. * norm.pdf(z)
        integral = partial(self.model.one_step_expectation_until, VAR_LEVEL,
                order=2)
        self.assert_integral_until(integral, func_to_integrate,
                self.__test_expectation_values, rtol=1e-4)

    def test_one_step_expectation_squared_is_integral_until(self):
        integral = partial(self.model.one_step_expectation_until, VAR_LEVEL,
                order=2)
        self.assert_is_integral_until(integral)

class TestGarch(TestTimeseriesGarchFixture,
        TestOneStepGarchFixture,
        pymaat.testing.TestCase):

    model = pymaat.models.Garch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)

    def test_one_step_filter_at_few_values(self):
        next_var, innov = self.model.one_step_filter(0, 1)
        self.assert_almost_equal(innov, 0.5-self.model.mu)
        self.assert_almost_equal(next_var,
                  self.model.omega
                + self.model.beta
                + self.model.alpha * (innov - self.model.gamma) ** 2)

    def test_one_step_simulate_at_few_values(self):
        next_var, ret = self.model.one_step_simulate(0, 0)
        self.assert_almost_equal(ret, 0)
        self.assert_almost_equal(next_var, self.model.omega)
        next_var, ret = self.model.one_step_simulate(0, 1)
        self.assert_almost_equal(ret, self.model.mu-0.5)
        self.assert_almost_equal(next_var,
                  self.model.omega
                + self.model.beta
                + self.model.alpha * self.model.gamma ** 2)

    def test_invalid_param_raise_exception(self):
        with self.assertRaises(ValueError):
            pymaat.models.Garch(
                mu=0,
                omega=0,
                alpha=0.5,
                beta=0.5,
                gamma=1)

    def test_one_step_expectation_cdf_factor(self):
        value = self.model._one_step_expectation_cdf_factor(VAR_LEVEL)
        expected_value = (self.model.omega + self.model.alpha
                + (self.model.beta+
                self.model.alpha*self.model.gamma**2) * VAR_LEVEL)
        self.assert_almost_equal(value, expected_value)

    def test_one_step_expectation_squared_cdf_factor(self):
        value = self.model._one_step_expectation_squared_cdf_factor(
                VAR_LEVEL)
        a = (self.model.omega +
                (self.model.beta+self.model.alpha*self.model.gamma**2.)
                *VAR_LEVEL)
        b = 2.*self.model.alpha*self.model.gamma*VAR_LEVEL**0.5
        c = self.model.alpha
        expected_value = (a**2. + 2.*a*c + b**2. + 3.*c**2)
        self.assert_almost_equal(value, expected_value)

    def test_one_step_expectation_pdf_factor(self):
        innovations = 1.453
        value = self.model._one_step_expectation_pdf_factor(
                VAR_LEVEL, innovations)
        expected_value = (self.model.alpha * (2.*self.model.gamma*VOL_LEVEL-
                innovations))
        self.assert_almost_equal(value, expected_value)

    def test_one_step_expectation_squared_pdf_factor(self):
        z = 1.453
        value = self.model._one_step_expectation_squared_pdf_factor(
                VAR_LEVEL, z)
        a = (self.model.omega +
                (self.model.beta+self.model.alpha*self.model.gamma**2.)
                *VAR_LEVEL)
        b = 2.*self.model.alpha*self.model.gamma*VAR_LEVEL**0.5
        c = self.model.alpha
        expected_value = (2.*a*b - 2.*a*c*z - b**2.*z + 2.*b*c*(z**2.+2.)
                -c**2.*z*(z**2.+3.))
        self.assert_almost_equal(value, expected_value)

    # TODO: send to estimator
    # def test_neg_log_like_at_few_values(self):
    #     nll = self.model.negative_log_likelihood(1, 1)
    #     self.assert_almost_equal(nll, 0.5)
    #     nll = self.model.negative_log_likelihood(0, np.exp(1))
    #     self.assert_almost_equal(nll, 0.5)
class TestGarchQuantizer(pymaat.testing.TestCase):

    model = pymaat.models.Garch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196.21)

    nper = 3
    nquant = 4
    quant = pymaat.models.Quantizer(model, nper=nper, nquant=nquant)

    prev_grid_sorted = np.sort((0.18**2/252)
            * np.array([0.50873928,0.55812529,1,1.73936377]))
    prev_proba = np.array([0.05,0.25,0.5,0.20])
    assert prev_proba.sum()==1.
    most_proba = 2 # ID to most probable in previous grid

    grid_sorted, _ = np.sort(model.one_step_simulate(
            np.array([0.1932423,-0.23491,0.93414,-0.5323434]),
            1e-4))

    def quantized_integral(self, type_, prev_grid, grid):
        if prev_grid.size>1:
            I = np.empty((prev_grid.size, grid.size))
            for (i,pg) in enumerate(prev_grid):
                 I[i] = self.quantized_integral(type_, pg, grid)
            return I
        else:
            assert prev_grid.size==1
            if type_ == 'distortion':
                func = lambda z,h,g: np.power(h-g,2) * norm.pdf(z)
            elif type_ == 'gradient':
                func = lambda z,h,g: (h-g) * norm.pdf(z)
            elif type_ == 'variance_squared':
                func = lambda z,h,g: h**2. * norm.pdf(z)
            elif type_ == 'variance':
                func = lambda z,h,g: h * norm.pdf(z)
            elif type_ == 'pdf':
                func = lambda z,h,g: norm.pdf(z)
            voronoi = self.quant._get_voronoi(grid[np.newaxis,:])
            voronoi = voronoi[0]
            def do_integration(bounds, pg, g):
                assert g>bounds[0] and g<bounds[1]
                def function_to_integrate(z):
                    # 0 if between bounds else integral input
                    (h, _) = self.model.one_step_simulate(z, pg)
                    if h<bounds[0] or h>bounds[1]:
                        return 0
                    else:
                        return func(z,h,g)
                # Help integration by providing discontinuities
                critical_pts = np.hstack([
                    self.model.one_step_roots(pg, b) for b in bounds])
                critical_pts = critical_pts[np.isfinite(critical_pts)]
                # Can safely neglect when z outside [-30,30]
                out = integrate.quad(function_to_integrate, -30, 30,
                        points=critical_pts)
                return out[0]
            return np.array([do_integration((lb,ub), prev_grid, g)
                for (lb,ub,g) in zip(voronoi[:-1], voronoi[1:], grid)])

    def marginal_distortion(self, x):
        grid = self.quant._optim_inv_transform(self.prev_grid_sorted, x)
        dist = self.quantized_integral(
                'distortion', self.prev_grid_sorted, grid)
        return self.prev_proba.dot(dist).sum()/VAR_LEVEL**2.

    def test_init(self):
        self.assert_equal(self.quant.nper, self.nper)
        self.assert_equal(self.quant.nquant, self.nquant)

    def test_initialize(self):
        first_variance = 1e-4
        grid, proba, trans = self.quant._initialize(first_variance)
        # Shapes
        self.assert_equal(grid.shape, (self.nper+1, self.nquant))
        self.assert_equal(proba.shape, (self.nper+1, self.nquant))
        self.assert_equal(trans.shape,
                (self.nper, self.nquant, self.nquant))
        # First grid
        self.assert_equal(grid[0,1:]==first_variance, True)

    def test_get_voronoi_2d(self):
        v = self.quant._get_voronoi(np.array(
                [[1,2,3],
                [11,12,13]]))
        self.assert_almost_equal(v, np.array(
                [[0,1.5,2.5,np.inf],
                [0,11.5,12.5,np.inf]]))

    def test_zeroth_order_integral(self):
        roots = self.quant._get_voronoi_roots(
                self.prev_grid_sorted[:,np.newaxis],
                self.grid_sorted[np.newaxis,:])
        I0 = self.quant._get_integral(
                self.prev_grid_sorted[:,np.newaxis],
                roots,
                order=0)
        expected_value = self.quantized_integral(
                'pdf', self.prev_grid_sorted, self.grid_sorted)
        self.assert_almost_equal(I0, expected_value, rtol=1e-4,
                msg='Incorrect pdf integral (I0)')

    def test_first_order_integral(self):
        roots = self.quant._get_voronoi_roots(
                self.prev_grid_sorted[:,np.newaxis],
                self.grid_sorted[np.newaxis,:])
        I1 = self.quant._get_integral(
                self.prev_grid_sorted[:,np.newaxis],
                roots,
                order=1)
        expected_value = self.quantized_integral(
                'variance', self.prev_grid_sorted, self.grid_sorted)
        self.assert_almost_equal(I1, expected_value, rtol=1e-4,
                msg='Incorrect model integral (I1)')

    def test_second_order_integral(self):
        roots = self.quant._get_voronoi_roots(
                self.prev_grid_sorted[:,np.newaxis],
                self.grid_sorted[np.newaxis,:])
        I2 = self.quant._get_integral(
                self.prev_grid_sorted[:,np.newaxis],
                roots,
                order=2)
        expected_value = self.quantized_integral(
                'variance_squared', self.prev_grid_sorted, self.grid_sorted)
        self.assert_almost_equal(I2, expected_value, rtol=1e-4,
                msg='Incorrect model squared integral (I2)')
    
        # TODO CLEAN NEXT 4 BLOCKS
    def assert_integral_derivative_minus_one_lag(self, factor, vor, order=0):
        if order==0:
            type_='pdf'
        elif order==1:
            type_='variance'
        elif order==2:
            type_='variance_squared'
        for j in range(self.nquant-1):
            def minus_one_lag_func(h):
                new_grid = self.grid_sorted.copy()
                new_grid[j] = h
                I = self.quantized_integral(
                        type_, self.prev_grid_sorted, new_grid)
                return I[:,j+1]
            def minus_one_lag_derivative(h):
                out = self.quant._get_integral_derivative(
                        factor, vor, order=order, lag=-1)
                return out[:,j+1]
            self.assert_derivative_at(minus_one_lag_derivative,
                    minus_one_lag_func, self.grid_sorted[j], rtol=1e-5)

    def assert_integral_derivative_zero_lag(self, factor, vor, order=0):
        if order==0:
            type_='pdf'
        elif order==1:
            type_='variance'
        elif order==2:
            type_='variance_squared'
        for j in range(self.nquant):
            def zero_lag_func(h):
                new_grid = self.grid_sorted.copy()
                new_grid[j] = h
                I = self.quantized_integral(
                        type_, self.prev_grid_sorted, new_grid)
                return I[:,j]
            def zero_lag_derivative(h):
                out = self.quant._get_integral_derivative(
                        factor, vor, order=order, lag=0)
                return out[:,j]
            self.assert_derivative_at(zero_lag_derivative,
                    zero_lag_func, self.grid_sorted[j], rtol=1e-5)

    def assert_integral_derivative_one_lag(self, factor, vor, order=0):
        if order==0:
            type_='pdf'
        elif order==1:
            type_='variance'
        elif order==2:
            type_='variance_squared'
        for j in range(self.nquant-1):
            def minus_one_lag_func(h):
                new_grid = self.grid_sorted.copy()
                new_grid[j+1] = h
                I = self.quantized_integral(
                        type_, self.prev_grid_sorted, new_grid)
                return I[:,j]
            def minus_one_lag_derivative(h):
                out = self.quant._get_integral_derivative(
                        factor, vor, order=order, lag=1)
                return out[:,j]
            self.assert_derivative_at(minus_one_lag_derivative,
                    minus_one_lag_func, self.grid_sorted[j+1], rtol=1e-5)

    def test_integral_derivative(self):
        vor = self.quant._get_voronoi(self.grid_sorted[np.newaxis,:])
        roots = self.quant._get_roots(self.prev_grid_sorted[:,np.newaxis],
                vor)
        factor = self.quant._get_factor_of_integral_derivative(
                self.prev_grid_sorted[:,np.newaxis], vor, roots)
        self.assert_integral_derivative_minus_one_lag(factor, vor, order=0)
        self.assert_integral_derivative_minus_one_lag(factor, vor, order=1)
        self.assert_integral_derivative_minus_one_lag(factor, vor, order=2)
        self.assert_integral_derivative_zero_lag(factor, vor, order=0)
        self.assert_integral_derivative_zero_lag(factor, vor, order=1)
        self.assert_integral_derivative_zero_lag(factor, vor, order=2)
        self.assert_integral_derivative_one_lag(factor, vor, order=0)
        self.assert_integral_derivative_one_lag(factor, vor, order=1)
        self.assert_integral_derivative_one_lag(factor, vor, order=2)

    def test_trans_reverts(self):
        x = np.array([-0.213,0.432,0.135,0.542])
        grid = self.quant._optim_inv_transform(self.prev_grid_sorted, x)
        x_ = self.quant._optim_transform(self.prev_grid_sorted, grid)
        self.assert_almost_equal(x, x_)

    def assert_inv_trans_is_in_space_for(self, x):
        x = np.array(x)
        grid = self.quant._optim_inv_transform(self.prev_grid_sorted, x)
        h_ = self.quant._get_minimal_variance(self.prev_grid_sorted)
        self.assert_equal(np.diff(grid)>0, True, msg='is not sorted')
        self.assert_true(grid[1]>h_)
        self.assert_true(0.5*(grid[0]+grid[1])>h_)

    def test_inv_trans_is_in_space(self):
        self.assert_inv_trans_is_in_space_for([-0.213,0.432,0.135,0.542])
        self.assert_inv_trans_is_in_space_for([-5.123,-3.243,5.234,2.313])
        self.assert_inv_trans_is_in_space_for([3.234,-6.3123,-5.123,0.542])

    def test_trans_jacobian(self):
        x = np.array([-0.213,0.432,0.135,0.542])
        jac = partial(self.quant._optim_jacobian,
                self.prev_grid_sorted)
        func = partial(self.quant._optim_inv_transform,
                self.prev_grid_sorted)
        self.assert_jacobian_at(jac, func, x, rtol=1e-6, atol=1e-8)

    def test_one_step_distortion(self):
        at = self.quant._optim_transform(self.prev_grid_sorted,
            self.grid_sorted)
        # Test distortion
        value, _, _ = self.quant._one_step_distortion(
                    self.prev_grid_sorted,
                    self.prev_proba,
                    at)
        expected_value = self.marginal_distortion(at)
        self.assert_almost_equal(value, expected_value, rtol=1e-2,
                msg='Incorrect distortion')

    def test_one_step_distortion_gradient(self):
        at = self.quant._optim_transform(self.prev_grid_sorted,
            self.grid_sorted)
        def gradient(grid):
                _, g, _ = self.quant._one_step_distortion(
                    self.prev_grid_sorted,
                    self.prev_proba,
                    grid)
                return g
        func = self.marginal_distortion
        self.assert_gradient_at(gradient, func, at, rtol=1e-2)

    def test_one_step_distortion_hessian(self):
        at = self.quant._optim_transform(self.prev_grid_sorted,
            self.grid_sorted)
        def hessian(grid):
                _, _, h = self.quant._one_step_distortion(
                    self.prev_grid_sorted,
                    self.prev_proba,
                    grid)
                return h
        func = self.marginal_distortion
        self.assert_hessian_at(hessian, func, at, rtol=1e-6)

    def test_trans_proba_size(self):
        trans = self.quant._transition_probability(
                self.prev_grid_sorted, self.grid_sorted)
        self.assert_equal(trans.shape, (self.nquant, self.nquant))

    def test_trans_proba_sum_to_one_and_non_negative(self):
        trans = self.quant._transition_probability(
                self.prev_grid_sorted, self.grid_sorted)
        self.assert_equal(trans>=0, True)
        self.assert_almost_equal(np.sum(trans,axis=1),1)

    def test_trans_proba(self):
        value = self.quant._transition_probability(
                self.prev_grid_sorted,
                self.grid_sorted)
        expected_value = self.quantized_integral(
                'pdf', self.prev_grid_sorted, self.grid_sorted)
        self.assert_almost_equal(value, expected_value)

    def test_transition_probability_from_first_grid(self):
        first_variance = 1e-4
        grid, *_ = self.quant._initialize(first_variance)
        trans = self.quant._transition_probability(grid[0],self.grid_sorted)
        first_row = trans[0]
        self.assert_equal(trans==first_row, True)

    # def test_one_step_quantize_sorted(self):
    #     (_, new_grid) = self.quant._one_step_quantize(
    #             self.prev_grid_sorted, self.prev_proba)
    #     self.assert_almost_equal(new_grid, np.sort(new_grid))
#     def test_quantize(self):
#         quant = pymaat.models.Quantizer(self.model, nquant=10)
#         quant.quantize((0.18**2)/252)
