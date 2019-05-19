import warnings

import numpy as np
from scipy.optimize import brentq

from pymaat.nputil import ravel
from pymaat.mathutil import normcdf, norminv

# Black and Scholes Utilities


@ravel
def price(money, total_volatility, put=False):
    logmoney = np.log(money)
    if put:
        d1 = -logmoney/total_volatility + 0.5*total_volatility
        d2 = -logmoney/total_volatility - 0.5*total_volatility
        Nd1 = normcdf(d1)
        Nd2 = normcdf(d2)
        return money*(1.-Nd2) - (1.-Nd1)
    else:
        d1 = -logmoney/total_volatility + 0.5*total_volatility
        d2 = -logmoney/total_volatility - 0.5*total_volatility
        Nd1 = normcdf(d1)
        Nd2 = normcdf(d2)
        return Nd1 - money*Nd2

# function [delta, price] = modelFreeDeltaHedging(optSurf, date, forwards, strikes)
# nPer = numel(forwards)-1;
# nStrikes = numel(strikes);
# strikes = reshape(strikes,[1, nStrikes]);
# %%
# price = nan(nPer, nStrikes);
# delta = nan(nPer, nStrikes);
# assert(numel(date)==nPer)
# for t = 1:nPer
#     price(t,:) = optSurf.getPrice(date(t), ones(size(strikes))*(nPer+1-t), strikes);
#     ddKtmp = optSurf.getDelta(date(t), ones(size(strikes))*(nPer+1-t), strikes);
#     delta(t,:) = (1/forwards(t)) * (price(t,:)-strikes.*ddKtmp);
# end
# end


@ravel
def brent_dekker(
        money, price, put=False,
        search_span=1, dnc=np.nan, rtol=1e-6):
    """
    Computes the implicit total volatility from normalized
      option prices and moneynesses using a root-finding algorithm

    Inputs:
      [`money` `price`] have the same number of elements, where:
          `money` is `StrikePrice/ForwardPrice`
          `price` is  `OptionPrice*BankAccountValue/ForwardPrice`
      put is a logical (True for put, False for call)
      For best results, set search_span to the maximum
          number of year-to-expiry
      dnc is the value used when the algorithm did not converge
      rtol is the relative tolerance on the total implicit volatility

    Output:
      B&S total implicit volatility parameter,
          i.e. `sigma*sqrt(time-to-expiry)`
    """
    # Default search space is 1% to 500% in total volatility
    # Increase search span for options with more than 1 year to maturity
    space = np.log(np.array([1e-2, 5.])*search_span)
    failsafe = np.array([-100., 10.])
    eps = np.finfo('float').eps
    if price.size != money.size:
        raise ValueError("Log-moneynesses and prices must be of same size")
    logmoney = np.log(money)

    if put:
        def price_fun(money, logmoney, total_volatility):
            d1 = -logmoney/total_volatility + 0.5*total_volatility
            d2 = -logmoney/total_volatility - 0.5*total_volatility
            Nd1 = normcdf(d1)
            Nd2 = normcdf(d2)
            return money*(1.-Nd2) - (1.-Nd1)

        zero_distance = price - np.maximum(money-1., 0.)
    else:
        def price_fun(money, logmoney, total_volatility):
            d1 = -logmoney/total_volatility + 0.5*total_volatility
            d2 = -logmoney/total_volatility - 0.5*total_volatility
            Nd1 = normcdf(d1)
            Nd2 = normcdf(d2)
            return Nd1 - money*Nd2

        zero_distance = price - np.maximum(1.-money, 0.)

    def get_implicit_total_volatility(money, logmoney, price):
        def fun(x):
            return price_fun(money, logmoney, np.exp(x)) - price
        if fun(space[0]) * fun(space[1]) <= 0.:
            # Zero found in default search space
            return brentq(fun, *space, rtol=rtol)
        elif fun(failsafe[0]) * fun(failsafe[1]) <= 0.:
            # Zero found in expanded "failsafe" search space
            return brentq(fun, *failsafe, rtol=rtol)
        else:
            # No zero found
            return dnc

    log_result = np.full_like(money, dnc)
    for i, (zd, k, logk, p) in enumerate(
            zip(zero_distance, money, logmoney, price)):
        if not np.isfinite(p):
            continue
        if np.absolute(zd) < eps:  # catch near-arbitrage opportunities
            log_result[i] = -np.inf
        elif zd > 0.:
            log_result[i] = get_implicit_total_volatility(k, logk, p)

    return np.exp(log_result)


SOR_COEF = np.array([[-0.00006103098165, 1],  # m00, n00
                     [5.33967643357688, 22.96302109010794],  # m01, n01
                     [-0.40661990365427, -0.48466536361620],  # m10, n10
                     [3.25023425332360, -0.77268824532468],  # m02, n02
                     [-36.19405221599028, -1.34102279982050],  # m11, n11
                     [0.08975394404851, 0.43027619553168],  # m20, n20
                     [83.84593224417796, -5.70531500645109],  # m03, n03
                     [41.21772632732834, 2.45782574294244],  # m12, n12
                     [3.83815885394565, -0.04763802358853],  # m21, n21
                     [-0.21619763215668, -0.03326944290044]])  # m30, n30


@ravel
def sor(money, price, put=False, rtol=1e-6, max_iter=10, dnc=np.nan):
    """
    Computes the implicit total volatility from normalized
      option prices and moneynesses using
      the adaptive successive over-relaxaton method of Li:2011
    Use for performance-critical applications

    Inputs:
      [`money` `price`] have the same number of elements, where:
          `money` is `StrikePrice/ForwardPrice`
          `price` is  `OptionPrice*BankAccountValue/ForwardPrice`
      put is a logical (True for put, False for call)

    Output:
      B&S total implicit volatility parameter,
          i.e. `sigma*sqrt(time-to-expiry)`
    """
    if price.size != money.size:
        raise ValueError("Log-moneynesses and prices must be of same size")
    price = np.copy(price)  # protect from in-place operations done next
    logmoney = np.log(money)
    # I. Convert to *calls* only
    if put:
        # .. using put-call parity
        price += 1.-money
    # II. Convert to *out-of-the-money* calls only
    itm = logmoney < 0.
    logmoney = np.absolute(logmoney)  # Out-of-the-money moneyness only...
    money = np.exp(logmoney)  # ... from now on
    # Black-scholes conversion of ITM -> OTM
    # We can easily show that price(1/k,sigma) = 1/k*price(k,sigma)+1-1/k
    # Hence, sigma_star solves p = price(k,sigma_star)
    #   iff p/k + 1 - 1/k = price(1/k, sigma_star)
    price[itm] *= money[itm]
    price[itm] += + 1. - money[itm]
    # III. Solve IV for OTM call options
    # Initialize (clean data)
    result = np.full_like(money, dnc)
    converged = np.isnan(price)
    converged = np.logical_or(converged, price <= 0.)
    converged = np.logical_or(converged, price >= 1.)
    flag = np.logical_not(converged)
    # First Guess (Rational Approx.)

    def get_first_guess(logk, p):
        ones = np.ones((logk.size,))
        quotient = np.column_stack((
            ones, p, -logk, p**2., -logk*p, -logk**2.,
            p**3., -logk*p**2., -logk**2.*p, -logk**3.
        )).dot(SOR_COEF)
        return quotient[:, 0]/quotient[:, 1]
    result[flag] = get_first_guess(logmoney[flag], price[flag])
    # SOR

    def do_sor(starting, logk, price):
            # SOR (omega = 1)
        absX = logk
        x = -logk
        xOverTv = x/starting
        tvOver2 = 0.5*starting
        Nm = np.exp(absX)*normcdf(xOverTv - tvOver2)
        Np = normcdf(xOverTv + tvOver2)
        F = price + Nm + Np
        ninvF = norminv(0.5*F)
        G = ninvF + np.sqrt(ninvF**2. + 2.*absX)
        # Transformation of sequence
        absXTimes2 = 2.*absX
        tvSquared = starting**2.
        phi = (tvSquared - absXTimes2)/(tvSquared + absXTimes2)
        fact = 2./(1.+phi)
        # Results
        price_estimate = Np - Nm
        flag = np.absolute(price_estimate/price-1.) > rtol
        total_volatility = fact*G + (1.-fact)*starting
        return flag, total_volatility
    for _ in range(max_iter):
        if not np.any(flag):  # Early stopping criteria
            break
        else:
            flag_tmp, result_tmp = do_sor(
                result[flag], logmoney[flag], price[flag])
            result[flag] = result_tmp
            # Update flag once result are set
            flag[flag] = flag_tmp
    if np.any(flag):
        result[flag] = dnc
        warnings.warns(
            '{0:d} implicit volatilities did not converge'.format(
                np.sum(flag)))
    return result
