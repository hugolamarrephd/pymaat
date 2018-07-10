import os
import csv
from collections import namedtuple
import math
import pytest
import numpy as np
import pymaat.testing as pt
from scipy.stats import truncnorm
from math import sqrt, ceil
from multiprocessing import Pool

from pymaat.garch.spec.hngarch import HestonNandiGarch
import pymaat.garch.plot
import pymaat.garch.quant
import pymaat.quantutil as qutil
from pymaat.plotutil import get_lim

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar
import time

SYM_MODEL = HestonNandiGarch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.96,
            gamma=0.)

ASYM_MODEL = HestonNandiGarch(
            mu=2.01,
            omega=9.75e-20,
            alpha=4.54e-6,
            beta=0.79,
            gamma=196)

@pytest.fixture
def starting_variance():
    return 0.18**2./252.

@pytest.fixture
def nper():
    return 21

@pytest.fixture
def nsim():
    return 1000000

@pytest.fixture(params=[True, False])
def asym(request):
    return request.param

@pytest.fixture
def model(asym):
    if asym:
        return ASYM_MODEL
    else:
        return SYM_MODEL

@pytest.fixture
def size_P():
    return np.power(2, np.arange(11))

@pytest.fixture
def size_KN(size_P):
    LIMIT = size_P[-1]
    size_K = []
    size_N = []
    for K in size_P:
        for N in size_P:
            if K*N<=LIMIT:
                size_K.append(K)
                size_N.append(N)
    out = (size_K, size_N)
    return out


@pytest.fixture
def size_K(size_KN):
    return size_KN[0]

@pytest.fixture
def size_N(size_KN):
    return size_KN[1]

@pytest.fixture
def simul(starting_variance, model, nper, nsim):
    np.random.seed(789123768)
    return _marginal_simulation(
            starting_variance, model, nper, nsim)


# Utilities
grid_alpha=0.95
grid_style=':'

def _get_lim(*args):
    prices = []
    variances = []
    probabilities= []
    for a in args:
        prices.append(a.quantizer[...,0])
        variances.append(a.quantizer[...,1])
        probabilities.append(a.probability)

    xlim = get_lim(
            *prices,
            bounds=[0.,np.inf],
            base=10,
            margin=0.1,
            formatter=lambda x: 100.*np.maximum(x,0.)
            )
    ylim = get_lim(
            *variances,
            bounds=[5.,np.inf],
            base=5,
            margin=0.1,
            formatter=lambda x: 100.*np.sqrt(252.*np.maximum(x,0.))
            )
    zlim = get_lim(
            *probabilities,
            bounds=[0.,100.],
            precision=1,
            margin=0.25,
            formatter=lambda x: 100.*np.maximum(x,0.))

    # De-format limits for quantization plot utility
    lim = np.concatenate(
            (
                xlim[:,np.newaxis]/100.,
                ((ylim[:,np.newaxis]/100.)**2.)/252.
            ),
            axis=1)
    plim = zlim/100.

    return lim, plim

def _plot_quant(quant, path, *, title=None, show_quantizer=False):
    lim, plim = _get_lim(quant)
    fig = plt.figure(figsize=(8,7), dpi=1000)

    cbar_ax = fig.add_axes([0.87, 0.1, 0.03, 0.8])
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.8])
    pymaat.garch.quant.plot(
           model, quant, ax, cbar_ax, lim=lim, plim=plim,
           show_quantizer=show_quantizer)
    ax.set_xlabel(r'Price (%)')
    ax.set_ylabel(r'Annual Vol. (%)')

    # Set color bar
    cbar_ax.set_title('p (%)')
    if title is not None:
        ax.set_title(title)
    plt.savefig(os.path.expanduser(path), format='pdf')
    plt.close(fig)

def _plot_all_quantizations(product, conditional, markov, marginal,
        path, title=None):
    lim, plim = _get_lim(product, conditional, markov, marginal)

    fig = plt.figure(figsize=(8,7), dpi=1000)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.03, 0.8])

    # Product quantization
    ax = fig.add_subplot(2,2,1)
    ax.set_title('Product')
    pymaat.garch.quant.plot(model, product, ax, lim=lim, plim=plim)
    ax.set_ylabel(r'Annual Vol. (%)')
    ax.xaxis.set_ticks([])

    # Conditional quantization
    ax = fig.add_subplot(2,2,2)
    ax.set_title('Conditional')
    pymaat.garch.quant.plot(model, conditional, ax, lim=lim, plim=plim)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    # Markov CLVQ
    ax = fig.add_subplot(2,2,3)
    ax.set_title('Markov-CLVQ')
    pymaat.garch.quant.plot(model, markov, ax, lim=lim, plim=plim)
    ax.set_xlabel(r'Price (%)')
    ax.set_ylabel(r'Annual Vol. (%)')

    # Marginal CLVQ
    ax = fig.add_subplot(2,2,4)
    cbar_ax = fig.add_axes(cbar_ax)  # Plot color bar
    ax.set_title('Marginal-CLVQ')
    pymaat.garch.quant.plot(model, marginal, ax, cbar_ax, lim=lim, plim=plim)
    ax.set_xlabel(r'Price (%)')
    ax.yaxis.set_ticks([])
    cbar_ax.set_title('p (%)')
    if title is not None:
        ax.set_title(title)
    plt.subplots_adjust(wspace=0.075, hspace=0.125,
            bottom=0.1, right=0.85, top=0.9)
    plt.savefig(os.path.expanduser(path), format='pdf')
    plt.close(fig)

def _plot_sym_vs_asym(path, sym, asym, lim, plim):
    fig = plt.figure(figsize=(10,4.5), dpi=1000)
    cbar_ax = fig.add_axes([0.925, 0.1, 0.03, 0.8])

    ax = fig.add_subplot(1,2,1)
    ax.set_title('Symmetric ($\gamma=0$)')
    pymaat.garch.quant.plot(SYM_MODEL, sym, ax, lim=lim, plim=plim)
    ax.set_ylabel(r'Annual Vol. (%)')
    ax.set_xlabel(r'Price (%)')
    ax.grid(alpha=grid_alpha, linestyle=grid_style)

    ax = fig.add_subplot(1,2,2)
    ax.set_title('Asymmetric ($\gamma=196$)')
    cbar_ax = fig.add_axes(cbar_ax)  # Plot color bar
    cbar_ax.set_title('p (%)')
    pymaat.garch.quant.plot(ASYM_MODEL, asym,
            ax, cbar_ax, lim=lim, plim=plim)
    ax.set_xlabel(r'Price (%)')
    ax.grid(alpha=grid_alpha, linestyle=grid_style)
    plt.subplots_adjust(wspace=0.125, hspace=0.125,
            bottom=0.1, right=0.9, top=0.9)
    plt.savefig(os.path.expanduser(path), format='eps')
    plt.close(fig)

def test_showcase(starting_variance, nsim, nper):
    shape = [5, 5]
    verbose = False
    parallel = True
    size = np.prod(shape)

    # Marginal
    marginal_sym = _do_marginal_quantization_at(
            starting_variance, SYM_MODEL, nper, nsim, size)
    marginal_asym = _do_marginal_quantization_at(
            starting_variance, ASYM_MODEL, nper, nsim, size)

    # Markov
    markov_sym = _do_markov_quantization(
            starting_variance, SYM_MODEL, nper, nsim, size)[-1]
    markov_asym = _do_markov_quantization(
            starting_variance, ASYM_MODEL, nper, nsim, size)[-1]

    # Product
    tmp = pymaat.garch.quant.PriceVarianceGrid(
            SYM_MODEL,
            starting_variance,
            price_size=shape[0],
            variance_size=shape[1],
            nper=nper,
            verbose=verbose
            )
    tmp.optimize() # Do computations
    product_sym = tmp.quantizations[-1]

    tmp = pymaat.garch.quant.PriceVarianceGrid(
            ASYM_MODEL,
            starting_variance,
            price_size=shape[0],
            variance_size=shape[1],
            nper=nper,
            verbose=verbose
            )
    tmp.optimize() # Do computations
    product_asym = tmp.quantizations[-1]

    # Conditional
    tmp = pymaat.garch.quant.PriceVarianceConditional(
            SYM_MODEL,
            starting_variance,
            price_size=shape[0],
            variance_size=shape[1],
            nper=nper,
            verbose=verbose,
            parallel=parallel
            )
    tmp.optimize() # Do computations
    conditional_sym = tmp.quantizations[-1]

    tmp = pymaat.garch.quant.PriceVarianceConditional(
            ASYM_MODEL,
            starting_variance,
            price_size=shape[0],
            variance_size=shape[1],
            nper=nper,
            verbose=verbose,
            parallel=parallel
            )
    tmp.optimize() # Do computations
    conditional_asym = tmp.quantizations[-1]

    # Plot
    lim, plim = _get_lim(marginal_sym, marginal_asym,
            markov_sym, marginal_asym,
            product_sym, product_asym,
            conditional_sym, conditional_asym)

    root = '~/garch_quant/'
    root = '~/Dropbox/MyPapers/GarchQuant_WV/'
    path = root + 'showcase_marginal.eps'
    _plot_sym_vs_asym(path, marginal_sym, marginal_asym, lim, plim)

    path = root + 'showcase_markov.eps'
    _plot_sym_vs_asym(path, markov_sym, markov_asym, lim, plim)

    path = root + 'showcase_product.eps'
    _plot_sym_vs_asym(path, product_sym, product_asym, lim, plim)

    path = root + 'showcase_conditional.eps'
    _plot_sym_vs_asym(path, conditional_sym, conditional_asym, lim, plim)

def test_distortion_error(starting_variance, asym, model, nsim, nper):
    P = 128
    ntry = 1000
    np.random.seed(789123768)

    weight = np.array([1e1, 1e4])
    bounds = np.array([[0.,0.],[np.inf,np.inf]])

    xi = _marginal_simulation(
            starting_variance, model, nper, nsim)[-1]

    starting = _marginal_simulation(
            starting_variance, model, nper, P)[-1]

    tmp = qutil.VoronoiQuantization.stochastic_optimization(
                    starting,
                    xi,
                    nclvq=P,
                    weight=weight,
                    bounds=bounds,
                    )

    dist = np.empty((ntry,))
    for _ in range(ntry):
        print(_)
        simul = _marginal_simulation(
            starting_variance, model, nper, 10000000)
        tmp.estimate_and_set_distortion(simul[-1])
        dist[_] = np.log(tmp.distortion)
        print(dist[_])
    print(np.std(dist))


def test_marginal_error_split(
        starting_variance, asym, model, nsim, nper, simul):
    root = os.path.expanduser('~/garch_quant/marginal/')

    if asym:
        root += 'asym_'
    else:
        root += 'sym_'

    path = root + 'marginal_split_analysis.csv'

    P = 128
    ntry = 1000
    np.random.seed(789123768)

    with open(path, 'w') as csvfile:
        w = csv.writer(csvfile)
        header = ['no_split', 'no_split_lloyd', 'split','split_lloyd']
        w.writerow(header)
        for _ in range(ntry):
            s = time.time()
            weight = np.array([1e1, 1e4])
            bounds = np.array([[0.,0.],[np.inf,np.inf]])

            xi = _marginal_simulation(
                    starting_variance, model, nper, nsim)[-1]

            starting = _marginal_simulation(
                    starting_variance, model, nper, P)[-1]

            # No split
            tmp = qutil.VoronoiQuantization.stochastic_optimization(
                            starting,
                            xi,
                            weight=weight,
                            bounds=bounds,
                            )
            tmp.estimate_and_set_distortion(simul[-1])
            no_split_log_dist = np.log(tmp.distortion)

            # No split + LLoyd I
            tmp = qutil.VoronoiQuantization.stochastic_optimization(
                            tmp.quantizer.copy(),
                            xi,
                            nclvq=0,
                            nlloyd=10,
                            weight=weight,
                            bounds=bounds,
                            )
            tmp.estimate_and_set_distortion(simul[-1])
            no_split_lloyd_log_dist = np.log(tmp.distortion)

            # Re-start
            starting = _marginal_simulation(
                    starting_variance, model, nper, P)[-1]

            # Split
            tmp = qutil.VoronoiQuantization.stochastic_optimization(
                            starting,
                            xi,
                            nclvq=P,
                            weight=weight,
                            bounds=bounds,
                            )
            tmp.estimate_and_set_distortion(simul[-1])
            split_log_dist = np.log(tmp.distortion)

            # Split + LLoyd I
            tmp = qutil.VoronoiQuantization.stochastic_optimization(
                            tmp.quantizer.copy(),
                            xi,
                            nclvq=0,
                            nlloyd=10,
                            weight=weight,
                            bounds=bounds,
                            )
            tmp.estimate_and_set_distortion(simul[-1])
            split_lloyd_log_dist = np.log(tmp.distortion)
            # Print
            row = [
                    '{0:.8f}'.format(no_split_log_dist),
                    '{0:.8f}'.format(no_split_lloyd_log_dist),
                    '{0:.8f}'.format(split_log_dist),
                    '{0:.8f}'.format(split_lloyd_log_dist),
                    ]
            print(row)
            w.writerow(row)
            csvfile.flush()
            e = time.time()
            print(e-s)


def test_all(starting_variance, asym, model, nsim, nper, simul):
    def _get_row(s, e, all_quant):
        dist = 0.
        for (t,q) in enumerate(all_quant):
            q.estimate_and_set_distortion(simul[t])
            dist += q.distortion
        return [
            '{0:.8f}'.format(np.log(dist)),
            '{0:.2f}'.format(e-s)
            ]


    h0 = starting_variance
    root = os.path.expanduser('~/garch_quant/')
    if asym:
        root += 'asym_'
    else:
        root += 'sym_'
    path = root + 'main_result.csv'
    K = [8, 16, 32, 64]
    N = [4, 8, 16, 32]
    if asym:
        K, N = N, K
    with open(path, 'w') as csvfile:
        w = csv.writer(csvfile)
        header = [
                'K',
                'N',
                'marginal-dist',
                'marginal-time',
                'markov-dist',
                'markov-time',
                'product-dist',
                'product-time',
                'conditional-dist',
                'conditional-time',
                ]
        w.writerow(header)
        for (k,n) in zip(K,N):
            p = k*n
            row = [
                    '{0:d}'.format(k),
                    '{0:d}'.format(n),
                    ]
            # Marginal
            s = time.time()
            tmp = _do_marginal_quantization(h0, model, nper, nsim, p)
            e = time.time()
            row = row + _get_row(s,e, tmp)
            # Markov
            s = time.time()
            tmp = _do_markov_quantization(h0, model, nper, nsim, p)
            e = time.time()
            row = row + _get_row(s,e, tmp)
            # Product
            s = time.time()
            core = pymaat.garch.quant.PriceVarianceGrid(
                    model,
                    h0,
                    price_size=k,
                    variance_size=n,
                    nper=nper,
                    verbose=False,
                    )
            core.optimize() # Do computations
            e = time.time()
            row = row + _get_row(s,e, core.quantizations)
            # Conditional
            s = time.time()
            core = pymaat.garch.quant.PriceVarianceConditional(
                    model,
                    h0,
                    price_size=k,
                    variance_size=n,
                    nper=nper,
                    verbose=False,
                    parallel=False
                    )
            core.optimize() # Do computations
            e = time.time()
            row = row + _get_row(s,e, core.quantizations)
            print(row)
            w.writerow(row)
            csvfile.flush()

def test_marginal(
        starting_variance, asym, model, nsim, nper, size_P, simul):
    h0 = starting_variance
    root = os.path.expanduser('~/garch_quant/marginal/')
    if asym:
        root += 'asym_'
    else:
        root += 'sym_'
    path = root + 'marginal_result.csv'

    with open(path, 'w') as csvfile:
        w = csv.writer(csvfile)
        header = ['T', 'P', 'distortion']
        w.writerow(header)
        for T in range(1,nper+1):
            for P in size_P:
                s = time.time()
                starting = _marginal_simulation(h0, model, T, P)[-1]
                xi = _marginal_simulation(h0, model, T, nsim)[-1]
                tmp = qutil.VoronoiQuantization.stochastic_optimization(
                                starting,
                                xi,
                                nclvq=P,
                                nlloyd=10,
                                weight=np.array([1e1, 1e4]),
                                bounds=np.array([[0.,0.],[np.inf,np.inf]]),
                                )
                tmp.estimate_and_set_distortion(simul[T])
                row = [
                        '{0:d}'.format(T),
                        '{0:d}'.format(P),
                        '{0:.8f}'.format(np.log(tmp.distortion))
                        ]
                print(row)
                w.writerow(row)
                csvfile.flush()
                e = time.time()
                print(e-s)


def test_markov(
        starting_variance, asym, model, nsim, nper, size_P, simul):
    root = os.path.expanduser('~/garch_quant/markov/')
    if asym:
        root += 'asym_'
    else:
        root += 'sym_'
    path = root + 'markov_result.csv'

    Stub = namedtuple('InitPrev', ['quantizer', 'size', 'probability'])
    tmp = Stub(np.array([[1., starting_variance]]), 1, np.array([1.]))
    prev = {}
    for P in size_P:
        prev[P] = tmp

    with open(path, 'w') as csvfile:
        w = csv.writer(csvfile)
        header = ['T', 'P', 'distortion']
        w.writerow(header)
        for T in range(1,nper+1):
            for P in size_P:
                s = time.time()
                starting = _markov_simulation(model, prev[P], P)
                xi = _markov_simulation(model, prev[P], nsim)
                tmp = qutil.VoronoiQuantization.stochastic_optimization(
                                starting,
                                xi,
                                nclvq=P,
                                nlloyd=10,
                                weight=np.array([1e1, 1e4]),
                                bounds=np.array([[0.,0.],[np.inf,np.inf]]),
                                )
                tmp.estimate_and_set_probability(simul[T], strict=False)
                tmp.estimate_and_set_distortion(simul[T])
                row = [
                        '{0:d}'.format(T),
                        '{0:d}'.format(P),
                        '{0:.8f}'.format(np.log(tmp.distortion))
                        ]
                print(row)
                w.writerow(row)
                csvfile.flush()
                prev[P] = tmp  # Prep for next iter
                e = time.time()
                print(e-s)


def test_product(
        starting_variance, asym, model, nsim, nper, size_K, size_N, simul):
    root = os.path.expanduser('~/garch_quant/product/')
    if asym:
        root += 'asym_'
    else:
        root += 'sym_'
    path = root + 'product_result.csv'

    with open(path, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['T', 'K', 'N', 'distortion'])
        for (K,N) in zip(size_K, size_N):
            s = time.time()
            core = pymaat.garch.quant.PriceVarianceGrid(
                    model,
                    starting_variance,
                    price_size=K,
                    variance_size=N,
                    nper=nper,
                    verbose=False,
                    )
            core.optimize() # Do computations
            for (T,q) in enumerate(core.quantizations):
                if T>0:
                    q.estimate_and_set_distortion(simul[T])
                    row = [
                            '{0:d}'.format(T),
                            '{0:d}'.format(K),
                            '{0:d}'.format(N)
                            ]
                    row.append('{0:.8f}'.format(np.log(q.distortion)))
                    w.writerow(row)
                    csvfile.flush()
                    print(row)
            e = time.time()
            print(e-s)

def test_conditional(
        starting_variance, asym, model, nsim, nper, size_K, size_N, simul):
    root = os.path.expanduser('~/garch_quant/conditional/')
    if asym:
        root += 'asym_'
    else:
        root += 'sym_'
    path = root + 'conditional_result.csv'
    with open(path, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['T', 'K', 'N', 'distortion'])
        for (K,N) in zip(size_K, size_N):
            s = time.time()
            core = pymaat.garch.quant.PriceVarianceConditional(
                    model,
                    starting_variance,
                    price_size=K,
                    variance_size=N,
                    nper=nper,
                    verbose=True,
                    parallel=True
                    )
            core.optimize() # Do computations
            for (T,q) in enumerate(core.quantizations):
                if T>0:
                    row = [
                            '{0:d}'.format(T),
                            '{0:d}'.format(K),
                            '{0:d}'.format(N)
                            ]
                    q.estimate_and_set_distortion(simul[T])
                    row.append('{0:.8f}'.format(np.log(q.distortion)))
                    w.writerow(row)
                    csvfile.flush()
                    print(row)
            e = time.time()
            print(e-s)
