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
import pymaat.bsiv as bsiv

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar
import time

from pymaat.perf import timeblock

@pytest.fixture
def starting_variance():
    return 0.14**2./252.

@pytest.fixture
def nper():
    return 21

@pytest.fixture
def model():
    return HestonNandiGarch(
                mu=2.04,
                omega=3.28e-7,
                alpha=4.5e-6,
                beta=0.962,
                gamma=0)
    # return HestonNandiGarch(
    #             mu=2.04,
    #             omega=3.28e-7,
    #             alpha=4.5e-6,
    #             beta=0.8,
    #             gamma=190)

@pytest.fixture
def rnmodel():
    return HestonNandiGarch(
                mu=0.,
                omega=4.14e-7,
                alpha=7.16e-6,
                beta=0.8,
                gamma=152)

class TestOptionPricing:

    @pytest.mark.parametrize('put', [True, False])
    def test(self, rnmodel, starting_variance, nper, put):
        normstrike = np.linspace(0.95, 1.05, 11)
        with timeblock('pricing'):
            p0 = rnmodel.option_price(
                    normstrike, variance=starting_variance, T=nper, put=put)
        with timeblock('brent'):
            iv0 = bsiv.brent_dekker(normstrike, p0, put)
        # with timeblock('sor'):
        #     iv1 = bsiv.sor(normstrike, p0, put)
        print(iv0*np.sqrt(252./21)*100.)


class TestQuantization:

    verbose = False

    @pytest.fixture
    def shape(self):
        return (10,10)

    @pytest.fixture
    def nsim(self):
        return 100000

    def _plot_at(self, core, t, path):
        root = '~/garch_quant/'
        fig = plt.figure(figsize=(8,7), dpi=1000)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.03, 0.8])
        ax = fig.add_axes([0.1, 0.1, 0.75, 0.8])
        core.plot_at(t, ax, cbar_ax, quantizer_size=5)
        ax.set_xlabel(r'Price (%)')
        ax.set_ylabel(r'Annual Vol. (%)')
        # Set color bar
        cbar_ax.set_title('p (%)')
        plt.savefig(
                os.path.expanduser(root+path+'.pdf'),
                format='pdf'
                )
        plt.close(fig)

    @pytest.mark.parametrize('type_',
            ['marginal', 'markov', 'product', 'conditional'])
    def test(self, type_, model, starting_variance, nper, shape, nsim):
        kwargs = {}
        if type_ == 'marginal':
            quantization = pymaat.garch.quant.Marginal
            shape = np.prod(shape)
        elif type_ == 'markov':
            quantization = pymaat.garch.quant.Markov
            shape = np.prod(shape)
        elif type_ == 'product':
            quantization = pymaat.garch.quant.Product
        elif type_ == 'conditional':
            quantization = pymaat.garch.quant.Conditional
        s = time.time()
        shape = 5
        tmp = quantization(
                model=model,
                shape=shape,
                first_variance=starting_variance,
                nper=nper,
                verbose=self.verbose,
                **kwargs
                )
        e = time.time()
        print("Time: {}".format(e-s))
        print(np.sqrt(np.sum(tmp.estimate_distortion(nsim)**2.)))
        # Plotting
        self._plot_at(tmp, -1, type_ + '/last')
        # # Option pricing
        # k = np.linspace(0.95, 1.05, 100)
        # with timeblock('analytical'):
        #     p0 = rnmodel.option_price(k, variance=starting_variance, T=nper)
        # with timeblock('quantized'):
        #     p1 = tmp.option_price(k, T=nper)
        # print(np.column_stack((p0, p1)))
        # iv0 = bsiv.brent_dekker(k, p0)
        # iv1 = bsiv.brent_dekker(k, p1)
        # print(np.column_stack((iv0, iv1))*np.sqrt(252./nper)*100.)
