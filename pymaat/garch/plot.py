import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.colorbar

from pymaat.mathutil import round_to_int
import pymaat.plotutil

def quant_values_3_by_3(core):
    fig = plt.figure()

    # Compute bounds
    all_quant = core.all_quant[1:]  # Disregard time 0
    all_vol = np.concatenate(
        [core.variance_formatter(q.variance).ravel() for q in all_quant])
    all_price = np.concatenate(
        [core.price_formatter(q.price) for q in all_quant])
    all_probas = np.concatenate(
        [100.*q.probability.ravel() for q in all_quant])
    m = 0
    vol_bounds = pymaat.plotutil.get_lim(
        all_vol, lb=0., margin=m)
    price_bounds = pymaat.plotutil.get_lim(
        all_price, lb=0., margin=m)
    proba_bounds = pymaat.plotutil.get_lim(
        all_probas, lb=0., ub=100., margin=m)

    # Initialize color map
    cmap = matplotlib.cm.get_cmap('gnuplot2')
    cmap = cmap.reversed()
    norm = matplotlib.colors.Normalize(vmin=proba_bounds[0],
                                       vmax=proba_bounds[1])

    step = math.floor(len(all_quant)/9)
    selected = list(range(0, len(all_quant), step))
    fd = {'fontsize': 8}

    for t in range(9):
        ax = fig.add_subplot(3, 3, t+1)
        quant = all_quant[selected[t]]

        # Tiles
        vprice = core.price_formatter(quant.voronoi_price)
        vprice[0] = price_bounds[0]
        vprice[-1] = price_bounds[1]
        for idx in range(quant.shape[0]):
            x_tiles = vprice[idx:idx+2]
            y_tiles = core.variance_formatter(
                quant.voronoi_variance[idx])
            y_tiles[0] = vol_bounds[0]
            y_tiles[-1] = vol_bounds[1]
            x_tiles, y_tiles = np.meshgrid(x_tiles, y_tiles)
            ax.pcolor(
                x_tiles, y_tiles,
                100.*quant.probability[idx][:, np.newaxis],
                cmap=cmap, norm=norm)

        # Quantizers

        bprice = np.broadcast_to(quant.price[:, np.newaxis], quant.shape)
        x_pts = core.price_formatter(bprice)
        y_pts = core.variance_formatter(quant.variance)
        ax.scatter(x_pts.ravel(),
                   y_pts.ravel(),
                   c=cmap(norm(100.*quant.probability.ravel())),
                   s=2, marker=".")

        # Title
        ax.set_title('t={}'.format(selected[t]+1), fontdict=fd, y=0.95)

        # Y-Axis
        if t in [0, 3, 6]:
            lb = round_to_int(vol_bounds[0], base=5, fcn=math.floor)
            ub = round_to_int(vol_bounds[1], base=5, fcn=math.ceil)
            tickspace = round_to_int((ub-lb)/5, base=5, fcn=math.ceil)
            ticks = np.arange(lb, ub, tickspace)
            ticks = np.unique(np.append(ticks, ub))
            labels = ['{0:d}'.format(yt) for yt in ticks]
            ax.set_ylabel(r'Annual Vol. (%)', fontdict=fd)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontdict=fd)
        else:
            ax.yaxis.set_ticks([])

        # X-Axis
        if t in [6, 7, 8]:
            lb = round_to_int(price_bounds[0], base=5, fcn=math.floor)
            ub = round_to_int(price_bounds[1], base=5, fcn=math.ceil)
            tickspace = round_to_int((ub-lb)/5, base=5, fcn=math.ceil)
            ticks = np.arange(lb, ub, tickspace)
            ticks = np.unique(np.append(ticks, ub))
            labels = ['{0:d}'.format(xt) for xt in ticks]
            ax.set_xlabel(r'Price (%)', fontdict=fd)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontdict=fd)
        else:
            ax.xaxis.set_ticks([])

        ax.set_ylim(vol_bounds)
        ax.set_xlim(price_bounds)

    # Add colorbar
    cbar_ax = fig.add_axes([0.925, 0.1, 0.025, 0.79])
    matplotlib.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm,
        orientation='vertical')
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), fontdict=fd)
    # cbar_ax.set_xlabel(r'%', fontdict=fd)
    cbar_ax.set_title('Proba. (%)', fontdict=fd)

#########################
# Variance Quantization #
#########################


def varquant_distortion(core):
    all_quant = core.all_quant[1:]
    y = [q.distortion for q in all_quant]
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.plot(np.arange(len(y))+1, np.array(y))


def varquant_values(core):
    fig = plt.figure()
    main_ax = fig.add_axes([0.1, 0.1, 0.7, 0.85])

    # Compute bounds
    all_quant = core.all_quant[1:]
    all_vol = np.concatenate(
        [core.print_formatter(q.value) for q in all_quant])
    all_probas = np.concatenate(
        [100.*q.probability for q in all_quant])
    vol_bounds = pymaat.plotutil.get_lim(
        all_vol, lb=0.)
    proba_bounds = pymaat.plotutil.get_lim(
        all_probas, lb=0., ub=100.)

    # Initialize color map
    cmap = matplotlib.cm.get_cmap('gnuplot2')
    cmap = cmap.reversed()
    norm = matplotlib.colors.Normalize(vmin=proba_bounds[0],
                                       vmax=proba_bounds[1])

    # Main plot
    for (t, quant) in enumerate(all_quant):
        # Plot tiles
        x_tiles = np.array([t+0.5, t+1.5])
        y_tiles = core.print_formatter(quant.voronoi)
        y_tiles[-1] = vol_bounds[1]
        x_tiles, y_tiles = np.meshgrid(x_tiles, y_tiles)
        main_ax.pcolor(
            x_tiles, y_tiles,
            100.*quant.probability[:, np.newaxis],
            cmap=cmap, norm=norm)

        #  Plot quantizer
        y_pts = core.print_formatter(quant.value)
        x_pts = np.broadcast_to(t+1, y_pts.shape)
        main_ax.scatter(x_pts, y_pts, c='k', s=2, marker=".")

    # Y-Axis
    main_ax.set_ylabel(r'Annualized Volatility (%)')
    main_ax.set_ylim(vol_bounds)
    # X-Axis
    main_ax.set_xlim(0.5, t+1.5)
    main_ax.set_xlabel(r'Trading Days ($t$)')
    tickspace = round_to_int(0.1*t, base=5, fcn=math.ceil)
    ticks = np.arange(tickspace, t+1, tickspace)
    main_ax.xaxis.set_ticks(ticks)
    main_ax.xaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%d'))

    # Add colorbar
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.85])
    matplotlib.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cbar_ax.set_ylabel(r'%')


def varquant_transition_at(core, t=1):
    if t <= 0:
        raise ValueError
    fig = plt.figure()
    main_ax = fig.add_axes([0.1, 0.1, 0.7, 0.85])

    prev = core.all_quant[t-1]
    current = core.all_quant[t]

    x_pts = core.print_formatter(prev.value)[:, np.newaxis]
    y_pts = core.print_formatter(current.value)[np.newaxis, :]
    x_bounds = pymaat.plotutil.get_lim(x_pts, lb=0)
    y_bounds = pymaat.plotutil.get_lim(y_pts, lb=0)

    x_tiles = core.print_formatter(prev.voronoi)
    x_tiles[-1] = x_bounds[1]

    y_tiles = core.print_formatter(current.voronoi)
    y_tiles[-1] = y_bounds[1]

    proba = current.transition_probability*100.
    proba_bounds = pymaat.plotutil.get_lim(proba, lb=0, ub=100)

    # Initialize color map
    cmap = matplotlib.cm.get_cmap('gnuplot2')
    cmap = cmap.reversed()
    norm = matplotlib.colors.Normalize(vmin=proba_bounds[0],
                                       vmax=proba_bounds[1])
    # Color tiles
    main_ax.pcolor(
        x_tiles, y_tiles, proba, cmap=cmap, norm=norm)

    # Plot quantizers
    x_pts, y_pts = np.broadcast_arrays(x_pts, y_pts)
    main_ax.scatter(x_pts, y_pts, c='k', s=2, marker=".")

    # X-Axis
    main_ax.set_xlabel('Annualized Volatility (%) on t={}'.format(t-1))
    main_ax.set_xlim(x_bounds)

    # Y-Axis
    main_ax.set_ylabel('Annualized Volatility (%) on t={}'.format(t))
    main_ax.set_ylim(y_bounds)

    # Add colorbar
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.85])
    matplotlib.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cbar_ax.set_ylabel(r'%')
