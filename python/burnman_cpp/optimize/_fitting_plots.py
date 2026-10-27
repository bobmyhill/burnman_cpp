# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Matplotlib fitting plots, copied from the pinned Python BurnMan implementation."""

# Original plotting functions: copyright (C) 2012-2015 by the BurnMan team,
# released under the GNU GPL v2 or later.

import copy

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    :param cov: The 2x2 covariance matrix to base the ellipse on.
    :type cov: numpy.array

    :param pos: The location of the center of the ellipse. Expects a 2-element
        sequence of [x0, y0].
    :type pos: list or numpy.array

    :param nstd: The radius of the ellipse in numbers of standard deviations.
        Defaults to 2 standard deviations.
    :type nstd: float

    :param ax: The axis that the ellipse will be plotted on. Defaults to the
        current axis.
    :type ax: matplotlib.pyplot.axes

    :param kwargs: Additional keyword arguments are passed on to the ellipse patch.

    :returns: The covariance ellipse (already applied to the desired axes object).
    :rtype: matplotlib.patches.Ellipse
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def corner_plot(popt, pcov, param_names=[], n_std=1.0):
    """
    Creates a corner plot of covariances

    :param popt: Optimized parameters.
    :type popt: numpy.array

    :param pcov: Covariance matrix of the parameters.
    :type pcov: 2D numpy.array

    :param param_names: Parameter names.
    :type param_names: list

    :param n_std: Number of standard deviations for ellipse.
    :type n_std: float

    :returns: ``matplotlib.pyplot.figure`` and list of ``matplotlib.pyplot.Axes``
        objects.
    :rtype: tuple
    """

    if len(pcov[0]) != len(pcov[:, 0]):
        raise Exception("Covariance matrices must be square")

    n_params = len(pcov[0])
    if n_params < 2:
        raise Exception(
            "Covariance matrix must be at least 2x2 for " "a corner plot to be plotted"
        )

    # ellipse plotting is prone to rounding errors, so we scale the plots here
    scaling = 1.0 / np.power(10.0, np.around(np.log10(np.abs(popt)) - 0.5))
    scaling = np.outer(scaling, scaling)

    fig, ax = plt.subplots(n_params - 1, n_params - 1)
    fig.set_size_inches(3.0 * (n_params - 1), 3.0 * (n_params - 1))

    for j in range(n_params - 1):
        for i in range(j):
            fig.delaxes(ax[i][j])

        for i in range(j, n_params - 1):
            ax[i][j].get_xaxis().get_major_formatter().set_useOffset(False)
            ax[i][j].get_yaxis().get_major_formatter().set_useOffset(False)
            ax[i][j].set_box_aspect(1)

            if j > 0:
                ax[i][j].get_yaxis().set_visible(False)
            if i < n_params - 2:
                ax[i][j].get_xaxis().set_visible(False)

            indices = np.array([j, i + 1])
            projected_cov = (pcov * scaling)[indices[:, None], indices]

            scaled_pos = np.array(
                [
                    popt[j] * np.sqrt(scaling[j][j]),
                    popt[i + 1] * np.sqrt(scaling[i + 1][i + 1]),
                ]
            )

            plot_cov_ellipse(
                cov=projected_cov, pos=scaled_pos, nstd=n_std, ax=ax[i][j], color="grey"
            )
            maxx = 1.5 * n_std * np.sqrt(projected_cov[0][0])
            maxy = 1.5 * n_std * np.sqrt(projected_cov[1][1])
            ax[i][j].set_xlim(scaled_pos[0] - maxx, scaled_pos[0] + maxx)
            ax[i][j].set_ylim(scaled_pos[1] - maxy, scaled_pos[1] + maxy)

    if param_names != []:
        for i in range(n_params - 1):
            ax[n_params - 2][i].set_xlabel(
                "{0:s} (x $10^{{{1:d}}}$)".format(
                    param_names[i], -int(np.log10(np.sqrt(scaling[i][i])))
                )
            )

        for j in range(1, n_params):
            ax[j - 1][0].set_ylabel(
                "{0:s} (x $10^{{{1:d}}}$)".format(
                    param_names[j], -int(np.log10(np.sqrt(scaling[j][j])))
                )
            )

    fig.set_layout_engine("tight")

    return fig, ax


def weighted_residual_plot(
    ax,
    model,
    flag=None,
    sd_limit=3,
    cmap=plt.cm.RdYlBu,
    plot_axes=[0, 1],
    scale_axes=[1.0, 1.0],
):
    """
    Creates a plot of the weighted residuals
    The user can choose the projection axes, and scaling to apply to those axes
    The chosen color palette (cmap) is discretised by standard deviation up
    to a cut off value of sd_limit.

    :param ax: Plot.
    :param type: ``matplotlib.pyplot.Axes``

    :param model: A model as used by
        :func:`burnman.nonlinear_least_squares_fit`.
        Must contain the attributes model.data,
        model.weighted_residuals and
        model.flags (if flag is not None).
    :type model: object

    :param flag: String to determine which data to plot.
        Finds matches with model.flags.
    :type flag: str

    :param sd_limit: Data with weighted residuals exceeding this
        limit are plotted in black.
    :type sd_limit: float

    :param cmap: Color palette.
    :type cmap: matplotlib color palette

    :param plot_axes: Data axes to use as plot axes.
    :type plot_axes: list of int

    :param scale_axes: Plot axes are scaled by multiplication
        of the data by these values.
    :type scale_axes: list of float

    :returns: Coloured scatter plot of the weighted residuals in data space.
    :rtype: matplotlib Axes object
    """
    if flag is None:
        mask = range(len(model.data[:, 0]))
    else:
        mask = [i for i, flg in enumerate(model.flags) if flg == flag]

    cmap_cp = copy.copy(cmap)
    cmap_cp.set_under("k")
    cmap_cp.set_over("k")
    bounds = np.linspace(-sd_limit, sd_limit, sd_limit * 2 + 1)
    norm = colors.BoundaryNorm(bounds, cmap_cp.N)

    im = ax.scatter(
        model.data[:, plot_axes[0]][mask] * scale_axes[0],
        model.data[:, plot_axes[1]][mask] * scale_axes[1],
        c=model.weighted_residuals[mask],
        cmap=cmap_cp,
        norm=norm,
        s=50,
    )
    plt.colorbar(im, ax=ax, label="Misfit (standard deviations)")


def plot_residuals(ax, weighted_residuals, n_bins=None, flags=[]):
    if flags is []:
        flags = [""] * len(weighted_residuals)
        list_flags = [""]
    else:
        list_flags = list(set(flags))

    if n_bins is None:
        try:  # Only works for recent versions of numpy
            bin_heights, bin_bounds = np.histogram(
                weighted_residuals, bins="auto", density=True
            )
            n_bins = len(bin_heights)
        except:
            n_bins = 11.0

    mask = [i for i, f in enumerate(flags)]
    for flag in list_flags:
        binwidth = np.ptp(weighted_residuals) / n_bins
        dmin = min(weighted_residuals) - binwidth
        dmax = max(weighted_residuals) + binwidth
        bins = np.linspace(dmin, dmax, n_bins)
        bin_heights, bin_bounds = np.histogram(
            weighted_residuals[mask], bins=bins, density=True
        )

        normalisation = float(len(weighted_residuals[mask])) / float(
            len(weighted_residuals)
        )
        bin_centers = (bin_bounds[:-1] + bin_bounds[1:]) / 2.0
        bin_heights = bin_heights * normalisation
        bin_widths = bin_bounds[1] - bin_bounds[0]
        plt.bar(bin_centers, bin_heights, width=bin_widths, label=flag, alpha=0.2)
        mask = [i for i, f in enumerate(flags) if f != flag and i in mask]

        x = np.linspace(bin_bounds[0], bin_bounds[-1], 1001)
        ax.plot(x, norm.pdf(x) * normalisation)

    ax.set_title("Residual plot versus expected normal distribution")
    ax.set_xlabel("Number of standard deviations from the mean")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right")
