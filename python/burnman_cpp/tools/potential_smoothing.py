# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------
"""Two-dimensional Gaussian smoothing of a natural thermodynamic potential.

The boundary correction restores the discrete kernel's missing moments from
local gradients and Hessians, including the mixed derivative. It reproduces
quadratic potentials at edges and corners. Widths are in grid cells.
"""

from copy import copy

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter


def _smooth_values(grids, sigma, *, truncate=4.0):
    """Convolve potential values with second-order moment correction."""
    f, fx, fy, fxx, fxy, fyy = (
        np.asarray(getattr(grids, name), dtype=np.float64)
        for name in ("values", "d_dx", "d_dy", "d2_dx2", "d2_dxdy", "d2_dy2")
    )
    if f.ndim != 2 or any(a.shape != f.shape for a in (fx, fy, fxx, fxy, fyy)):
        raise ValueError("All fields must be 2D arrays with the same shape")
    if not all(np.isfinite(a).all() for a in (f, fx, fy, fxx, fxy, fyy)):
        raise ValueError(
            "Infill potential values and all derivative gaps before smoothing."
        )
    dx, dy = grids.spacings()
    widths = np.asarray(sigma, dtype=float)
    if widths.ndim == 0:
        widths = np.repeat(widths, 2)
    if widths.shape != (2,) or not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("sigma must be a positive finite scalar or (y, x) pair")
    if not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("truncate must be finite and positive")
    sigma_y, sigma_x = widths

    def moments_1d(n, sigma_pixels, spacing):
        # Match scipy.ndimage's radius selection and normalized kernel.
        radius = int(truncate * sigma_pixels + 0.5)
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / sigma_pixels) ** 2)
        weights /= weights.sum()
        displacements = offsets * spacing
        ones = np.ones(n, dtype=np.float64)

        # correlate1d samples at i + offset.  Constant padding means that
        # samples outside the known grid do not contribute to each moment.
        def partial_moment(power):
            return correlate1d(
                ones,
                weights * displacements**power,
                mode="constant",
                cval=0.0,
            )

        m0 = partial_moment(0)
        m1 = partial_moment(1)
        m2 = partial_moment(2)
        full_variance = np.dot(weights, displacements**2)
        return m0, m1, m2, full_variance

    ny, nx = f.shape
    ax0, ax1, ax2, vx = moments_1d(nx, sigma_x, dx)
    ay0, ay1, ay2, vy = moments_1d(ny, sigma_y, dy)

    # Contribution from known grid points only.
    result = gaussian_filter(
        f,
        sigma=(sigma_y, sigma_x),
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )

    # Restore zeroth, first and second moments missing outside the rectangle.
    result += (1.0 - ay0[:, None] * ax0[None, :]) * f
    result -= ay0[:, None] * ax1[None, :] * fx
    result -= ay1[:, None] * ax0[None, :] * fy
    result += 0.5 * (vx - ay0[:, None] * ax2[None, :]) * fxx
    result += 0.5 * (vy - ay2[:, None] * ax0[None, :]) * fyy
    result -= ay1[:, None] * ax1[None, :] * fxy
    return result


def smooth_potential_grid(grids, sigma=2.0, *, truncate=4.0):
    """Return smoothed potential values and all five natural-coordinate derivatives.

    Apply a 2D Gaussian (default sigma=2 cells along both axes), correcting
    boundary moments with the supplied derivatives. All six arrays must be
    finite; infill gaps first. At least three points per axis are required.
    Analytically differentiate a tensor-product cubic interpolant of the
    smoothed values, including at the domain edges. Metadata and original
    arrays are retained unchanged. No equilibrium solves or np.gradient calls
    are used. The correction is not an integral-conserving boundary condition.
    """
    from scipy.interpolate import CubicSpline

    values = _smooth_values(grids, sigma, truncate=truncate)
    if min(values.shape) < 3:
        raise ValueError(
            "Smoothing derivatives requires at least three points per axis."
        )
    result = copy(grids)
    result.values = values
    for coordinates, axis, first, second in (
        (grids.x, 1, "d_dx", "d2_dx2"),
        (grids.y, 0, "d_dy", "d2_dy2"),
    ):
        spline = CubicSpline(coordinates, values, axis=axis)
        setattr(result, first, spline(coordinates, 1))
        setattr(result, second, spline(coordinates, 2))
    result.d2_dxdy = CubicSpline(grids.y, result.d_dx, axis=0)(grids.y, 1)
    return result
