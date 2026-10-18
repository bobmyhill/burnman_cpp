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

Temporary samples outside the domain come from weighted quadratic fits to
nearby potential values. This preserves quadratic potentials at edges and
corners without extending a sharp local Hessian over the Gaussian support.
Widths are in grid cells.
"""

from copy import copy

import numpy as np
from scipy.ndimage import gaussian_filter


def _extend_axis(values, axis, sigma, radius):
    """Fit boundary polynomials anchored to the observed edge values.

    Each fit uses the available Gaussian support and at least three samples.
    Scaled index coordinates keep the two-coefficient fit independent of
    physical units. Applying this helper along both axes also fills corners.
    """
    data = np.moveaxis(values, axis, -1)
    count = min(data.shape[-1], max(3, radius + 1))
    indices = np.arange(count, dtype=float)
    u = indices / (count - 1)
    weights = np.exp(-0.25 * (indices / sigma) ** 2)
    design = np.column_stack((u, u * u)) * weights[:, None]
    outside = -np.arange(1, radius + 1, dtype=float) / (count - 1)

    def predict(edge):
        target = (edge[..., :count] - edge[..., :1]) * weights
        coefficients = np.linalg.lstsq(design, target.reshape(-1, count).T, rcond=None)[
            0
        ].T
        coefficients = coefficients.reshape(*data.shape[:-1], 2)
        return (
            edge[..., :1]
            + coefficients[..., :1] * outside
            + coefficients[..., 1:] * outside**2
        )

    left, right = predict(data), predict(data[..., ::-1])
    extended = np.concatenate((left[..., ::-1], data, right), axis=-1)
    return np.moveaxis(extended, -1, axis)


def _smooth_values(grids, sigma, *, truncate=4.0):
    """Extend the potential, convolve in 2D and crop to the original domain."""
    f, fx, fy, fxx, fxy, fyy = (
        np.asarray(getattr(grids, name), dtype=np.float64)
        for name in ("values", "d_dx", "d_dy", "d2_dx2", "d2_dxdy", "d2_dy2")
    )
    if f.ndim != 2 or any(a.shape != f.shape for a in (fx, fy, fxx, fxy, fyy)):
        raise ValueError("All fields must be 2D arrays with the same shape")
    if min(f.shape) < 3:
        raise ValueError(
            "Smoothing derivatives requires at least three points per axis."
        )
    if not all(np.isfinite(a).all() for a in (f, fx, fy, fxx, fxy, fyy)):
        raise ValueError(
            "Infill potential values and all derivative gaps before smoothing."
        )
    grids.spacings()
    widths = np.asarray(sigma, dtype=float)
    if widths.ndim == 0:
        widths = np.repeat(widths, 2)
    if widths.shape != (2,) or not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("sigma must be a positive finite scalar or (y, x) pair")
    if not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("truncate must be finite and positive")
    padded, crop = f, [slice(None), slice(None)]
    for axis in (1, 0):
        # Match scipy.ndimage's discrete kernel radius, including radius zero.
        radius = int(truncate * widths[axis] + 0.5)
        if radius:
            padded = _extend_axis(padded, axis, widths[axis], radius)
            crop[axis] = slice(radius, -radius)
    return gaussian_filter(padded, widths, mode="nearest", truncate=truncate)[
        tuple(crop)
    ]


def smooth_potential_grid(grids, sigma=2.0, *, truncate=4.0):
    """Return smoothed potential values and all five natural-coordinate derivatives.

    Apply a 2D Gaussian (default sigma=2 cells along both axes) to the potential,
    extending its edges with weighted, anchored quadratic fits to nearby values.
    The input gradients and Hessians do not determine the extension. All six
    input arrays must be finite; infill gaps first. At least three points per
    axis are required. sigma accepts a scalar or (y, x) pair in grid cells.
    Analytically differentiate a tensor-product cubic interpolant of the
    smoothed values, including at the domain edges. Metadata and original
    arrays are retained unchanged. No equilibrium solves or np.gradient calls
    are used. The extension reproduces quadratics but does not guarantee
    thermodynamic stability for every surface; validate the derived properties.
    """
    from scipy.interpolate import CubicSpline

    values = _smooth_values(grids, sigma, truncate=truncate)
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
