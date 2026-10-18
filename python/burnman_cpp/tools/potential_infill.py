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
"""Infill thermodynamic potentials with coordinate-wise stability constraints.

Preserve observed values and derivatives, including at phase boundaries. A
sparse constrained fit reconstructs gaps; local polynomial fits reconstruct
missing derivatives. Edges use asymmetric support. The inequalities constrain
coordinate slopes and curvatures; they do not certify joint convexity or the
discovery of phase transitions absent from the original data.
"""

from __future__ import annotations

import warnings
from copy import copy
import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize
from scipy.spatial import cKDTree

from .potential_grid import DERIVATIVES as _DERIVATIVES


class _Rows:
    """Sparse linear stencils after eliminating the fixed, observed values."""

    def __init__(self, known, unknown_ids):
        self.known = known
        self.unknown_ids = unknown_ids
        self.n = int((unknown_ids >= 0).sum())
        self.row, self.col, self.coeff, self.rhs = [], [], [], []

    def add(self, terms, rhs=0.0, weight=1.0):
        """Append sqrt(weight) * (sum(terms) - rhs), only if unknowns occur."""
        unknowns = [
            (int(self.unknown_ids[r, c]), a)
            for r, c, a in terms
            if self.unknown_ids[r, c] >= 0 and a
        ]
        if not unknowns or weight == 0:
            return False
        known_part = sum(
            a * self.known[r, c]
            for r, c, a in terms
            if self.unknown_ids[r, c] < 0 and a
        )
        ir = len(self.rhs)
        scale = np.sqrt(weight)
        for j, a in unknowns:
            self.row.append(ir)
            self.col.append(j)
            self.coeff.append(scale * a)
        self.rhs.append(scale * (rhs - known_part))
        return True

    def matrix(self):
        A = sparse.coo_matrix(
            (self.coeff, (self.row, self.col)), shape=(len(self.rhs), self.n)
        ).tocsr()
        return A, np.asarray(self.rhs, dtype=float)


def _taylor_prediction(h, derivs, missing, dx, dy, k):
    """Local derivative-aware predictions, computed in original physical units."""
    sy, px = np.indices(h.shape)
    known = ~missing
    sources = np.column_stack((px[known].astype(float), sy[known].astype(float)))
    destinations = np.column_stack(
        (px[missing].astype(float), sy[missing].astype(float))
    )
    # Search in index coordinates to avoid unstable kNN anisotropy if the coordinates
    # are expressed in wildly different physical units.
    _, indices = cKDTree(sources).query(destinations, k=min(k, len(sources)))
    indices = np.asarray(indices).reshape(len(destinations), -1)
    delta = destinations[:, None, :] - sources[indices]
    px_delta = delta[:, :, 0] * dx
    sy_delta = delta[:, :, 1] * dy
    dsq = (delta**2).sum(axis=2)
    w = (dsq + 0.25) ** -1.5
    preds = h[known][indices].copy()
    for dfield, factor in zip(
        derivs,
        (px_delta, sy_delta, 0.5 * px_delta**2, px_delta * sy_delta, 0.5 * sy_delta**2),
    ):
        if dfield is not None:
            values = dfield[known][indices]
            preds += np.where(np.isfinite(values), values, 0) * factor
    return (preds * w).sum(axis=1) / w.sum(axis=1)


def _fill_derivative_grids(
    grids, h, derivatives, spacings, weights, k, gradient_bounds, info
):
    """Fit missing derivatives jointly, keeping the input observations fixed.

    At each target, the potential is fixed to its reconstructed value. The five
    remaining Taylor coefficients describe one quadratic in index coordinates.
    Existing centre derivatives fix their coefficients exactly. Neighbouring
    values and original derivatives are Gaussian-weighted observations. Infilled
    values receive less weight with distance from original observations.
    Bounds apply only to estimated coefficients. Fits never consume estimates
    from earlier targets. Positive field numbers restrict support to that field;
    an unlabelled target has no known phase-boundary restriction.
    """
    result = copy(grids)
    result.values = h
    scales = np.array(
        [
            spacings[0],
            spacings[1],
            spacings[0] ** 2,
            spacings[0] * spacings[1],
            spacings[1] ** 2,
        ]
    )
    observed = [np.full(h.shape, np.nan) if d is None else d for d in derivatives]
    output = [d.copy() for d in observed]
    missing = np.stack([np.isnan(d) for d in observed])
    fields = getattr(grids, "field_numbers", None)
    if fields is not None:
        fields = np.asarray(fields)
        if fields.shape != h.shape:
            raise ValueError("field_numbers must match values.shape")
    distance = distance_transform_edt(np.isnan(grids.values))
    lower = np.array(
        [
            gradient_bounds[0, 0] * scales[0],
            gradient_bounds[1, 0] * scales[1],
            -np.inf,
            -np.inf,
            -np.inf,
        ]
    )
    upper = np.array(
        [
            gradient_bounds[0, 1] * scales[0],
            gradient_bounds[1, 1] * scales[1],
            np.inf,
            np.inf,
            np.inf,
        ]
    )
    for index, sign in zip((2, 4), grids.curvature):
        if sign > 0:
            lower[index] = 0.0
        else:
            upper[index] = 0.0
    failures = []
    max_radius = max(h.shape)
    for row, col in np.argwhere(missing.any(axis=0)):
        coefficients = np.array([d[row, col] for d in observed]) * scales
        free = np.isnan(coefficients)
        n_free = int(free.sum())
        base_radius = min(max(2, int(np.ceil(np.sqrt(k)))), max_radius)
        restrict_field = fields is not None and fields[row, col] > 0
        radius = base_radius
        # Grow the stencil (and, as a last resort, drop the same-field
        # restriction) until enough independent local observations are
        # found to determine every free coefficient, instead of giving up
        # at the first undersized/rank-deficient neighbourhood.
        while True:
            rows, cols = np.mgrid[
                max(0, row - radius) : min(h.shape[0], row + radius + 1),
                max(0, col - radius) : min(h.shape[1], col + radius + 1),
            ]
            take = np.ones(rows.shape, dtype=bool)
            if restrict_field:
                take &= fields[rows, cols] == fields[row, col]
            rr, cc = rows[take], cols[take]
            x, y = (cc - col).astype(float), (rr - row).astype(float)
            one, zero = np.ones(len(x)), np.zeros(len(x))
            designs = [
                np.column_stack((x, y, 0.5 * x * x, x * y, 0.5 * y * y)),
                np.column_stack((one, zero, x, y, zero)),
                np.column_stack((zero, one, zero, x, y)),
                *(np.broadcast_to(b, (len(x), 5)) for b in np.eye(5)[2:]),
            ]
            values = [h[rr, cc] - h[row, col]] + [
                d[rr, cc] * s for d, s in zip(observed, scales)
            ]
            kernel = np.exp(-0.5 * (x * x + y * y) / radius**2)
            fit_weights = [weights[0] * kernel / (1 + distance[rr, cc] ** 2)] + [
                w * kernel for w in (weights[1], weights[1], *([weights[2]] * 3))
            ]
            matrices, targets = [], []
            for design, value, weight in zip(designs, values, fit_weights):
                valid = np.isfinite(value) & (weight > 0)
                root = np.sqrt(weight[valid])
                matrices.append(design[valid] * root[:, None])
                targets.append(value[valid] * root)
            matrix, target = np.vstack(matrices), np.concatenate(targets)
            target -= matrix[:, ~free] @ coefficients[~free]
            matrix = matrix[:, free]
            estimate, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
            if rank >= n_free:
                break
            if radius < max_radius:
                radius = min(radius * 2, max_radius)
            elif restrict_field:
                restrict_field, radius = False, base_radius
            else:
                break
        if rank < n_free:
            failures.append(
                (int(row), int(col), "Insufficient independent local observations")
            )
            continue
        if ((estimate < lower[free]) | (estimate > upper[free])).any():
            fit = lsq_linear(
                matrix, target, bounds=(lower[free], upper[free]), method="bvls"
            )
            if not fit.success:
                failures.append((int(row), int(col), str(fit.message)))
                continue
            estimate = fit.x
        coefficients[free] = estimate
        for i in np.flatnonzero(free):
            output[i][row, col] = coefficients[i] / scales[i]
    estimated_masks = {"values": np.isnan(grids.values) & np.isfinite(h)}
    for name, original, filled in zip(_DERIVATIVES, observed, output):
        setattr(result, name, filled)
        estimated_masks[name] = np.isnan(original) & np.isfinite(filled)
    info.update(
        estimated_masks=estimated_masks,
        derivative_fit_failures=failures,
        n_estimated={name: int(mask.sum()) for name, mask in estimated_masks.items()},
        success=bool(info["success"] and not failures),
    )
    return result


def infill_potential_grid(
    grids,
    *,
    k=12,
    taylor_weight=1.0,
    derivative_weight=4.0,
    curvature_weight=1.0,
    bending_weight=0.2,
    huber_delta=None,
    bounds="thermodynamic",
    gradient_bounds=None,
    maxiter=500,
    tol=1e-8,
    fill_derivatives=False,
    return_info=False,
):
    """Return a new PotentialGrid with missing potential values reconstructed.

    E(V,S), F(V,T), G(P,T) and H(P,S) use their natural coordinate curvature
    signs. Gradient bounds enforce positive V/T and nonnegative P; entropy's
    reference is unrestricted. gradient_bounds optionally supplies ((xmin,
    xmax), (ymin, ymax)) bounds on derivatives in physical units.

    fill_derivatives=True also reconstructs derivative NaNs from constrained
    local quadratic fits, preserving every original observation and metadata.
    Positive field numbers restrict support where sufficient observations are
    available. Underdetermined fits remain NaN and are reported.

    k controls the number of Taylor neighbours and initial derivative-fit
    radius. taylor_weight, derivative_weight, curvature_weight and bending_weight
    control the fit and robust thin-plate penalty. bounds='rim' additionally
    limits missing components to the extrema of their finite neighbours;
    the default imposes physical inequalities without clipping amplitudes.
    maxiter/tol control the sparse constrained optimisation.

    return_info=True returns (grid, diagnostics). Already observed violations
    cannot be repaired without changing those observations and are reported.
    The inequalities apply along each coordinate, not to the full Hessian.
    """
    h = np.asarray(grids.values, dtype=float)
    if h.ndim != 2 or np.isinf(h).any() or not np.isfinite(h).any():
        raise ValueError(
            "Potential values must be 2D with at least one finite value and no infinity"
        )
    spacings = grids.spacings()
    dx, dy = spacings
    if k < 1 or tol <= 0:
        raise ValueError("k and tol must be positive")
    if bounds not in ("thermodynamic", "rim"):
        raise ValueError("bounds must be 'thermodynamic' or 'rim'")
    if (
        min(taylor_weight, derivative_weight, curvature_weight, bending_weight) < 0
        or taylor_weight == 0
    ):
        raise ValueError("Weights must be nonnegative, and taylor_weight > 0")
    gradient_bounds = np.asarray(
        grids.gradient_bounds if gradient_bounds is None else gradient_bounds, float
    )
    if (
        gradient_bounds.shape != (2, 2)
        or np.isnan(gradient_bounds).any()
        or (gradient_bounds[:, 0] > gradient_bounds[:, 1]).any()
    ):
        raise ValueError("gradient_bounds must contain two ordered lower/upper pairs.")
    derivatives = []
    for name in _DERIVATIVES:
        arg = getattr(grids, name, None)
        if arg is None:
            derivatives.append(None)
        else:
            arr = np.asarray(arg, dtype=float)
            if arr.shape != h.shape or np.isinf(arr).any():
                raise ValueError(
                    "Derivatives must match h.shape and contain no infinity"
                )
            derivatives.append(arr)

    def complete(potential_values, info):
        if fill_derivatives:
            potential_values = _fill_derivative_grids(
                grids,
                potential_values,
                derivatives,
                spacings,
                (taylor_weight, derivative_weight, curvature_weight),
                k,
                gradient_bounds,
                info,
            )
        else:
            result = copy(grids)
            result.values = potential_values
            potential_values = result
        return (potential_values, info) if return_info else potential_values

    missing = np.isnan(h)
    if not missing.any():
        info = {"n_missing": 0, "success": True, "max_constraint_violation": 0.0}
        return complete(h.copy(), info)

    # Normalize values before creating a sparse problem; shift doesn't change
    # derivative stencils or inequalities, and scaling is by a positive number.
    h_offset = float(np.mean(h[~missing]))
    scale = float(np.ptp(h[~missing]))
    if scale == 0:
        scale = max(abs(h_offset), 1.0)
    z = (h - h_offset) / scale
    ny, nx = h.shape
    idx = np.full(h.shape, -1, dtype=int)
    idx[missing] = np.arange(missing.sum())
    n = int(missing.sum())

    prediction = _taylor_prediction(h, derivatives, missing, dx, dy, k)
    target = (prediction - h_offset) / scale
    # Use normalized fields and lattice-unit derivatives to avoid division by
    # very small/large powers of physical dx and dy in optimisation matrices.
    normalized_derivatives = []
    for da, multiplier in zip(derivatives, (dx, dy, dx * dx, dx * dy, dy * dy)):
        normalized_derivatives.append(None if da is None else da * (multiplier / scale))

    # No artificial amplitude bound by default. Physical monotonicity rules out
    # new extrema inside a surrounded hole, while allowing legitimate
    # extrapolation for boundary-touching NaN regions.
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    if bounds == "rim":
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        components, ncomp = label(missing, structure=structure)
        for component in range(1, ncomp + 1):
            mask = components == component
            rim = binary_dilation(mask, structure=structure) & ~missing
            if not rim.any():
                raise ValueError("Missing component has no finite adjacent boundary")
            ids = idx[mask]
            lower[ids], upper[ids] = z[rim].min(), z[rim].max()
            target[ids] = np.clip(target[ids], lower[ids], upper[ids])

    def horizontal1(r, c):
        return [(r, c, -1.0), (r, c + 1, 1.0)]

    def vertical1(r, c):
        return [(r, c, -1.0), (r + 1, c, 1.0)]

    def horizontal2(r, c):
        return [(r, c - 1, 1.0), (r, c, -2.0), (r, c + 1, 1.0)]

    def vertical2(r, c):
        return [(r - 1, c, 1.0), (r, c, -2.0), (r + 1, c, 1.0)]

    def mix2(r, c):
        return [
            (r - 1, c - 1, 0.25),
            (r - 1, c + 1, -0.25),
            (r + 1, c - 1, -0.25),
            (r + 1, c + 1, 0.25),
        ]

    # Linear inequality constraints, with known values eliminated.
    constraint_rows = _Rows(z, idx)
    n_fixed_violations = 0
    max_fixed_violation = 0.0
    slope_bounds = gradient_bounds * np.asarray(spacings)[:, None] / scale

    def inequality(terms, sign, floor=0.0):
        # sign*stencil >= floor; second differences also use this representation
        nonlocal n_fixed_violations, max_fixed_violation
        scaled = [(r, c, sign * a) for r, c, a in terms]
        added = constraint_rows.add(scaled, rhs=floor)
        if not added:
            value = sum(a * z[r, c] for r, c, a in scaled)
            violation = floor - value
            if violation > 1e-10:
                n_fixed_violations += 1
                max_fixed_violation = max(max_fixed_violation, violation)

    for r in range(ny):
        for c in range(nx - 1):
            if np.isfinite(slope_bounds[0, 0]):
                inequality(horizontal1(r, c), 1.0, slope_bounds[0, 0])
            if np.isfinite(slope_bounds[0, 1]):
                inequality(horizontal1(r, c), -1.0, -slope_bounds[0, 1])
        for c in range(1, nx - 1):
            inequality(horizontal2(r, c), grids.curvature[0])
    for c in range(nx):
        for r in range(ny - 1):
            if np.isfinite(slope_bounds[1, 0]):
                inequality(vertical1(r, c), 1.0, slope_bounds[1, 0])
            if np.isfinite(slope_bounds[1, 1]):
                inequality(vertical1(r, c), -1.0, -slope_bounds[1, 1])
        for r in range(1, ny - 1):
            inequality(vertical2(r, c), grids.curvature[1])

    C, rhs = constraint_rows.matrix()
    constraint = LinearConstraint(C, rhs, np.inf) if C.shape[0] else None

    # Sparse quadratic prior: Taylor predictions and derivative observations.
    data_rows = _Rows(z, idx)
    for r, c in np.argwhere(missing):
        j = idx[r, c]
        data_rows.add([(r, c, 1.0)], rhs=target[j], weight=taylor_weight)

    d_x, d_y, dxx, dxy, dyy = normalized_derivatives
    if derivative_weight or curvature_weight:
        for r in range(ny):
            for c in range(nx):
                if derivative_weight:
                    if d_x is not None and c and c < nx - 1 and np.isfinite(d_x[r, c]):
                        data_rows.add(
                            [(r, c - 1, -0.5), (r, c + 1, 0.5)],
                            d_x[r, c],
                            derivative_weight,
                        )
                    if d_y is not None and r and r < ny - 1 and np.isfinite(d_y[r, c]):
                        data_rows.add(
                            [(r - 1, c, -0.5), (r + 1, c, 0.5)],
                            d_y[r, c],
                            derivative_weight,
                        )
                if curvature_weight:
                    if dxx is not None and 0 < c < nx - 1 and np.isfinite(dxx[r, c]):
                        data_rows.add(horizontal2(r, c), dxx[r, c], curvature_weight)
                    if dyy is not None and 0 < r < ny - 1 and np.isfinite(dyy[r, c]):
                        data_rows.add(vertical2(r, c), dyy[r, c], curvature_weight)
                    if (
                        dxy is not None
                        and 0 < r < ny - 1
                        and 0 < c < nx - 1
                        and np.isfinite(dxy[r, c])
                    ):
                        data_rows.add(mix2(r, c), dxy[r, c], 2 * curvature_weight)

    A, b = data_rows.matrix()
    Q = (A.T @ A).tocsc() + sparse.eye(n, format="csc") * (1e-12 * taylor_weight)
    q = -(A.T @ b)

    bend_rows = _Rows(z, idx)
    if bending_weight:
        for r in range(ny):
            for c in range(1, nx - 1):
                bend_rows.add(horizontal2(r, c))
        for c in range(nx):
            for r in range(1, ny - 1):
                bend_rows.add(vertical2(r, c))
        for r in range(1, ny - 1):
            for c in range(1, nx - 1):
                bend_rows.add(mix2(r, c), weight=2.0)
    B, d = bend_rows.matrix()

    if huber_delta is None:
        vals = []
        for stencil_kind in ("x", "y"):
            if stencil_kind == "x":
                diff = z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]
            else:
                diff = z[2:, :] - 2 * z[1:-1, :] + z[:-2, :]
            vals.extend(np.abs(diff[np.isfinite(diff)]).tolist())
        natural = max(nx, ny)
        huber_d = max(float(np.percentile(vals, 65)) if vals else 0.0, 1.0 / natural**2)
    else:
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        huber_d = huber_delta / scale

    def fun(x):
        val = 0.5 * x @ (Q @ x) + q @ x
        if B.shape[0]:
            resid = B @ x - d
            # Delta^2 * (sqrt(1 + (resid/delta)^2) - 1).
            val += bending_weight * np.sum(
                huber_d * huber_d * np.hypot(1.0, resid / huber_d) - huber_d * huber_d
            )
        return val

    def jac(x):
        grad = Q @ x + q
        if B.shape[0]:
            resid = B @ x - d
            grad = grad + bending_weight * B.T @ (
                resid / np.hypot(1.0, resid / huber_d)
            )
        return grad

    def hess(x):
        if not B.shape[0]:
            return Q
        resid = B @ x - d
        factors = bending_weight / (1.0 + (resid / huber_d) ** 2) ** 1.5
        return Q + B.T @ sparse.diags(factors) @ B

    # Taylor predictions are only a starting guess, not a constrained solution.
    options = dict(
        maxiter=maxiter,
        gtol=tol,
        xtol=tol,
        barrier_tol=tol,
        sparse_jacobian=True,
        verbose=0,
    )
    res = minimize(
        fun,
        np.clip(target, lower, upper),
        method="trust-constr",
        jac=jac,
        hess=hess,
        constraints=[] if constraint is None else [constraint],
        bounds=Bounds(lower, upper),
        options=options,
    )
    x = res.x
    max_sign_violation = max(0.0, float(np.max(rhs - C @ x))) if C.shape[0] else 0.0
    max_bounds_violation = max(0.0, float(np.max(lower - x)), float(np.max(x - upper)))
    max_violation = max(max_sign_violation, max_bounds_violation)
    if max_violation > max(10 * tol, 1e-7):
        raise RuntimeError(
            "Physical shape inequalities/bounds could not be satisfied "
            f"(scaled max violation {max_violation:.4g}). "
            "Observed grid entries may be inconsistent or bounds too restrictive. "
            f"Optimizer message: {res.message}"
        )
    if not res.success:
        warnings.warn(
            "Optimiser did not declare convergence: " + res.message, RuntimeWarning
        )
    if n_fixed_violations:
        warnings.warn(
            f"{n_fixed_violations} entirely observed stencils already violate "
            "the requested thermodynamic constraints and cannot be repaired.",
            RuntimeWarning,
        )
    out = h.copy()
    out[missing] = x * scale + h_offset
    info = {
        "n_missing": n,
        "success": bool(res.success),
        "iterations": int(res.nit),
        "optimizer_message": str(res.message),
        "n_inequality_constraints": int(C.shape[0]),
        "max_constraint_violation_scaled": float(max_violation),
        "n_unrepairable_observed_stencils": int(n_fixed_violations),
        "max_observed_stencil_violation_scaled": float(max_fixed_violation),
        "effective_huber_delta": float(huber_d * scale),
    }
    return complete(out, info)
