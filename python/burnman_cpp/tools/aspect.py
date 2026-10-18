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
"""ASPECT entropy-model tables from H(P,S) and retained elastic equilibria.

The nine-column format follows ASPECT's EntropyReader and the example in
data/material-model/entropy-table/opxtable. Pressure is in bar, entropy varies
fastest, and all extensive quantities are converted to a kilogram basis.
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError

from .. import _core
from .phase_diagram_helpers import _figure_formats, save_figure
from .potential_grid import PotentialGrid, load_potential_grid
from .properties import _saved_property_grid_batches, evaluate_property_grid


@dataclass
class AspectTableExport:
    """Export paths and diagnostics; table is None when validation rejects it."""

    table: Path | None
    plots: tuple[Path, ...]
    diagnostics: tuple[str, ...]


def write_aspect_entropy_pressure_table(
    smoothed_grids,
    saved_equilibria,
    phases,
    path,
    *,
    averaging_scheme=_core.AveragingType.VRH,
    plot=False,
    plot_formats="png",
):
    """Export smoothed thermodynamics and unsmoothed elastic Vp/Vs to ASPECT.

    ``smoothed_grids`` is an H(P,S) PotentialGrid or its saved NPZ path.
    ``saved_equilibria`` is the matching retained equilibrium grid, JSON path,
    or completed chunk manifest/directory. Supply the original candidate models
    in their original order as ``phases``. No new equilibria are calculated.

    H and S are totals on the original bulk scale. The bulk mass M (kg) is
    calculated from the saved bulk composition and atomic masses. We calculate
    T=H_S, rho=M/H_P, Cp=T/(M*H_SS), alpha=H_PS/(H_P*H_SS) and h=H/M.
    Native elastic velocities use the saved, unsmoothed P/T and compositions,
    with phase amounts/compositions held fixed during the elastic response.
    ``averaging_scheme`` is an AveragingType, just as for
    Assemblage.set_averaging_scheme; its default is AveragingType.VRH.

    Missing Vp/Vs are linearly interpolated in range-scaled (P,S) coordinates;
    outside the observed convex hull, the nearest observed velocity is used.
    Observed velocities are preserved exactly. Interpolation counts appear in
    the returned diagnostics. No smoothing is applied to either velocity.

    Columns are s[J/(kg K)], P[bar], T[K], rho[kg/m³], alpha[1/K], Cp[J/(kg K)],
    Vp[km/s], Vs[km/s], h[J/kg], with entropy varying fastest. Nonfinite values,
    nonpositive T/volume/Cp/Vp, negative Vs or thermodynamically unstable H
    curvatures reject the table. Rejection leaves any existing table untouched
    and returns ``table=None`` with diagnostics. Negative alpha is permitted.
    ``plot=True`` saves a panel for every dependent property against specific
    entropy and pressure, even when the table is rejected. Red crosses mark
    invalid thermodynamic nodes; plot_formats accepts a string or list.
    """
    grids = (
        smoothed_grids
        if isinstance(smoothed_grids, PotentialGrid)
        else load_potential_grid(smoothed_grids)
    )
    if grids.potential != "H":
        raise ValueError("ASPECT entropy tables require an H(P,S) potential grid.")
    grids.spacings()
    for name in ("d_dx", "d_dy", "d2_dx2", "d2_dxdy", "d2_dy2"):
        if getattr(grids, name) is None:
            raise ValueError("Supply H and all its first and second derivatives.")
    if not isinstance(averaging_scheme, _core.AveragingType):
        raise TypeError("averaging_scheme must be an AveragingType.")
    formats = _figure_formats(plot_formats) if plot else ()
    if plot and not formats:
        raise ValueError("Supply at least one plot format.")

    mass, vp, vs = _elastic_grids(grids, saved_equilibria, phases, averaging_scheme)
    diagnostics = []
    for name, velocity in (("Vp", vp), ("Vs", vs)):
        count = np.count_nonzero(~np.isfinite(velocity))
        if count:
            edge_count = _interpolate_velocity_gaps(velocity, grids.x, grids.y)
            filled = count - np.count_nonzero(~np.isfinite(velocity))
            diagnostics.append(
                f"{name}: interpolated {filled} of {count} missing velocities"
                f" ({edge_count} used nearest observations outside the interpolation hull)."
            )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        properties = dict(
            temperature=grids.d_dy,
            density=mass / grids.d_dx,
            thermal_expansivity=grids.d2_dxdy / (grids.d_dx * grids.d2_dy2),
            specific_heat=grids.d_dy / (mass * grids.d2_dy2),
            vp=vp / 1e3,
            vs=vs / 1e3,
            specific_enthalpy=grids.values / mass,
        )
    invalid = np.zeros(grids.values.shape, bool)
    checks = {
        "nonpositive volume H_P": grids.d_dx <= 0,
        "nonpositive temperature H_S": grids.d_dy <= 0,
        "positive pressure curvature H_PP": grids.d2_dx2 > 0,
        "nonpositive entropy curvature H_SS": grids.d2_dy2 <= 0,
        "negative pressure": np.broadcast_to(grids.x < 0, grids.values.shape),
    }
    for name in ("values", "d_dx", "d_dy", "d2_dx2", "d2_dxdy", "d2_dy2"):
        checks[f"nonfinite H derivative/value {name}"] = ~np.isfinite(
            getattr(grids, name)
        )
    for name, values in properties.items():
        checks[f"nonfinite {name}"] = ~np.isfinite(values)
    checks["nonpositive Cp"] = properties["specific_heat"] <= 0
    checks["nonpositive Vp"] = vp <= 0
    checks["negative Vs"] = vs < 0
    for name, mask in checks.items():
        if np.any(mask):
            invalid |= mask
            diagnostics.append(f"{np.count_nonzero(mask)} nodes: {name}.")
    path = Path(path)
    plots = (
        _plot_properties(properties, grids.y / mass, grids.x, invalid, path, formats)
        if plot
        else ()
    )
    if invalid.any():
        diagnostics.insert(
            0, f"Table rejected: {np.count_nonzero(invalid)} invalid nodes."
        )
        return AspectTableExport(None, plots, tuple(diagnostics))

    # Natural H arrays are (entropy, pressure); ASPECT advances entropy first.
    s, p = np.meshgrid(grids.y / mass, grids.x / 1e5, indexing="ij")
    columns = [s, p, *properties.values()]
    header = (
        "BurnMan: smoothed H(P,S); unsmoothed elastic velocities; "
        f"averaging_scheme={averaging_scheme.name}\n"
        f"Independent variables: specific entropy and pressure.\n"
        f"POINTS: {len(grids.y)} {len(grids.x)}"
    )
    names = "s,J/K/kg P(bar) T(K) rho,kg/m3 alpha,1/K cp,J/K/kg vp,km/s vs,km/s h,J/kg"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write("\n".join(f"# {line}" for line in header.splitlines()) + "\n")
            stream.write(names + "\n")
            np.savetxt(
                stream, np.column_stack([a.T.ravel() for a in columns]), fmt="%.16e"
            )
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return AspectTableExport(path, plots, tuple(diagnostics))


def _elastic_grids(grids, source, phases, averaging_scheme):
    """Replay exact active faces batch by batch; verify axes and fixed bulk."""
    x, y, batches = _saved_property_grid_batches(source)
    if not (np.array_equal(x, grids.x) and np.array_equal(y, grids.y)):
        raise ValueError("Equilibrium coordinates must match the H(P,S) grid.")
    vp, vs = (np.full(grids.values.shape, np.nan) for _ in range(2))
    covered = np.zeros(grids.values.shape, bool)
    mass, bulk = None, None
    for rows, cols, saved in batches:
        if saved.diagram_type != "PS":
            raise ValueError("Saved equilibria must use P,S coordinates.")
        if not (np.array_equal(saved.x, x[cols]) and np.array_equal(saved.y, y[rows])):
            raise ValueError("Saved batch coordinates disagree with the H(P,S) grid.")
        if np.any(covered[rows, cols]):
            raise ValueError("Saved equilibrium batches overlap.")
        covered[rows, cols] = True
        if bulk is None:
            bulk = saved.bulk_composition
            # Mineral datasets may round molar masses differently between
            # phases. Elemental amounts define a single mass for the bulk.
            mass = sum(_core.Composition(bulk, "molar").mass_composition.values())
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("The saved bulk composition must have positive mass.")
        elif saved.bulk_composition != bulk:
            raise ValueError("Saved batches must have the same bulk composition.")
        replay = evaluate_property_grid(
            saved,
            phases,
            ["p_wave_velocity", "shear_wave_velocity"],
            relaxed=False,
            averaging_scheme=averaging_scheme,
        )
        observed = saved.equilibrium_success
        if grids.n_moles is not None and observed.any():
            moles = np.full(observed.shape, np.nan)
            for j, i in np.argwhere(observed):
                moles[j, i] = saved.solution_at(int(j), int(i)).n_moles
            if not np.allclose(
                grids.n_moles[rows, cols][observed], moles[observed], rtol=1e-8, atol=0
            ):
                raise ValueError(
                    "The potential and equilibrium grids use different bulk amounts."
                )
        vp[rows, cols] = replay.values["p_wave_velocity"]
        vs[rows, cols] = replay.values["shear_wave_velocity"]
    if not covered.all() or mass is None:
        raise ValueError("Supply a complete grid of retained equilibrium samples.")
    return mass, vp, vs


def _interpolate_velocity_gaps(values, x, y):
    """Fill only absent velocities, preserving every finite native observation."""
    p, s = np.meshgrid((x - x[0]) / np.ptp(x), (y - y[0]) / np.ptp(y))
    points = np.column_stack((p.ravel(), s.ravel()))
    observed = np.isfinite(values).ravel()
    if not observed.any():
        return 0  # The caller rejects the nonfinite column and can still plot it.
    known, targets = points[observed], points[~observed]
    data = values.ravel()[observed]
    estimates = np.full(len(targets), np.nan)
    if len(known) >= 3:
        try:
            estimates = LinearNDInterpolator(known, data)(targets)
        except QhullError:
            pass  # Collinear observations supply nearest values only.
    outside = ~np.isfinite(estimates)
    if outside.any():
        estimates[outside] = NearestNDInterpolator(known, data)(targets[outside])
    values[~observed.reshape(values.shape)] = estimates
    return int(np.count_nonzero(outside))


def _plot_properties(properties, entropy, pressure, invalid, path, formats):
    """Plot every exported dependent property; highlight rejected nodes."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    labels = (
        "Temperature (K)",
        "Density (kg/m³)",
        "Thermal expansivity (1/K)",
        "Specific heat Cp (J/(kg K))",
        "Vp (km/s)",
        "Vs (km/s)",
        "Specific enthalpy (J/kg)",
    )
    fig, axes = plt.subplots(4, 2, figsize=(13, 17), layout="constrained")
    for ax, (name, values), label in zip(axes.flat, properties.items(), labels):
        data = np.ma.masked_invalid(values.T)
        norm = None
        if name == "specific_heat" and np.any(np.isfinite(values) & (values > 0)):
            data = np.ma.masked_less_equal(data, 0)
            norm = LogNorm()
        image = ax.pcolormesh(entropy, pressure / 1e9, data, shading="auto", norm=norm)
        fig.colorbar(image, ax=ax, label=label)
        j, i = np.nonzero(invalid)
        ax.scatter(
            entropy[j], pressure[i] / 1e9, marker="x", s=7, c="red", linewidths=0.4
        )
        ax.set(
            xlabel="Specific entropy (J/(kg K))", ylabel="Pressure (GPa)", title=label
        )
    axes.flat[-1].set_axis_off()
    fig.suptitle("ASPECT properties (red crosses: invalid nodes)")
    return save_figure(
        fig,
        path.with_name(path.stem + "_properties").with_suffix("." + formats[0]),
        formats=formats,
    )
