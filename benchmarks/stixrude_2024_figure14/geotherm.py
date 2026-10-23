#!/usr/bin/env python3
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
"""Compare Figure 14's geotherm with an isentrope plus an erfc boundary layer."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc

import burnman_cpp as bm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stixrude_2024 import CMB_DEPTH, PREM, bulk_composition, candidate_phases


def boundary_layer_temperature(
    height_km,
    isentrope_temperature,
    isentrope_temperature_at_cmb,
    *,
    basal_temperature=4000.0,
    height_scale_km=180.0,
):
    """Add an erfc layer whose amplitude gives the requested CMB temperature.

    Height is measured upward from the CMB. Obtain the basal isentrope
    temperature from its equilibrium calculation, never from a fitted or
    rounded constant. This is the prescribed Figure 14 comparison model;
    the paper does not explicitly give its geotherm formula.
    """
    amplitude = basal_temperature - isentrope_temperature_at_cmb
    return np.asarray(isentrope_temperature) + amplitude * erfc(
        np.asarray(height_km) / height_scale_km
    )


def published_geotherm():
    """Digitize the green line in the original embedded Figure 14 raster.

    Read the median green pixel in each image row, then use the independent
    PDF tick calibration. Retain the source image's finite resolution; do not
    smooth, interpolate missing rows or fit an assumed geotherm during extraction.
    """
    config = json.loads((HERE / "reference/axis_registration.json").read_text())
    rgb = mpimg.imread(HERE / "reference/expansivity.jpg").astype(float)
    red, green, blue = rgb.transpose(2, 0, 1)
    mask = (green > red + 35) & (green > blue + 35) & (red < 140) & (blue < 160)
    rows, columns = [], []
    for row in range(mask.shape[0]):
        pixels = np.flatnonzero(mask[row, :240])
        if pixels.size:
            rows.append(row + 0.5)
            columns.append(np.median(pixels) + 0.5)
    sx, _, _, sy, x0, y0 = config["raster_transform"]
    x, y = x0 + sx * np.asarray(columns), y0 + sy * np.asarray(rows)
    top, bottom = config["height_frame_pdf_points"]
    height = 600 * (bottom - y) / (bottom - top)
    # Register between the 1000 K and 4000 K tick centres.
    first, *_, last = config["temperature_tick_centres_pdf_points"]
    temperature = 1000 + (x - first) * 3000 / (last - first)
    keep = (height >= 0) & (height <= 600)
    order = np.argsort(height[keep])
    return height[keep][order], temperature[keep][order]


def calculate_geotherm(output_dir, *, points=601):
    """Solve the 1400 K pyrolite isentrope and save its boundary-layer comparison.

    Return increasing height, pressure, isentropic temperature and prescribed
    boundary-layer temperature arrays spanning 0–600 km above the PREM CMB.
    The published curve supplies an independent comparison, not fitted inputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bulk, phases = bulk_composition("pyrolite"), candidate_phases()
    surface = bm.stable_equilibrium(bulk.atomic_composition, phases, 0.0, 1400.0)
    if not surface.success or surface.outside_model_domain:
        raise RuntimeError(surface.message)
    pressure_range = PREM.pressure([CMB_DEPTH - 600e3, CMB_DEPTH])
    path = bm.calculate_equilibrium_path(
        bulk,
        phases,
        dict(P=pressure_range[0], S=surface.entropy),
        dict(P=pressure_range[1], S=surface.entropy),
        n_points=points,
        temperature_range=(1200.0, 4500.0),
    )
    if not path.success.all():
        raise RuntimeError("; ".join(path.diagnostics))
    pressure, isentrope = path.coordinates("P")[::-1], path.coordinates("T")[::-1]
    height = (CMB_DEPTH - PREM.depth(pressure)) / 1000
    # The endpoint is the CMB equilibrium itself, not a temperature extrapolation.
    cmb_isentrope_temperature = isentrope[0]
    prescribed = boundary_layer_temperature(
        height, isentrope, cmb_isentrope_temperature
    )
    ref_height, reference = published_geotherm()
    ref_isentrope = np.interp(ref_height, height, isentrope)
    physical_model = boundary_layer_temperature(
        ref_height, ref_isentrope, cmb_isentrope_temperature
    )
    addition = reference - ref_isentrope
    fits = {}
    for name, function, initial in [
        ("exponential", lambda h, L: np.exp(-h / L), [1800.0, 100.0]),
        ("erfc", lambda h, L: erfc(h / L), [4000.0 - cmb_isentrope_temperature, 180.0]),
    ]:
        fit = least_squares(
            lambda q: q[0] * function(ref_height, q[1]) - addition,
            initial,
            bounds=([0.0, 1.0], [5000.0, 3000.0]),
        )
        fits[name] = dict(
            amplitude_K=float(fit.x[0]),
            height_scale_km=float(fit.x[1]),
            rms_error_K=float(np.sqrt(np.mean(fit.fun**2))),
            maximum_error_K=float(np.max(abs(fit.fun))),
        )
    error = physical_model - reference
    report = dict(
        potential_temperature_K=1400.0,
        surface_entropy_J_per_K=surface.entropy,
        cmb_isentrope_temperature_K=float(cmb_isentrope_temperature),
        prescribed_cmb_temperature_K=4000.0,
        prescribed_height_scale_km=180.0,
        calculated_amplitude_K=float(4000.0 - cmb_isentrope_temperature),
        prescribed_model_rms_error_K=float(np.sqrt(np.mean(error**2))),
        prescribed_model_maximum_error_K=float(np.max(abs(error))),
        comparison_fits=fits,
        digitized_rows=len(ref_height),
        reference_horizontal_pixel_K=0.359703 * 3000 / (251.025847 - 186.566462),
        isentrope_equilibrium_solves=path.equilibrium_solves,
        max_relative_entropy_error=float(
            np.max(abs(path.coordinates("S") / surface.entropy - 1))
        ),
    )
    (output_dir / "geotherm_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez_compressed(
        output_dir / "geotherm.npz",
        height_km=height,
        pressure=pressure,
        isentrope_temperature=isentrope,
        temperature=prescribed,
        reference_height_km=ref_height,
        reference_temperature=reference,
        reference_isentrope_temperature=ref_isentrope,
        prescribed_model_residual_K=error,
    )
    fig, (ax, residual) = plt.subplots(
        1,
        2,
        figsize=(10, 6),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    ax.plot(
        reference,
        ref_height,
        color="#008c18",
        lw=2.5,
        label="Published green geotherm (digitized)",
    )
    ax.plot(isentrope, height, color="0.3", lw=1.5, label="Calculated 1400 K isentrope")
    ax.plot(
        prescribed,
        height,
        color="#ba158b",
        ls="--",
        lw=1.7,
        label="Isentrope + [4000 − T(CMB)] erfc(h / 180 km)",
    )
    ax.set(
        xlabel="Temperature (K)",
        ylabel="Height above CMB (km)",
        ylim=(0, 600),
        xlim=(2000, 4100),
    )
    ax.legend(fontsize=8, loc="upper right")
    pixel = report["reference_horizontal_pixel_K"]
    residual.axvspan(
        -pixel, pixel, color="0.92", label="One horizontal reference-image pixel"
    )
    residual.axvline(0, color="0.4", lw=0.8)
    residual.plot(error, ref_height, color="#ba158b", lw=1.3)
    residual.set(xlabel="Model minus published temperature (K)")
    residual.text(
        0.05,
        0.98,
        f"RMS: {report['prescribed_model_rms_error_K']:.1f} K\n"
        f"Maximum |error|: {report['prescribed_model_maximum_error_K']:.1f} K",
        transform=residual.transAxes,
        va="top",
        fontsize=10,
    )
    residual.legend(fontsize=7, loc="lower right")
    fig.suptitle("Figure 14 geotherm: 180 km erfc layer, 4000 K at the CMB")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"geotherm_comparison.{extension}", dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    return height, pressure, isentrope, prescribed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parents[1] / "build/benchmarks/stixrude_2024_figure14",
    )
    parser.add_argument("--points", type=int, default=601)
    args = parser.parse_args()
    calculate_geotherm(args.output_dir, points=args.points)
