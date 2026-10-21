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
"""Reproduce SLB2024 Figure 13 modes, native iron and ferropericlase spin.

Recalculate atomic phase fractions and native iron, with a closed oxygen budget
and bulk Fe3+/total Fe = 0.02. Omit ferric/spin shading and oxygen fugacity.
PREM supplies only the pressure-depth conversion, not the temperature profile.
Overlay the saved calculated curves on the included published reference panels.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import burnman_cpp as bm
from burnman_cpp.minerals import SLB24

HERE = Path(__file__).resolve().parent
PAPER = "https://doi.org/10.1093/gji/ggae126"


# Share the published bulk and complete candidate catalogue with Figure 14.
sys.path.insert(0, str(HERE.parent))
from stixrude_2024 import bulk_composition, candidate_phases, save_equilibrium_path


def pyrolite_composition():
    """Table 1 pyrolite with initial Fe3+/total Fe = 0.02, normalized to 100 g."""
    return bulk_composition("pyrolite")


def ferropericlase_low_spin_fraction(path, fp_model):
    """Eq. (24): n(wustite_LS)/(n(wustite_HS)+n(wustite_LS)), within fp.

    Sum species amounts over any coexisting fp copies before taking the ratio.
    The ferric magnetite endmember is outside this ferrous spin ratio. Leave
    the ratio undefined (NaN) where no ferrous fp is present.
    """
    candidate = path.phase_names.index(fp_model.name)
    names = list(fp_model.endmember_names)
    high = names.index(SLB24.wustite().name)
    low = names.index(SLB24.wustite_low_spin().name)
    fraction = np.full(len(path.states), np.nan)
    for i, (state, success) in enumerate(zip(path.states, path.success)):
        if not success:
            continue
        species = sum(
            (
                p.amount * np.asarray(p.composition)
                for p in state.phases
                if p.candidate_index == candidate
            ),
            np.zeros(len(names)),
        )
        total = species[low] + species[high]
        if total > 0.0:
            fraction[i] = species[low] / total
    return fraction


def plot_figure(path, depth_km, native_iron, fp_low_spin):
    """Stack modes and magnify the metal band downward from its upper edge."""
    fractions = path.fractions()
    present = {
        name for j, name in enumerate(path.phase_names) if np.any(fractions[:, j] > 0)
    }
    order = [
        name
        for name in (
            "fp",
            "ol",
            "wa",
            "ri",
            "gt",
            "ak",
            "opx",
            "hpcpx",
            "cpx",
            "bg",
            "ppv",
            "cf",
            "nal",
            "q",
            "coe",
            "st",
            "seif",
            "ky",
            "neph",
            "sp",
            "plg",
            "wo",
            "pwo",
            "capv",
            "Fe-bcc",
            "Fe-fcc",
            "Fe-hcp",
        )
        if name in present
    ]
    colors = dict(
        fp="0.78",
        bg="#c7edc2",
        ppv="#fff0dd",
        gt="#f4caca",
        cpx="#ced5f4",
        sp="#efbfef",
        cf="0.85",
        nal="0.9",
    )
    colors = {name: colors.get(name, "white") for name in order}
    fig, (ax, spin_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 2]},
    )
    bm.plot_mode_box(
        path, ax, x=depth_km, xlabel="Depth (km)", order=order, colors=colors
    )
    lower = np.zeros(len(path.states))
    for name in order:
        mode = fractions[:, path.phase_names.index(name)]
        if np.any(mode > 0.025):
            # Label the centre of the largest visible interval, keeping narrow
            # polymorph fields distinct while suppressing trace-phase clutter.
            visible = np.flatnonzero(mode > 0.025)
            runs = np.split(visible, np.flatnonzero(np.diff(visible) > 1) + 1)
            run = max(runs, key=lambda r: depth_km[r[-1]] - depth_km[r[0]])
            index = run[len(run) // 2]
            position = (depth_km[index], lower[index] + 0.5 * mode[index])
            if name in ("sp", "plg"):
                ax.annotate(
                    name,
                    position,
                    xytext=(0, 15) if name == "sp" else (-25, 0),
                    textcoords="offset points",
                    ha="center" if name == "sp" else "right",
                    va="bottom" if name == "sp" else "center",
                    fontsize=10,
                    arrowprops=dict(arrowstyle="->", lw=0.7),
                )
            else:
                ax.text(*position, name, ha="center", va="center", fontsize=11)
        lower += mode
    # In the paper the enlarged metal abundance is measured down from 1:
    # a dashed ordinate of 0.75 therefore represents X_Fe = 0.0025.
    ax.plot(
        depth_km,
        1 - 100 * native_iron,
        "k--",
        lw=1.6,
        label="Native Fe ×100 (downward from 1)",
    )
    ax.set(
        xlim=(0, depth_km[-1]),
        ylabel="Atomic fraction",
        title="SLB2024 pyrolite: 1600 K surface isentrope, bulk Fe³⁺/ΣFe = 0.02",
    )
    ax.set_title(ax.get_title(), pad=24)
    ax.set_xlabel("")
    # The mode labels are inside their fields; keep only the enlarged Fe key.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-1:], labels[-1:], loc="lower right", framealpha=0.9)
    ax.tick_params(direction="in", top=True, right=True)
    spin_ax.plot(depth_km, fp_low_spin, "k-", lw=1.5)
    spin_ax.set(xlabel="Depth (km)", ylabel="Low-spin fraction in fp", ylim=(0, 1))
    spin_ax.tick_params(direction="in", top=True, right=True)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=801)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parents[1] / "build/benchmarks/stixrude_2024_figure13",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = perf_counter()
    bulk, phases = pyrolite_composition(), candidate_phases()
    settings = bm.PhaseDiagramSettings()
    settings.verbose = args.verbose
    surface = bm.stable_equilibrium(
        bulk.atomic_composition, phases, 0.0, 1600.0, settings
    )
    if not surface.success or surface.outside_model_domain:
        raise RuntimeError(surface.message)
    print(f"Surface entropy: {surface.entropy:.9f} J/K per 100 g", flush=True)
    prem = np.loadtxt(HERE / "reference" / "prem_depth_pressure.txt")
    path = bm.calculate_equilibrium_path(
        bulk,
        phases,
        {"P": 0.0, "S": surface.entropy},
        {"P": float(prem[-1, 1]), "S": surface.entropy},
        n_points=args.points,
        settings=settings,
        temperature_range=(1400.0, 4500.0),
    )
    fractions = path.fractions()
    target_pressure = np.linspace(0.0, prem[-1, 1], args.points)
    pressure = path.coordinates("P")
    depth = np.interp(target_pressure, prem[:, 1], prem[:, 0]) / 1e3
    metal_indices = [
        j for j, name in enumerate(path.phase_names) if name.startswith("Fe-")
    ]
    native_iron = fractions[:, metal_indices].sum(axis=1)
    fp_model = next(p for p in phases if p.name == "fp")
    fp_low_spin = ferropericlase_low_spin_fraction(path, fp_model)
    if (np.isfinite(fp_low_spin) & ((fp_low_spin < 0) | (fp_low_spin > 1))).any():
        raise RuntimeError("Ferropericlase spin fraction is outside 0..1.")
    fig = plot_figure(path, depth, native_iron, fp_low_spin)
    mode_curves = fig.axes[0].lines[:-1]
    boundaries = np.array([line.get_ydata() for line in mode_curves]).T
    np.savez_compressed(
        args.output_dir / "mode_box.npz",
        pressure=pressure,
        target_pressure=target_pressure,
        depth_km=depth,
        temperature=path.coordinates("T"),
        entropy=path.coordinates("S"),
        volume=path.coordinates("V"),
        phase_names=path.phase_names,
        atomic_fractions=fractions,
        molar_amounts=path.molar_amounts,
        atomic_amounts=path.atomic_amounts,
        native_iron_atomic_fraction=native_iron,
        success=path.success,
        cumulative_atomic_fractions=boundaries,
        fp_low_spin_fraction=fp_low_spin,
    )
    save_equilibrium_path(path, bulk, args.output_dir / "equilibria.json.gz")
    for extension in ("png", "svg"):
        fig.savefig(args.output_dir / f"figure13.{extension}", dpi=240)
    plt.close(fig)
    peak = int(np.nanargmax(native_iron))
    report = dict(
        paper=PAPER,
        potential_temperature_K=1600.0,
        bulk_ferric_fraction=0.02,
        bulk_elements=bulk.atomic_composition,
        oxide_wt_percent={k: v * 1000 for k, v in bulk.mass_composition.items()},
        dataset="SLB24",
        burnman_cpp_version=bm.__version__,
        settings=settings.to_dict(),
        pressure_range_Pa=[0.0, float(prem[-1, 1])],
        temperature_search_range_K=[1400.0, 4500.0],
        entropy_J_per_K=surface.entropy,
        n_points=args.points,
        successful_points=int(path.success.sum()),
        diagnostics=path.diagnostics,
        max_relative_entropy_error=float(
            np.nanmax(np.abs(path.coordinates("S") / surface.entropy - 1))
        ),
        max_relative_mass_balance_error=max(
            s.mass_balance_error for s in path.states if s.success
        ),
        native_iron_peak=dict(
            atomic_fraction=float(native_iron[peak]),
            depth_km=float(depth[peak]),
            pressure_GPa=float(pressure[peak] / 1e9),
            temperature_K=float(path.coordinates("T")[peak]),
        ),
        fp_low_spin=dict(
            definition="n_wuls / (n_wu + n_wuls), Eq. (24)",
            high_spin_endmember=SLB24.wustite().name,
            low_spin_endmember=SLB24.wustite_low_spin().name,
            valid_points=int(np.isfinite(fp_low_spin).sum()),
            core_mantle_boundary_fraction=float(fp_low_spin[-1]),
        ),
        maximum_relative_atomic_balance_error=float(
            np.nanmax(
                np.abs(
                    path.atomic_amounts.sum(axis=1)
                    / sum(bulk.atomic_composition.values())
                    - 1
                )
            )
        ),
        elapsed_seconds=perf_counter() - start_time,
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    from stixrude_2024_figure13.overlay import make_overlay

    make_overlay(args.output_dir / "mode_box.npz", args.output_dir)
    print(f"Figure: {args.output_dir / 'figure13.png'}", flush=True)


if __name__ == "__main__":
    main()
