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
"""Reproduce both panels of SLB2024 Figure 14 near the core–mantle boundary.

Calculate pyrolite/harzburgite/basalt Vs along a 1400 K pyrolite isentrope
modified by a 180 km erfc layer, reaching exactly 4000 K at the CMB. Calculate
relaxed pyrolite expansivity and trace its phase boundaries over the whole map.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

import burnman_cpp as bm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stixrude_2024 import (
    CMB_DEPTH,
    PREM,
    bulk_composition,
    candidate_phases,
    FERRIC_FRACTIONS,
    save_equilibrium_path,
)
from stixrude_2024_figure14.geotherm import calculate_geotherm

COLORS = dict(pyrolite="red", harzburgite="#009900", basalt="#003eff")


def height_at_pressure(pressure):
    """Convert pressure to km above the mantle-side PREM CMB."""
    return (CMB_DEPTH - PREM.depth(pressure)) / 1000


def plot_figure(data, diagram, output_dir):
    """Plot actual Vs, relaxed alpha, solved phase boundaries and the geotherm."""
    fig, (vel, thermal) = plt.subplots(
        1,
        2,
        figsize=(11, 8),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    for name, color in COLORS.items():
        vel.plot(
            data[name + "_vs"] / 1000,
            data["profile_height_km"],
            color=color,
            lw=1.6,
            label=name.capitalize(),
        )
    vel.set(
        xlabel="Shear-wave velocity (km s⁻¹)",
        ylabel="Height above CMB (km)",
        ylim=(0, 600),
        xlim=(6.8, 7.5),
        title="Equilibrated lithologies",
    )
    vel.legend(loc="upper left", fontsize=9)
    mesh = thermal.pcolormesh(
        data["grid_temperature"],
        data["grid_height_km"],
        np.ma.masked_invalid(data["thermal_expansivity"] * 1e5),
        cmap="bwr",
        vmin=0,
        vmax=5,
        shading="auto",
        rasterized=True,
    )
    colorbar = fig.colorbar(mesh, ax=thermal, extend="both", pad=0.02)
    colorbar.set_label("Relaxed thermal expansivity (10⁻⁵ K⁻¹)")
    for line in diagram.boundaries:
        thermal.plot(
            [p.temperature for p in line.points],
            height_at_pressure([p.pressure for p in line.points]),
            color="black",
            lw=0.7,
        )
    thermal.set(
        xlim=(800, 4800),
        ylim=(0, 600),
        xlabel="Temperature (K)",
        title="Pyrolite phase equilibria and expansivity",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    font = FontProperties(size=7)
    # Reuse the polygon builder's rectangle test so labels remain inside their
    # actual field. Retain narrow fields in the complete assemblage key.
    geometry = bm.phase_diagram_field_polygons(diagram)
    key = []
    for polygon in geometry.polygons:
        names = [diagram.phase_names[i].replace("Fe-hcp", "Fe") for i in polygon.phases]
        label = " ".join(names)
        key.append(f"Field {polygon.field_number}: {label}")
        pressure, temperature = polygon.label_position
        lines = [" ".join(names[i : i + 3]) for i in range(0, len(names), 3)]
        label = "\n".join(lines)
        half_width = (
            max(
                renderer.get_text_width_height_descent(s, font, False)[0] for s in lines
            )
            * 0.55
        )
        half_height = len(lines) * 7 * fig.dpi / 72 * 0.65
        centre = thermal.transData.transform(
            (temperature, height_at_pressure(pressure))
        )
        low = thermal.transData.inverted().transform(centre - [half_width, half_height])
        high = thermal.transData.inverted().transform(
            centre + [half_width, half_height]
        )
        pressures = PREM.pressure(CMB_DEPTH - np.array([low[1], high[1]]) * 1000)
        half_size = [max(abs(pressures - pressure)), 0.5 * (high[0] - low[0])]
        if not polygon.contains_rectangle(polygon.label_position, half_size):
            continue
        thermal.text(
            temperature,
            height_at_pressure(pressure),
            label,
            fontsize=7,
            ha="center",
            va="center",
            clip_on=True,
        )
    (output_dir / "field_labels.txt").write_text("\n".join(key) + "\n")
    thermal.plot(
        data["geotherm_temperature"],
        data["geotherm_height_km"],
        color="#00a000",
        lw=2,
        label="1400 K isentrope + erfc layer",
    )
    thermal.set(
        xlabel="Temperature (K)",
        xlim=(800, 4800),
        title="Pyrolite phase equilibria and expansivity",
    )
    thermal.legend(loc="upper right", fontsize=8)
    for ax in (vel, thermal):
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_yticks(np.arange(0, 601, 100))
    fig.suptitle("SLB2024 Figure 14: 180 km basal layer, T(CMB) = 4000 K")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"figure14.{extension}", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=601)
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=(121, 161), metavar=("NP", "NT")
    )
    parser.add_argument("--diagram-json", type=Path)
    parser.add_argument(
        "--replot", action="store_true", help="Use existing saved data; no equilibria."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parents[1] / "build/benchmarks/stixrude_2024_figure14",
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        with np.load(out / "figure14.npz", allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        diagram = bm.PhaseDiagramResult.from_dict(
            bm.load_phase_diagram(out / "phase_diagram.json")
        )
    else:
        start = perf_counter()
        height, pressure, isentrope, temperature = calculate_geotherm(
            out, points=args.points
        )
        phases = candidate_phases()
        data = dict(
            geotherm_height_km=height,
            geotherm_temperature=temperature,
            geotherm_isentrope_temperature=isentrope,
        )
        # Sample common heights for every lithology, while retaining the curved
        # geotherm itself. Chemistry and spin populations equilibrate independently.
        profile_height = np.linspace(600.0, 0.0, args.points)
        profile_pressure = np.interp(profile_height, height, pressure)
        profile_temperature = np.interp(profile_height, height, temperature)
        data.update(
            profile_height_km=profile_height,
            profile_pressure=profile_pressure,
            profile_temperature=profile_temperature,
            phase_names=np.array([p.name for p in phases]),
        )
        report = dict(
            dataset="SLB_2024",
            paper="https://doi.org/10.1093/gji/ggae126",
            burnman_cpp_version=bm.__version__,
            averaging_scheme="VRH",
            path_settings=bm.PhaseDiagramSettings().to_dict(),
            profiles={},
            map={},
        )
        for name in COLORS:
            bulk = bulk_composition(name)
            tick = perf_counter()
            path = bm.sample_equilibrium_path(
                bulk,
                phases,
                dict(P=profile_pressure, T=profile_temperature),
                properties=["shear_wave_velocity", "density", "shear_modulus"],
            )
            data[name + "_vs"] = path.values["shear_wave_velocity"]
            data[name + "_density"] = path.values["density"]
            data[name + "_shear_modulus"] = path.values["shear_modulus"]
            data[name + "_atomic_fractions"] = path.fractions()
            data[name + "_success"] = path.success
            save_equilibrium_path(path, bulk, out / f"{name}_equilibria.json.gz")
            report["profiles"][name] = dict(
                elapsed_seconds=perf_counter() - tick,
                points=len(path.states),
                successful_points=int(path.success.sum()),
                equilibrium_solves=path.equilibrium_solves,
                equilibrium_diagnostics=path.diagnostics,
                property_diagnostics=path.property_diagnostics,
                initial_ferric_fraction=FERRIC_FRACTIONS[name],
                oxide_wt_percent={
                    k: v * 1000 for k, v in bulk.mass_composition.items()
                },
                max_relative_mass_balance_error=max(
                    (s.mass_balance_error for s in path.states if s.success),
                    default=None,
                ),
                max_relative_velocity_identity_error=float(
                    np.nanmax(
                        abs(
                            path.values["shear_wave_velocity"] ** 2
                            * path.values["density"]
                            / path.values["shear_modulus"]
                            - 1
                        )
                    )
                ),
            )
            print(name, json.dumps(report["profiles"][name]), flush=True)
        bulk = bulk_composition("pyrolite")
        tick = perf_counter()
        if args.diagram_json:
            diagram = bm.PhaseDiagramResult.from_dict(
                bm.load_phase_diagram(args.diagram_json)
            )
        else:
            settings = bm.PhaseDiagramSettings()
            settings.pressure_seeds = 9
            settings.temperature_seeds = 17
            settings.verbose = args.verbose
            diagram = bm.phase_diagram(
                bulk.atomic_composition,
                phases,
                pressure_range=(float(pressure[-1]), float(pressure[0])),
                temperature_range=(800.0, 4800.0),
                settings=settings,
            )
        bm.save_phase_diagram(diagram, out / "phase_diagram.json")
        if not diagram.resolved or diagram.diagnostics:
            raise RuntimeError(
                "Map gridding requires a resolved diagram: "
                + "; ".join(diagram.diagnostics)
            )
        # Refine equilibrium points on accepted boundaries, preserving topology.
        diagram = bm.refine_phase_diagram(
            bulk.atomic_composition, phases, diagram, resolution=tuple(args.resolution)
        )
        bm.save_phase_diagram(diagram, out / "phase_diagram.json")
        if not diagram.resolved or diagram.diagnostics:
            raise RuntimeError(
                "Refined diagram is unresolved: " + "; ".join(diagram.diagnostics)
            )
        print(
            f"Resolved phase diagram: {len(diagram.boundaries)} boundaries", flush=True
        )
        grid = bm.phase_diagram_grid(
            diagram,
            phases,
            ["thermal_expansivity"],
            resolution=tuple(args.resolution),
            relaxed=True,
            store_solutions=True,
        )
        bm.save_property_grid(grid, out / "expansivity_grid.json.gz")
        data.update(
            grid_temperature=grid.y,
            grid_pressure=grid.x,
            grid_height_km=height_at_pressure(grid.x),
            thermal_expansivity=grid.values["thermal_expansivity"].T,
            grid_success=grid.success.T,
            grid_equilibrium_success=grid.equilibrium_success.T,
        )
        report["map"] = dict(
            elapsed_seconds=perf_counter() - tick,
            diagram_resolved=diagram.resolved,
            settings=diagram.settings.to_dict(),
            pressure_range_Pa=list(diagram.pressure_range),
            temperature_range_K=list(diagram.temperature_range),
            grid_resolution=list(args.resolution),
            diagram_diagnostics=diagram.diagnostics,
            spacing_diagnostics=diagram.spacing_diagnostics,
            diagram_equilibrium_solves=diagram.equilibrium_solves,
            grid_equilibrium_solves=grid.equilibrium_solves,
            points=int(grid.success.size),
            successful_properties=int(grid.success.sum()),
            successful_equilibria=int(grid.equilibrium_success.sum()),
            diagnostics=grid.diagnostics,
            thermodynamic_validation=grid.thermodynamic_validation,
            minimum_alpha_per_K=float(np.nanmin(data["thermal_expansivity"])),
            maximum_alpha_per_K=float(np.nanmax(data["thermal_expansivity"])),
        )
        report["elapsed_seconds"] = perf_counter() - start
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(out / "figure14.npz", **data)
        print(json.dumps(report["map"], indent=2), flush=True)
        print(f"Total calculation time: {report['elapsed_seconds']:.3f} s", flush=True)
    plot_figure(data, diagram, out)
    from stixrude_2024_figure14.overlay import make_overlay

    make_overlay(out / "figure14.npz", out / "phase_diagram.json", out)


if __name__ == "__main__":
    main()
