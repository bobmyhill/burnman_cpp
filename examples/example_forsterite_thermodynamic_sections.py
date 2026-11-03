#!/usr/bin/env python3
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

"""
Mg2SiO4 phase diagrams and P, T, S, V contours
in four coordinate systems: P-T, P-S, V-T, and V-S.

This example recreates the phase boundaries and contours
in the blog post illustration:
https://blogs.egu.eu/divisions/gd/2021/01/27/thermodynamics-and-geodynamics-the-perfect-couple/
It uses the SLB2011 periclase, bridgmanite, akimotoite and ringwoodite models.
"""

import json
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011
from burnman_cpp.tools.phase_diagram_helpers import (
    save_figure,
    write_phase_diagram,
)

OUTPUT_DIR = Path(__file__).resolve().with_name("forsterite_sections_output")
SEEDS = 9
VOLUME_UNIT = "kg/m3"  # Or "m3" or "cm3" for volume rather than density.
ENTROPY_UNIT = "kB/atom"  # Or "J/K" or "kJ/K".

BULK = bm.Composition({"Mg2SiO4": 1.0}, "molar")
BULK_MASS = sum(BULK.mass_composition.values())
BULK_ATOMS = sum(BULK.atomic_composition.values())
ENTROPY_SCALE = BULK_ATOMS * bm.constants.gas_constant
ALIASES = ["per", "bdg", "aki", "rw"]
# Native SI targets; alternate solid/dotted S and T levels as in the figure.
CONTOURS = {
    "P": (bm.PressureConstraint, [p * 1.0e8 for p in range(233, 238)], "blue"),
    "T": (bm.TemperatureConstraint, list(range(1300, 1901, 50)), "orange"),
    "S": (
        bm.EntropyConstraint,
        [s * BULK_ATOMS for s in range(42, 49)],
        "red",
    ),
    "V": (bm.VolumeConstraint, [BULK_MASS / 4200.0], "cyan"),
}


def candidate_phases():
    phases = [
        SLB_2011.periclase(),
        SLB_2011.mg_perovskite(),
        SLB_2011.mg_akimotoite(),
        SLB_2011.mg_ringwoodite(),
    ]
    for phase, name in zip(phases, ALIASES):
        phase.set_name(name)
    return phases


def calculate(verbose=False):
    pressure_range = (23.2e9, 23.9e9)
    temperature_range = (1400.0, 1700.0)
    entropy_range = tuple(s * ENTROPY_SCALE for s in (5.2, 5.8))
    volume_range = tuple(BULK_MASS / rho for rho in (4500.0, 3600.0))
    domains = {
        "PT": dict(pressure_range=pressure_range, temperature_range=temperature_range),
        "PS": dict(
            pressure_range=pressure_range,
            entropy_range=entropy_range,
            temperature_range=(500.0, 3000.0),
        ),
        "TV": dict(
            temperature_range=temperature_range,
            volume_range=volume_range,
            pressure_range=(0.0, 100.0e9),
        ),
        "SV": dict(
            entropy_range=entropy_range,
            volume_range=volume_range,
            pressure_range=(0.0, 100.0e9),
            temperature_range=(500.0, 3000.0),
        ),
    }
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = SEEDS
    settings.temperature_seeds = SEEDS
    settings.entropy_seeds = SEEDS
    settings.volume_seeds = SEEDS
    settings.max_phase_instances = 1
    settings.verbose = verbose
    results = {}
    for diagram, domain in domains.items():
        results[diagram] = bm.phase_diagram(
            BULK.atomic_composition,
            candidate_phases(),
            diagram=diagram,
            settings=settings,
            **domain,
        )
    return results


def calculate_contours(results, verbose=False):
    """Trace the two properties absent from each section's coordinate axes."""
    contours = {}
    for diagram, result in results.items():
        contours[diagram] = {
            name: bm.phase_diagram_contour_levels(
                result, candidate_phases(), levels, constraint, verbose=verbose
            )
            for name, (constraint, levels, _) in CONTOURS.items()
            if name not in diagram
        }
    return contours


def plot(
    results,
    output_dir,
    label_fontsize=9.0,
    *,
    contours=None,
    volume_unit="kg/m3",
    entropy_unit="kB/atom",
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colours = {
        ("per", "bdg", "aki"): "#bdd7ee",
        ("per", "aki", "rw"): "#cde6d5",
        ("per", "bdg", "rw"): "#edb4a9",
        ("per", "bdg", "aki", "rw"): "#cbb6e6",
        ("per", "bdg"): "#e5e5e5",
        ("per", "aki"): "#e5e5e5",
        ("rw",): "#e5e5e5",
    }
    for ax, (diagram, result) in zip(axes.flat, results.items()):
        bm.plot_phase_diagram(
            result,
            ax=ax,
            colorbar=False,
            fill_alpha=1.0,
            assemblage_colors=colours,
            pressure_unit="GPa",
            temperature_unit="K",
            entropy_unit=entropy_unit,
            volume_unit=volume_unit,
            swap_axes=diagram in ("TV", "SV"),
            label_fontsize=label_fontsize,
            label_key_path=output_dir / f"forsterite_{diagram.lower()}_labels.md",
        )
        handles = []
        for name, records in (contours or {}).get(diagram, {}).items():
            colour = CONTOURS[name][2]
            for i, record in enumerate(records):
                bm.plot_phase_diagram_contours(
                    result,
                    record["contours"],
                    ax,
                    color=colour,
                    line_style="dotted" if name in "ST" and i % 2 else "solid",
                    variable=name,
                    value=record["value"],
                    label_fontsize=label_fontsize,
                    label_placement="coexistence" if name == "P" else "auto",
                )
            legend_name = {
                "P": "Pressure",
                "T": "Temperature",
                "S": "Entropy",
                "V": "Density" if volume_unit == "kg/m3" else "Volume",
            }[name]
            handles.append(Line2D([], [], color=colour, label=legend_name))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=label_fontsize)
        ax.set_axisbelow(True)
        ax.grid(color="white", linewidth=1.0)
    return fig, axes


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = calculate(verbose=True)
    for diagram, result in results.items():
        write_phase_diagram(result, OUTPUT_DIR / f"forsterite_{diagram.lower()}")
    contours = calculate_contours(results, verbose=True)
    (OUTPUT_DIR / "forsterite_contours.json").write_text(
        json.dumps(contours, indent=2, allow_nan=False) + "\n"
    )
    fig, _ = plot(
        results,
        OUTPUT_DIR,
        contours=contours,
        volume_unit=VOLUME_UNIT,
        entropy_unit=ENTROPY_UNIT,
    )
    save_figure(fig, OUTPUT_DIR / "forsterite_sections.png", formats=("pdf",))
    print(f"Results saved to {OUTPUT_DIR}")
