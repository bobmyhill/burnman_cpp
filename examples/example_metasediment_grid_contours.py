#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""
Composition and property contours interpolated from a saved thermodynamic grid.

Run example_metasediment_phase_diagram_grid.py first. Edit CONTOUR_SETS below
to choose phases, expressions, scalar attribute properties and levels.
This example performs no equilibrium calculations.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import save_figure

SOURCE = Path(__file__).resolve().with_name("metasediment_phase_diagram_grid_output")
OUTPUT_DIR = SOURCE / "composition_contours"
# Freeze saved amounts and compositions, evaluating volumes at 1 bar, 298.15 K.
STANDARD_STATE_VOLUME_FRACTIONS = True
VOLUME_FRACTION_EXCLUDE_PHASES = ("H2O", "melt")  # Use () to include every phase.
CONTOUR_SETS = [
    dict(
        phase="g",
        expression="Mgx_A / (Mgx_A + Fex_A + Cax_A)",
        levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        color="crimson",
        line_style="solid",
        label="g: Mg/(Mg+Fe²⁺+Ca), X site",
    ),
    dict(
        phase="fsp_1",
        expression="Ka_A / (Ka_A + Naa_A + Caa_A)",
        levels=(0.65, 0.675, 0.70, 0.725, 0.75),
        label_format=".3f",
        color="royalblue",
        line_style="dashed",
        label="fsp_1: K/(K+Na+Ca)",
    ),
    dict(
        phase="fsp_2",
        expression="Ka_A / (Ka_A + Naa_A + Caa_A)",
        levels=(0.01, 0.02, 0.03, 0.04, 0.05),
        label_format=".3f",
        color="orange",
        line_style="dashed",
        label="fsp_2: K/(K+Na+Ca)",
    ),
    dict(
        phase="bi",
        expression=(
            "standard_volume_fraction"
            if STANDARD_STATE_VOLUME_FRACTIONS
            else "volume_fraction"
        ),
        levels=(0.05, 0.10, 0.15, 0.20, 0.25),
        exclude_phases=VOLUME_FRACTION_EXCLUDE_PHASES,
        color="forestgreen",
        line_style="dotted",
        label="bi: volume fraction"
        + (" (standard state)" if STANDARD_STATE_VOLUME_FRACTIONS else "")
        + (
            "; excluding " + "+ ".join(VOLUME_FRACTION_EXCLUDE_PHASES)
            if VOLUME_FRACTION_EXCLUDE_PHASES
            else ""
        ),
    ),
    dict(
        name="density",
        property="density / 1000",  # kg/m³ to g/cm³, at the saved P–T.
        include_phases=None,  # E.g. ("g", "bi") selects only garnet and biotite.
        exclude_phases=("H2O", "melt"),
        levels=(2.8, 2.9, 3.0, 3.1),
        label_format=".1f",
        color="purple",
        line_style="dashdot",
        label="Solid density (g/cm³)",
    ),
    dict(
        name="melt_density",
        property="density / 1000",
        include_phases=("melt",),  # Melt only; the H2O fluid is not included.
        levels=(2.10, 2.15, 2.20, 2.25, 2.30),
        label_format=".2f",
        color="teal",
        line_style="dashdot",
        label="Melt density (g/cm³)",
    ),
]


def main():
    diagram = bm.load_phase_diagram(SOURCE / "refined.json")
    phases = model_sets.metapelite().phases
    equilibria = bm.load_property_grid(SOURCE / "grid_equilibria.json.gz")
    figure, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.055, right=0.98, bottom=0.18, top=0.94, wspace=0.12)
    panels = (
        ("Compositional contours", [p for p in CONTOUR_SETS if "property" not in p]),
        ("Density contours", [p for p in CONTOUR_SETS if "property" in p]),
    )
    for ax, (title, contour_sets) in zip(axes, panels):
        bm.plot_phase_diagram(
            diagram,
            ax,
            fill_alpha=0.15,
            phase_aliases={"g": "gt"},
            label_key_path=OUTPUT_DIR / "phase_diagram_labels.md",
        )
        for parameters in contour_sets:
            if "property" in parameters:
                grid = bm.phase_diagram_property_grid(
                    equilibria,
                    phases,
                    parameters["property"],
                    phase=parameters.get("phase"),
                    include_phases=parameters.get("include_phases"),
                    exclude_phases=parameters.get("exclude_phases", ()),
                    diagram=diagram,
                )
                name = parameters.get("name") or "_".join(
                    p for p in (parameters.get("phase"), parameters["property"]) if p
                )
                filename = "property.json.gz"
            else:
                grid = bm.phase_composition_grid(
                    equilibria,
                    phases,
                    parameters["phase"],
                    parameters["expression"],
                    diagram=diagram,
                    exclude_phases=parameters.get("exclude_phases", ()),
                )
                name, filename = parameters["phase"], "composition.json.gz"
            directory = OUTPUT_DIR / name
            bm.save_property_grid(grid, directory / filename)
            contours = bm.phase_diagram_grid_contours(
                grid, parameters["levels"], diagram=diagram, phases=phases
            )
            (directory / "contours.json").write_text(
                json.dumps(contours, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            for record in contours:
                bm.plot_phase_diagram_contours(
                    diagram,
                    record["contours"],
                    ax,
                    color=parameters["color"],
                    line_style=parameters["line_style"],
                    value=record["value"],
                    label_format=parameters.get("label_format", ".2f"),
                )
        ax.set_title(title)
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=p["color"],
                    linestyle=p["line_style"],
                    label=p["label"],
                )
                for p in contour_sets
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            fontsize=7,
        )
    axes[1].set_ylabel("")
    save_figure(
        figure, OUTPUT_DIR / "contours.png", formats=("svg",), bbox_inches="tight"
    )
    print(f"Saved scalar grids and interpolated contours: {OUTPUT_DIR}")
    print("New equilibrium solves: 0")


if __name__ == "__main__":
    main()
