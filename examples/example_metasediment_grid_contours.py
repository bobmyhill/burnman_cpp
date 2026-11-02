#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Composition contours interpolated from the saved metasediment grid.

Run example_metasediment_phase_diagram_grid.py first. Edit CONTOUR_SETS below
to choose phases, expressions and levels. This example accepts no command-line
arguments and performs no equilibrium calculations.
"""

from pathlib import Path
import json

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
        expression="Mgx_A / (Mgx_A + Fex_A)",
        levels=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        color="crimson",
        line_style="solid",
        label="g: Mg/(Mg+Fe²⁺)",
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
            "; excluding " + ", ".join(VOLUME_FRACTION_EXCLUDE_PHASES)
            if VOLUME_FRACTION_EXCLUDE_PHASES
            else ""
        ),
    ),
]


def main():
    diagram = bm.load_phase_diagram(SOURCE / "refined.json")
    phases = model_sets.metapelite().phases
    equilibria = bm.load_property_grid(SOURCE / "grid_equilibria.json.gz")
    figure, axes = bm.plot_phase_diagram(
        diagram,
        fill_alpha=0.15,
        phase_aliases={"g": "gt"},
        label_key_path=OUTPUT_DIR / "phase_diagram_labels.md",
    )
    for parameters in CONTOUR_SETS:
        grid = bm.phase_composition_grid(
            equilibria,
            phases,
            parameters["phase"],
            parameters["expression"],
            diagram=diagram,
            exclude_phases=parameters.get("exclude_phases", ()),
        )
        directory = OUTPUT_DIR / parameters["phase"]
        bm.save_property_grid(grid, directory / "composition.json.gz")
        contours = bm.phase_composition_contours(
            grid, parameters["levels"], diagram=diagram, phases=phases
        )
        (directory / "contours.json").write_text(
            json.dumps(contours, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for record in contours:
            bm.plot_phase_diagram_contours(
                diagram,
                record["contours"],
                axes,
                color=parameters["color"],
                line_style=parameters["line_style"],
                value=record["value"],
                label_format=parameters.get("label_format", ".2f"),
            )
    axes.set_title("Metasediment compositional contours")
    axes.legend(
        handles=[
            Line2D(
                [], [], color=p["color"], linestyle=p["line_style"], label=p["label"]
            )
            for p in CONTOUR_SETS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        fontsize=7,
    )
    save_figure(figure, OUTPUT_DIR / "contours.png", formats=("svg",))
    print(f"Saved composition grid and interpolated contours: {OUTPUT_DIR}")
    print("New equilibrium solves: 0")


if __name__ == "__main__":
    main()
