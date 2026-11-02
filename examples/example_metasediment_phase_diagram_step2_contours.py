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

"""Step 2: trace density and garnet composition contours on a saved diagram.

Run example_metasediment_phase_diagram.py first. Edit the paths and contour
levels below, then run this script without arguments.

The saved phase lines, closed fields and equilibrium states are reused.
Native VolumeConstraint sets total volume to bulk mass / density.
Native PhaseCompositionConstraint selects garnet's Mg and ferrous-Fe sites;
its ferric-Fe site is excluded. Seeding, correction and continuation use C++.
These native constraint contours perform new solves. For contours from stored
equilibria only, use example_metasediment_phase_diagram_grid_step2_contours.py instead.
"""

import json
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.tools.phase_diagram_helpers import load_example, save_figure

EXAMPLES = Path(__file__).resolve().parent
SOURCE = EXAMPLES / "metasediment_phase_diagram_output/metasediment_phase_diagram.json"
OUTPUT_DIR = EXAMPLES / "metasediment_contours_output"
DENSITY_LEVELS = (2600.0, 2800.0, 3000.0, 3200.0)  # kg/m³.
GARNET_LEVELS = (0.05, 0.1, 0.15, 0.2, 0.3)  # Mg/(Mg+Fe2+).
SEED_GRID = 5
STEP = 0.02
SAVED_CONTOURS = None  # Set to a contour JSON path to replot without solving.

metasediment = load_example(
    Path(__file__).with_name("example_metasediment_phase_diagram.py")
)
candidate_phases = metasediment.candidate_phases


def calculate(diagram, density_levels, garnet_levels, seed_grid=5, step=0.02):
    phases = candidate_phases()
    settings = bm.PhaseDiagramContourSettings()
    settings.seed_grid = seed_grid
    settings.step = step
    composition = diagram["composition_start"]
    # The Composition object keeps density on the saved bulk's amount scale.
    mass = sum(bm.Composition(composition, "molar").mass_composition.values())
    # mp50NCKFMASHTO garnet's A sites contain Mg and Fe2+; its Fe3+ B site is excluded.
    constructors = {
        "density": (density_levels, lambda value: bm.VolumeConstraint(mass / value)),
        "garnet": (
            garnet_levels,
            lambda value: bm.PhaseCompositionConstraint.for_diagram(
                diagram, "g", ["Mgx_A", "Fex_A"], [1.0, 0.0], [1.0, 1.0], value
            ),
        ),
    }
    return {
        name: bm.phase_diagram_contour_levels(
            diagram,
            phases,
            levels,
            constructor,
            settings=settings,
            verbose=True,
        )
        for name, (levels, constructor) in constructors.items()
    }


def plot(diagram, contours, output_dir, label_fontsize=7.0):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(19, 9), layout="constrained")
    for ax, name in zip(axes, ["density", "garnet"]):
        bm.plot_phase_diagram(
            diagram,
            ax=ax,
            phase_aliases={"g": "gt"},
            fill_alpha=0.2,
            label_fontsize=label_fontsize,
            label_key_path=output_dir / f"{name}_assemblages.md",
        )
        for record in contours[name]:
            bm.plot_phase_diagram_contours(
                diagram,
                record["contours"],
                ax,
                color="crimson" if name == "density" else "royalblue",
                value=record["value"],
                label_fontsize=label_fontsize,
            )
        ax.set_title(
            "Bulk density (kg/m³)"
            if name == "density"
            else r"Garnet molar Mg / (Mg + Fe$^{2+}$)"
        )
        ax.set_xlim(300.0, 900.0)
        ax.set_ylim(0.0, 20.0)
    save_figure(
        fig, output_dir / "metasediment_contours.png", formats=("pdf",), dpi=180
    )


def main():
    diagram = bm.load_phase_diagram(SOURCE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contours = (
        json.loads(Path(SAVED_CONTOURS).read_text())
        if SAVED_CONTOURS is not None
        else calculate(diagram, DENSITY_LEVELS, GARNET_LEVELS, SEED_GRID, STEP)
    )
    (OUTPUT_DIR / "metasediment_contours.json").write_text(
        json.dumps(contours, indent=2, allow_nan=False) + "\n"
    )
    plot(diagram, contours, OUTPUT_DIR)


if __name__ == "__main__":
    main()
