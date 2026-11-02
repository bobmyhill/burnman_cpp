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

"""Shaly metasediment at 5-10 kbar and 500-900 °C, in three parts.

First create, refine and save the phase diagram. Then save the equilibria on
an inclusive 101x101 P-T grid for later use. Finally, optionally label solution
branches in the completed diagram and save a separate plot. The closed bulk
contains 2 wt% H2O and uses the matched HPx metapelite model set, as in
example_metasediment_phase_diagram.py.

Run this script without command-line arguments; edit the settings below.
Results go to a new output directory beside this script. The equilibrium
data are saved without subsequent property or potential processing.
"""

from pathlib import Path
import json

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import save_figure

OUTPUT_DIR = (
    Path(__file__).resolve().with_name("metasediment_phase_diagram_grid_output")
)
PRESSURE_RANGE = (5.0e8, 1.0e9)  # Pa: 5-10 kbar.
TEMPERATURE_RANGE = (773.15, 1173.15)  # K: 500-900 °C.
GRID_RESOLUTION = (101, 101)
SECONDARY_LABELS = True  # Run the optional third step on the completed diagram.

# Masses in a 100 g wet rock, including exactly 2 g H2O.
METASEDIMENT_OXIDES = dict(
    SiO2=61.5,
    Al2O3=20.0,
    FeO=5.0,
    Fe2O3=1.0,
    MgO=2.5,
    CaO=1.0,
    Na2O=2.0,
    K2O=4.0,
    TiO2=1.0,
    H2O=2.0,
)
METASEDIMENT_COMPOSITION = bm.Composition(
    {oxide: grams * 1.0e-3 for oxide, grams in METASEDIMENT_OXIDES.items()},
    unit_type="mass",
)


def main():
    # Part 1: Phase diagram creation.
    run = bm.calculate_phase_diagram_grid(
        dataset=model_sets.metapelite(),
        composition=METASEDIMENT_COMPOSITION,
        diagram="PT",
        lower_corner={"P": PRESSURE_RANGE[0], "T": TEMPERATURE_RANGE[0]},
        upper_corner={"P": PRESSURE_RANGE[1], "T": TEMPERATURE_RANGE[1]},
        seeds=(7, 7),
        boundary_resolution=(101, 101),
        grid_resolution=GRID_RESOLUTION,
        output_dir=OUTPUT_DIR,
        make_grid=False,
        plot=True,
        plot_kwargs=dict(
            pressure_unit="kbar",
            temperature_unit="C",
            phase_aliases={"g": "gt"},
        ),
        verbose=True,
    )
    print(f"Refined phase diagram: {run.refined_diagram}")
    print(f"Phase diagram plot: {run.diagram_plot}")
    for diagnostic in run.diagnostics:
        print(diagnostic)

    # Part 2: Gridding at 101x101; save equilibrium data for later use.
    polygons = bm.phase_diagram_field_polygons(run.diagram)
    if not run.diagram.resolved or polygons.diagnostics:
        print("Gridding skipped: the phase diagram has unresolved fields.")
        return

    grid = bm.phase_diagram_grid(
        run.diagram,
        run.phases,
        # Store the temperature coordinate alongside the retained equilibria.
        properties=["temperature"],
        resolution=GRID_RESOLUTION,
        store_solutions=True,
    )
    path = bm.save_property_grid(grid, OUTPUT_DIR / "grid_equilibria.json.gz")
    print(f"Saved equilibria: {path}")
    print(
        f"Successful equilibria: {grid.equilibrium_success.sum()}/{grid.equilibrium_success.size}"
    )
    for diagnostic in grid.diagnostics:
        print(diagnostic)

    # Part 3: Optionally label solution branches in the completed phase diagram.
    if SECONDARY_LABELS:
        label_dir = OUTPUT_DIR / "secondary_labels"
        labelled = bm.label_phase_diagram_solutions(run.refined_diagram)
        path = bm.save_phase_diagram(labelled, label_dir / "labelled.json")
        figure, axes = bm.plot_phase_diagram(
            labelled,
            secondary_labels=True,
            phase_aliases={"g": "gt"},
            label_key_path=label_dir / "phase_diagram_labels.md",
        )
        axes.set_title("Metasediment: solution composition branches")
        save_figure(figure, label_dir / "phase_diagram.png", formats=("svg",))
        (label_dir / "secondary_labels.json").write_text(
            json.dumps(labelled["secondary_phase_labels"], indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"Annotated diagram and labelled plots: {path.parent}")


if __name__ == "__main__":
    main()
