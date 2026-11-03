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
Compare the fO2 of pyrolites with initial Fe3+/total Fe of 3% and 10%
(on a metal-free basis). This script uses the dataset presented in
Stixrude and Lithgow-Bertelloni (2024), https://doi.org/10.1093/gji/ggae126,
which includes metallic iron but excludes melts or fluids.

This script calculates the phase diagrams at 20-25 GPa and 1500-2000 C.
The equilibria are saved on identical P-T grids. The
second script (example_pyrolite_fugacity_comparison_step2_contours.py) reads
the saved grids, calculates the oxygen fugacity difference, and plots contours
of the difference and the offset from the FMQ reference.
"""

from pathlib import Path
import json

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2024
from burnman_cpp.tools.phase_diagram_helpers import load_example

EXAMPLES = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLES / "pyrolite_fugacity_comparison_output"
PRESSURE_RANGE = (20.0e9, 25.0e9)  # Pa.
TEMPERATURE_RANGE = (1773.15, 2273.15)  # K: 1500-2000 C.
FERRIC_FRACTIONS = {"fe3_03": 0.03, "fe3_10": 0.10}
SEEDS = (7, 7)
BOUNDARY_RESOLUTION = (101, 101)
GRID_RESOLUTION = (101, 101)

pyrolite = load_example(EXAMPLES / "example_pyrolite_phase_diagram.py")


def compositions():
    """
    Change only oxygen, retaining the reference pyrolite's cation amounts.
    """
    reference = pyrolite.pyrolite_composition(0.03).atomic_composition
    return {
        name: reference
        | {"O": reference["O"] + 0.5 * reference["Fe"] * (fraction - 0.03)}
        for name, fraction in FERRIC_FRACTIONS.items()
    }


def main():
    # Part 1: Complete both phase diagrams before gridding either composition.
    runs = {}
    for name, bulk in compositions().items():
        print(
            f"Calculating pyrolite with {100 * FERRIC_FRACTIONS[name]:g}% initial ferric iron.",
            flush=True,
        )
        runs[name] = bm.calculate_phase_diagram_grid(
            dataset=SLB_2024,
            composition=bulk,
            diagram="PT",
            lower_corner={"P": PRESSURE_RANGE[0], "T": TEMPERATURE_RANGE[0]},
            upper_corner={"P": PRESSURE_RANGE[1], "T": TEMPERATURE_RANGE[1]},
            seeds=SEEDS,
            boundary_resolution=BOUNDARY_RESOLUTION,
            output_dir=OUTPUT_DIR / name,
            make_grid=False,
            plot=True,
            plot_formats=("png", "svg"),
            plot_kwargs=dict(
                pressure_unit="GPa",
                temperature_unit="C",
                phase_aliases=SLB_2024.phase_aliases,
            ),
            verbose=True,
        )
    if any(not run.diagram.resolved for run in runs.values()):
        raise RuntimeError("Resolve the saved diagram diagnostics before gridding.")

    # Part 2: Retain equilibrium states; fugacity processing uses existing solves.
    for name, run in runs.items():
        grid = bm.phase_diagram_grid(
            run.diagram,
            run.phases,
            "temperature",
            resolution=GRID_RESOLUTION,
            relaxed=False,
            store_solutions=True,
        )
        path = bm.save_property_grid(
            grid, OUTPUT_DIR / name / "grid_equilibria.json.gz"
        )
        print(
            f"{name}: {grid.equilibrium_success.sum()}/{grid.equilibrium_success.size} equilibria saved to {path}",
            flush=True,
        )
        for message in grid.diagnostics:
            print(message, flush=True)
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(
            dict(
                initial_ferric_fractions=FERRIC_FRACTIONS,
                bulk_compositions=compositions(),
                difference="fe3_10 minus fe3_03",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
