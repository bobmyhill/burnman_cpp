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
The equilibria are saved on identical P-T grids. A second script
(example_pyrolite_fugacity_comparison_step2_contours.py) reads
the saved grids, calculates the oxygen fugacity difference, and plots contours
of the difference and the offset from the FMQ reference.
"""

from pathlib import Path
import json

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets

EXAMPLES = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLES / "pyrolite_fugacity_comparison_output"
PRESSURE_RANGE = (20.0e9, 25.0e9)  # Pa.
TEMPERATURE_RANGE = (1773.15, 2273.15)  # K: 1500-2000 C.
FERRIC_FRACTIONS = (0.03, 0.10)  # On a metal-free basis.
SEEDS = (7, 7)
BOUNDARY_RESOLUTION = (101, 101)
GRID_RESOLUTION = (101, 101)
PAPER_URL = "https://doi.org/10.1093/gji/ggae126"
MODEL_SET = model_sets.slb2024()
PHASE_GLOSSARY = MODEL_SET.phase_glossary

# Get the pyrolite composition from Table 1 of Stixrude & Lithgow-Bertelloni (2024).
# This corresponds to an Fe3+/sum(Fe) ratio of 2% on a metal-free basis.
# The composition is normalized to 100 g (0.1 kg).
PYROLITE_COMPOSITION = bm.Composition(
    dict(
        SiO2=45.00,
        MgO=38.88,
        FeO=8.05,
        Fe2O3=0.18,
        CaO=3.18,
        Al2O3=4.00,
        Na2O=0.13,
        Cr2O3=0.57,
    ),
    unit_type="mass",
)
PYROLITE_COMPOSITION.renormalize("mass", "total", 0.1)

PYROLITE_OXIDES = {
    oxide: kg * 1.0e3 for oxide, kg in PYROLITE_COMPOSITION.mass_composition.items()
}


def settings(seeds=SEEDS, verbose=False):
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds, opts.temperature_seeds = seeds
    # Mask the diagram where the EoS of Mg-Tschermak breaks down.
    opts.required_eos_phases = ["opx"]
    # Mask the diagram where endmembers of cf have negative heat capacities.
    opts.required_thermal_phases = ["cf"]
    opts.verbose = verbose
    return opts


def compositions():
    """
    Derive and return the two pyrolite compositions given the
    initial composition.
    """
    total_fe = PYROLITE_COMPOSITION.atomic_composition["Fe"]
    composition_dict = {}
    for ferric_fraction in FERRIC_FRACTIONS:
        if ferric_fraction < 0.0 or ferric_fraction > 1.0:
            raise ValueError(
                f"Initial Fe3+/sum(Fe) fraction {ferric_fraction} is outside [0,1]."
            )
        oxides = PYROLITE_COMPOSITION.molar_composition.copy()
        oxides["FeO"] = total_fe * (1.0 - ferric_fraction)
        oxides["Fe2O3"] = total_fe * ferric_fraction / 2.0
        reference = bm.Composition(oxides, unit_type="molar")
        reference.renormalize("mass", "total", 0.1)
        composition_dict[f"{ferric_fraction:.2f}"] = reference
    return composition_dict


if __name__ == "__main__":
    # Part 1: Complete both phase diagrams before gridding either composition.
    runs = {}
    for name, bulk in compositions().items():
        print(
            f"Calculating pyrolite with {100 * float(name):g}% initial ferric iron.",
            flush=True,
        )
        runs[name] = bm.calculate_phase_diagram_grid(
            dataset=MODEL_SET,
            composition=bulk.atomic_composition,
            diagram="PT",
            lower_corner={"P": PRESSURE_RANGE[0], "T": TEMPERATURE_RANGE[0]},
            upper_corner={"P": PRESSURE_RANGE[1], "T": TEMPERATURE_RANGE[1]},
            seeds=SEEDS,
            settings=settings(SEEDS, verbose=True),
            boundary_resolution=BOUNDARY_RESOLUTION,
            output_dir=OUTPUT_DIR / name,
            make_grid=False,
            plot=True,
            plot_formats=("png", "svg"),
            plot_kwargs=dict(
                pressure_unit="GPa",
                temperature_unit="C",
            ),
            verbose=True,
        )
        key_path = OUTPUT_DIR / name / "phase_diagram_labels.md"
        if key_path.exists():
            with key_path.open("a") as key:
                key.write(
                    "\n## Phase abbreviations\n\n| Label | Model |\n| --- | --- |\n"
                )
                for label, phase_name in PHASE_GLOSSARY.items():
                    key.write(f"| {label} | {phase_name} |\n")
                key.write(
                    f"\nDataset and bulk source: [Stixrude & Lithgow-Bertelloni (2024)]({PAPER_URL}).\n"
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
                reference_oxide_wt_percent=PYROLITE_OXIDES,
                dataset="SLB_2024",
                dataset_source=PAPER_URL,
                model_set=MODEL_SET.to_dict(),
                bulk_compositions={
                    name: bulk.atomic_composition
                    for name, bulk in compositions().items()
                },
                difference=f"{FERRIC_FRACTIONS[1]:.2f} minus {FERRIC_FRACTIONS[0]:.2f}",
            ),
            indent=2,
        )
        + "\n"
    )
