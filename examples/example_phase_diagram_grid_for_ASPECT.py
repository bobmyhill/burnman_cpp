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
Calculate a P-S phase diagram and a 401x401 equilibrium grid
for dry pyrolite with initial Fe3+/total Fe of 2% on a metal-free basis
suitable for use in the geodynamics code ASPECT.

The composition is from Table 1 of Stixrude and Lithgow-Bertelloni (2024),
https://doi.org/10.1093/gji/ggae126.
The candidate phases include metallic iron but exclude melts and fluids.

The pyrolite diagram covers 0-40 GPa. Its entropy bounds are calculated from
stable equilibria at (0 GPa, 0 °C) and (140 GPa, 4000 °C).
Starting from 41x41 pressure-entropy seeds, the example traces and refines
phase boundaries. Resolved diagrams supply phase amounts and endmember
compositions on the regular grid.
An unresolved initial diagram is saved, then the example stops before refinement.

The saved equilibria supply total enthalpy H(P,S) and its relaxed derivatives.
Missing enthalpies and derivatives are interpolated, then Gaussian smoothing
with a standard deviation of one grid cell produces the enthalpy used for
an ASPECT table of specific entropy in J/(kg K) and pressure in bar.
The table contains temperature, density, thermal expansivity, specific heat
capacity, P- and S-wave velocities, and specific enthalpy. The velocities use
the unsmoothed assemblages with Voigt-Reuss-Hill averaging. Raw, infilled and
smoothed enthalpy grids are retained; invalid thermodynamic derivatives
produce diagnostic plots but prevent table export.
The pyrolite outputs are saved in phase_diagram_grid_for_ASPECT_output.

An alternative entry calculates quartz-coesite-stishovite equilibria for
1 mol of SiO2 on a 31x31 P-S grid. Its reference states are (0.5 GPa, 900 K)
and (15 GPa, 1500 K). The pyrolite entry is selected by default.
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2024, load_dataset, model_sets

EXAMPLES = Path(__file__).resolve().parent
PAPER_URL = "https://doi.org/10.1093/gji/ggae126"
MODEL_SET = model_sets.slb2024()
PHASES = MODEL_SET.phases
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


# The pyrolite diagram requires a large number of seeds
# because the phase boundaries are complex and the diagram
# covers a wide range of P and S.
def settings(seeds=(41, 41), verbose=False):
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds = seeds[0]
    opts.temperature_seeds = seeds[1]
    # Mask the diagram where the EoS of Mg-Tschermak breaks down.
    opts.required_eos_phases = ["opx"]
    # Mask the diagram where endmembers of cf have negative heat capacities.
    opts.required_thermal_phases = ["cf"]
    opts.verbose = verbose
    return opts


PHASE_DIAGRAMS = [
    dict(
        dataset="SLB_2024",
        composition=bm.Composition({"SiO2": 1.0}, "molar"),
        diagram="PS",
        lower_corner={"P": 0.5e9, "T": 900.0},
        upper_corner={"P": 15e9, "T": 1500.0},
        phase_names=["quartz", "coesite", "stishovite"],
        seeds=(5, 5),
        boundary_resolution=(101, 101),
        grid_resolution=(31, 31),
        plot_kwargs={"pressure_unit": "GPa", "phase_aliases": SLB_2024.phase_aliases},
        output_dir=EXAMPLES / "silica_phase_diagram_grid_output",
    ),
    dict(
        dataset=MODEL_SET,
        composition=PYROLITE_COMPOSITION,
        diagram="PS",
        lower_corner={"P": 0.0, "T": 273.15},
        upper_corner={"P": 140e9, "T": 4273.15},
        pressure_range=(0.0, 40e9),
        temperature_range=(273.15, 4273.15),
        settings=settings(),
        seeds=(41, 41),
        boundary_resolution=(401, 401),
        grid_resolution=(401, 401),
        plot_kwargs={
            "pressure_unit": "GPa",
            "entropy_unit": "kB/atom",
        },
        output_dir=EXAMPLES / "phase_diagram_grid_for_ASPECT_output",
    ),
]


if __name__ == "__main__":
    parameters = PHASE_DIAGRAMS[1]

    run_phase_diagram = True  # False reuses the saved equilibrium grid.
    if run_phase_diagram:
        run = bm.calculate_phase_diagram_grid(
            **parameters,
            # For diagrams only, set make_grid=False.
            make_grid=True,
            stop_on_unresolved=True,
            plot=True,
            plot_formats=["png", "svg"],  # A single string, e.g. "pdf", also works.
            verbose=True,
        )
        print(f"Refined diagram: {run.refined_diagram}")
        for path in run.diagram_plots:
            print(f"Phase diagram plot: {path}")
        key_path = parameters["output_dir"] / "phase_diagram_labels.md"
        if parameters["dataset"] is MODEL_SET and key_path.exists():
            with key_path.open("a") as key:
                key.write(
                    "\n## Phase abbreviations\n\n| Label | Model |\n| --- | --- |\n"
                )
                for label, name in PHASE_GLOSSARY.items():
                    key.write(f"| {label} | {name} |\n")
                key.write(
                    f"\nDataset and bulk source: [Stixrude & Lithgow-Bertelloni (2024)]({PAPER_URL}).\n"
                )
        if run.grid_manifest is None:
            print("Gridding skipped:", *run.diagnostics, sep="\n")
            raise SystemExit(0)
        grid_manifest, phases = run.grid_manifest, run.phases
    else:
        grid_manifest = parameters["output_dir"] / "grid_solutions" / "manifest.json"
        # Restore the original candidate order recorded alongside the diagram.
        saved = bm.load_phase_diagram(parameters["output_dir"] / "initial.json")
        models = saved["workflow"]["models"]
        if "phase_names" in models:
            dataset = models["dataset"].rsplit(".", 1)[-1]
            phases, _ = load_dataset(dataset, models["phase_names"])
        else:
            if models != MODEL_SET.to_dict():
                raise ValueError(
                    "Saved model set does not match the configured SLB2024 models."
                )
            phases = list(PHASES)

    print(f"Saved equilibria: {grid_manifest}")
    potentials = bm.calculate_potential_grid(
        grid_manifest,
        phases,
        output_dir=parameters["output_dir"] / "potential_grids",
        make_infilled_potential_grid=True,
        make_smoothed_potential_grid=True,
        smoothing_sigma=1.0,
        verbose=True,
    )
    print(f"Raw potential and derivatives: {potentials.raw_potential}")
    print(f"Infilled potential and derivatives: {potentials.infilled_potential}")
    print(f"Smoothed potential and derivatives: {potentials.smoothed_potential}")
    for diagnostic in potentials.diagnostics:
        print(diagnostic)

    if potentials.smoothed_potential is not None:
        export = bm.write_aspect_entropy_pressure_table(
            potentials.smoothed_potential,
            grid_manifest,
            phases,
            parameters["output_dir"] / "material_table_entropy_pressure.txt",
            averaging_scheme=bm.AveragingType.VRH,  # Or Voigt, Reuss, etc.
            plot=True,
            plot_formats=["png", "svg"],
        )
        print(f"ASPECT table: {export.table}")
        for path in export.plots:
            print(f"ASPECT properties: {path}")
        for diagnostic in export.diagnostics:
            print(diagnostic)

    # To repeat only potential processing, call calculate_potential_grid() with
    # the saved manifest and original phases, or a potential NPZ without models.
