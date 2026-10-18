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
"""Create, refine and grid silica and pyrolite diagrams using one public driver.

PHASE_DIAGRAMS holds the diagram-specific inputs; main() applies shared grid,
processing and plotting options to each entry. Edit the list to choose which
calculations to run. Results go to separate directories beside this script.
Only fully resolved diagrams are gridded.
After smoothing, export an ASPECT entropy-pressure table and property plots.
Invalid thermodynamic grids produce diagnostic plots but no table.

The pyrolite entry defines a 100 g, 3% ferric-iron composition locally and
uses the SLB2024 solid catalogue. Its entropy bounds come from P-T reference
states; the plotted pressure range is independently limited to 0-40 GPa.
See docs/phase_diagram_workflow.md for the four thermodynamic potentials and
subsequent infilling/smoothing.
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import SLB24

EXAMPLES = Path(__file__).resolve().parent


def pyrolite_composition(ferric_fraction=0.03):
    """SLB2024 Table 1 bulk, normalised to 100 g after redistributing iron.

    Retain the tabulated total Fe while setting molar Fe3+/total Fe to the
    requested fraction. Oxygen is closed; all mass amounts below are in kg.
    """
    if not 0.0 <= ferric_fraction <= 1.0:
        raise ValueError("ferric_fraction must be between zero and one.")
    iron = bm.Composition(dict(FeO=8.05e-3, Fe2O3=0.18e-3), "mass")
    total_fe = iron.atomic_composition["Fe"]
    composition = bm.Composition(
        dict(
            SiO2=45.0e-3,
            MgO=38.88e-3,
            CaO=3.18e-3,
            Al2O3=4.0e-3,
            Na2O=0.13e-3,
            Cr2O3=0.57e-3,
        ),
        "mass",
    )
    composition.add_components(
        dict(
            FeO=total_fe * (1.0 - ferric_fraction),
            Fe2O3=total_fe * ferric_fraction / 2.0,
        ),
        unit_type="molar",
    )
    composition.renormalize("mass", "total", 0.1)
    return composition


PYROLITE_SETTINGS = bm.PhaseDiagramSettings()
PYROLITE_SETTINGS.pressure_seeds = PYROLITE_SETTINGS.temperature_seeds = 41
# Mask the Mg-Tschermak EOS limit at low pressure and high temperature.
PYROLITE_SETTINGS.required_eos_phases = [SLB24.orthopyroxene().name]

PHASE_DIAGRAMS = [
    dict(
        dataset="SLB24",
        composition=bm.Composition({"SiO2": 1.0}, "molar"),
        diagram="PS",
        lower_corner={"P": 0.5e9, "T": 900.0},
        upper_corner={"P": 15e9, "T": 1500.0},
        phase_names=["quartz", "coesite", "stishovite"],
        seeds=(5, 5),
        boundary_resolution=(101, 101),
        grid_resolution=(31, 31),
        plot_kwargs={"pressure_unit": "GPa"},
        output_dir=EXAMPLES / "silica_phase_diagram_grid_output",
    ),
    dict(
        dataset="SLB24",
        composition=pyrolite_composition(),
        diagram="PS",
        lower_corner={"P": 0.0, "T": 273.15},
        upper_corner={"P": 140e9, "T": 4273.15},
        pressure_range=(0.0, 40e9),
        temperature_range=(273.15, 4273.15),
        settings=PYROLITE_SETTINGS,
        seeds=(41, 41),
        boundary_resolution=(401, 401),
        grid_resolution=(401, 401),
        plot_kwargs={"pressure_unit": "GPa", "entropy_unit": "kB/atom"},
        output_dir=EXAMPLES / "pyrolite_sp_phase_diagram_grid_output",
    ),
]


def main():
    parameters = PHASE_DIAGRAMS[1]
    run = bm.calculate_phase_diagram_grid(
        **parameters,
        # For diagrams only, disable all four grid/processing stages.
        make_grid=True,
        make_potential_grid=True,
        make_infilled_potential_grid=True,
        make_smoothed_potential_grid=True,
        smoothing_sigma=2.0,
        plot=True,
        plot_formats=["png", "svg"],  # A single string, e.g. "pdf", also works.
        verbose=True,
    )
    print(f"Refined diagram: {run.refined_diagram}")
    for path in run.diagram_plots:
        print(f"Phase diagram plot: {path}")
    if run.grid_manifest is None:
        print("Gridding skipped:", *run.diagnostics, sep="\n")

    print(f"Saved equilibria: {run.grid_manifest}")
    print(f"Raw potential and derivatives: {run.raw_potential}")
    print(f"Infilled potential and derivatives: {run.infilled_potential}")
    print(f"Smoothed potential and derivatives: {run.smoothed_potential}")

    if run.smoothed_potential is not None:
        export = bm.write_aspect_entropy_pressure_table(
            run.smoothed_potential,
            run.grid_manifest,
            run.phases,
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

    # Optional postprocessing; saved equilibria stay unchanged:
    # grids = bm.load_potential_grid(run.grid_manifest, run.phases)
    # grids, info = bm.infill_potential_grid(grids, fill_derivatives=True, return_info=True)
    # smoothed_grids = bm.smooth_potential_grid(grids, sigma=2.)


if __name__ == "__main__":
    main()
