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

"""Closed-system shaly metasediment, 300–900 °C and 0–20 kbar.

All thermodynamics, Gibbs minimisation, phase selection, equilibrium solves
and boundary continuation run in C++. Python specifies the model/bulk and
writes JSON and figures. No Python BurnMan or optimisation library is needed.

The alumina-rich shale contains exactly 2 wt% H2O in a 100 g wet
bulk. FeO and Fe2O3 specify its closed oxygen inventory; water and oxygen are
not buffered. The matched HPx metapelite set reproduces the 23 January 2022
THERMOCALC calibration: White et al. (2014) solids and granitic melt, HP
dataset 6.2, and PS94 H2O with the Holland-Powell thermal reference. It is
available as burnman_cpp.minerals.model_sets.metapelite().

One ternary feldspar model represents plagioclase and alkali feldspar by
coexisting solution instances; fsp means feldspar and 2fsp means two
compositions. ms denotes the white-mica solution (including Na/Ca-bearing
endmembers); 2ms likewise denotes two compositions. Do not add k4tr as a
second candidate: it duplicates pl4tr's thermodynamic model.

Calculations start at 1 bar to avoid the fluid singularity at zero pressure;
figures display 0–20 kbar. Increase --seeds to check narrow/disconnected fields.
--refine-json resumes unfinished lines from accepted native equilibrium states.
With --refine-json, --resolution NX NY also subdivides phase lines using verified
equilibria; finer chords can resolve fields narrower than the saved line spacing.
"""

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import (
    example_parser,
    plot_phase_diagram_example,
    refine_saved,
    run_phase_diagram_example,
)

MODEL_SET = model_sets.metapelite()

# Masses in a 100 g wet rock; the sum is 100, including exactly 2 g of H2O.
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


def candidate_phases():
    return model_sets.metapelite().phases


def calculate(
    seeds=7,
    quick=False,
    verbose=False,
    *,
    pressure_range=(1.0e5, 2.0e9),
    temperature_range=(573.15, 1173.15),
):
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = 4 if quick else seeds
    settings.step = 0.04 if quick else 0.025
    settings.verbose = verbose
    return bm.phase_diagram(
        METASEDIMENT_COMPOSITION.atomic_composition,
        candidate_phases(),
        pressure_range,
        temperature_range,
        settings,
    )


def refine(source, verbose=False, resolution=None):
    """Resume saved phase lines and optionally refine their point spacing."""
    return refine_saved(
        source,
        candidate_phases(),
        composition=METASEDIMENT_COMPOSITION.atomic_composition,
        verbose=verbose,
        resolution=resolution,
    )


def save_json(result, path):
    bm.save_phase_diagram(
        result,
        path,
        metadata=dict(
            oxide_wt_percent=METASEDIMENT_OXIDES,
            closed_system=True,
            model_set=MODEL_SET.to_dict(),
        ),
    )


def plot(
    result,
    path,
    label_fontsize=7.0,
    *,
    temperature_range=(300, 900),
    pressure_range=(0, 20),
):
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        phase_aliases={"g": "gt"},
        axes=dict(
            xlim=temperature_range,
            ylim=pressure_range,
            xlabel="Temperature (°C)",
            ylabel="Pressure (kbar)",
            title="Shaly metasediment + 2 wt% H₂O, closed system",
        ),
        scope="HPx metapelite 2022-01-23 + ds62 + PS94 H₂O",
        notes="ms: white mica; fsp: feldspar; 2fsp means two compositions; numbered fields: ",
    )


def main():
    parser = example_parser(
        __doc__,
        "metasediment_phase_diagram_output",
        seeds=7,
        replot=True,
        refine=True,
        resolution=True,
    )
    return run_phase_diagram_example(
        parser,
        lambda args: calculate(args.seeds, args.quick, args.verbose),
        stem="metasediment_phase_diagram",
        save_json=save_json,
        plot=plot,
        refine=refine,
    )


if __name__ == "__main__":
    main()
