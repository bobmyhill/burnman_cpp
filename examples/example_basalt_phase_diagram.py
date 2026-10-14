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

"""Closed-system basalt phase diagram, 300–900 °C and 0–20 kbar.

All thermodynamics, Gibbs minimisation, phase selection, equilibrium solves
and boundary continuation run in C++. Python specifies the model/bulk and
writes JSON and a figure. No Python BurnMan or optimisation library is needed.

The matched HPx metabasite set reproduces the 30 January 2022 THERMOCALC
calibration: Green et al. (2016) solids and tonalitic melt, HP dataset 6.2,
and PS94 H2O with the Holland-Powell thermal reference. The model set is
available as burnman_cpp.minerals.model_sets.metabasite().
The closed bulk contains exactly 2 wt% H2O; no water/oxygen buffers are used.
A 1 bar lower calculation limit avoids the fluid singularity at P=0; the
figure uses the requested 0–20 kbar axis. Increase --seeds to check discovery
of narrow or disconnected fields. Failed calculations remain explicit.
"""

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import (
    example_parser,
    plot_phase_diagram_example,
    refine_saved,
    run_phase_diagram_example,
)

MODEL_SET = model_sets.metabasite()

# Masses in a 100 g wet rock; the sum is 100, including exactly 2 g of H2O.
BASALT_OXIDES = dict(
    SiO2=50.0,
    Al2O3=15.5,
    FeO=8.5,
    Fe2O3=1.5,
    MgO=7.5,
    CaO=10.5,
    Na2O=3.0,
    K2O=0.5,
    TiO2=1.0,
    H2O=2.0,
)
BASALT_COMPOSITION = bm.Composition(
    {oxide: grams * 1.0e-3 for oxide, grams in BASALT_OXIDES.items()}, unit_type="mass"
)


def candidate_phases():
    return model_sets.metabasite().phases


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
        BASALT_COMPOSITION.atomic_composition,
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
        composition=BASALT_COMPOSITION.atomic_composition,
        verbose=verbose,
        resolution=resolution,
    )


def save_json(result, path):
    bm.save_phase_diagram(
        result,
        path,
        metadata=dict(
            oxide_wt_percent=BASALT_OXIDES,
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
            title="Basalt + 2 wt% H₂O, closed system",
        ),
        scope="HPx metabasite 2022-01-30 + ds62 + PS94 H₂O",
        notes="2hb means two hornblende compositions; numbered fields: ",
    )


def main():
    parser = example_parser(
        __doc__,
        "basalt_phase_diagram_output",
        seeds=7,
        replot=True,
        refine=True,
        resolution=False,
    )
    return run_phase_diagram_example(
        parser,
        lambda args: calculate(args.seeds, args.quick, args.verbose),
        stem="basalt_phase_diagram",
        save_json=save_json,
        plot=plot,
        refine=refine,
    )


if __name__ == "__main__":
    main()
