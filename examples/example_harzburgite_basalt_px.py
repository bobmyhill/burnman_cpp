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
This example calculates the pressure-composition phase diagram
for a chemically equilibrated mixture of harzburgite and basalt.
It uses the SLB2024 solid solution model at 1673 K
and pressures from 0 to 25 GPa.

The dry compositions are taken from Table 1 of
Stixrude & Lithgow-Bertelloni (2024), doi:10.1093/gji/ggae126.
X is the basalt mass fraction: bulk(X)=(1-X)*harzburgite+X*basalt.
The compositions always sum to 100 g.
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import (
    plot_phase_diagram_example,
    write_phase_diagram,
)

OUTPUT_DIR = Path(__file__).resolve().with_name("harzburgite_basalt_px_output")
SEEDS = 7
PAPER_URL = "https://doi.org/10.1093/gji/ggae126"


def endmember(oxides):
    composition = bm.Composition(oxides, unit_type="mass")
    composition.renormalize("mass", "total", 0.1)
    return composition


HARZBURGITE = endmember(
    dict(
        SiO2=43.43,
        MgO=45.93,
        FeO=8.34,
        Fe2O3=0.09,
        CaO=0.90,
        Al2O3=1.00,
        Na2O=0.01,
        Cr2O3=0.30,
    )
)
BASALT = endmember(
    dict(
        SiO2=50.42,
        MgO=9.77,
        FeO=7.10,
        Fe2O3=1.07,
        CaO=12.54,
        Al2O3=16.80,
        Na2O=2.23,
        Cr2O3=0.07,
    )
)


def calculate(seeds=7, verbose=False):
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds = opts.temperature_seeds = seeds
    opts.composition_seeds = opts.pressure_seeds
    # Mask undefined Mg-Tschermak EoS states and the cf ordering transition.
    opts.required_eos_phases = ["opx"]
    opts.required_thermal_phases = ["cf"]
    opts.verbose = verbose
    return bm.phase_diagram(
        HARZBURGITE.atomic_composition,
        model_sets.slb2024().phases,
        diagram="PX",
        composition_end=BASALT.atomic_composition,
        pressure_range=(0.0, 25.0e9),
        temperature=1673.0,
        settings=opts,
    )


def save_json(result, path):
    bm.save_phase_diagram(
        result,
        path,
        metadata=dict(
            endpoint_masses_kg={
                "harzburgite": HARZBURGITE.mass_composition,
                "basalt": BASALT.mass_composition,
            },
            composition_source=PAPER_URL,
            closed_system=True,
        ),
    )


def plot(result, path, label_fontsize=8.0):
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        formats=("pdf",),
        pressure_unit="GPa",
        composition_label="Basalt mass fraction",
        axes=dict(title="Harzburgite-basalt at 1673 K (SLB2024 solids)"),
        key_path=path.with_name(path.stem + "_labels.md"),
    )


if __name__ == "__main__":
    write_phase_diagram(
        calculate(seeds=SEEDS, verbose=True),
        OUTPUT_DIR / "harzburgite_basalt_px",
        save_json=save_json,
        plot=plot,
        label_fontsize=8.0,
    )
