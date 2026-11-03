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
Olivine-polymorph P-X, T-X and V-X diagrams,
covering compositions from Mg2SiO4 to Fe2SiO4.
X is the molar iron fraction Fe/(Mg+Fe).

The candidate set of phases comprises olivine,
wadsleyite and ringwoodite from SLB_2011.

The P-X diagram spans 0-20 GPa at 1673 K.
The T-X diagram spans 1000-3000 K at 14 GPa.
The V-X diagram spans 36-45 cm3 at 1673 K.

Edit DIAGRAMS below to select which diagrams to calculate.
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011
from burnman_cpp.tools.phase_diagram_helpers import (
    plot_phase_diagram_example,
    write_phase_diagram,
)

OUTPUT_ROOT = Path(__file__).resolve().parent
DIAGRAMS = ("PX", "TX", "VX")
SEEDS = 5

FORSTERITE = bm.Composition({"Mg2SiO4": 1.0}, "molar")
FAYALITE = bm.Composition({"Fe2SiO4": 1.0}, "molar")


def candidate_phases():
    phases = [
        SLB_2011.mg_fe_olivine(),
        SLB_2011.mg_fe_wadsleyite(),
        SLB_2011.mg_fe_ringwoodite(),
    ]
    for phase, name in zip(phases, ["ol", "wa", "ri"]):
        phase.set_name(name)
    return phases


def calculate(quick=False, verbose=False, *, diagram="PX"):
    domains = {
        "PX": dict(pressure_range=(0.0, 20.0e9), temperature=1673.0),
        "TX": dict(temperature_range=(1000.0, 3000.0), pressure=14.0e9),
        "VX": dict(volume_range=(36.0e-6, 45.0e-6), temperature=1673.0),
    }
    if diagram not in domains:
        raise ValueError("The olivine example supports PX, TX or VX diagrams.")
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = SEEDS
    settings.temperature_seeds = SEEDS
    settings.volume_seeds = SEEDS
    settings.composition_seeds = SEEDS
    settings.max_phase_instances = 1
    settings.verbose = verbose
    return bm.phase_diagram(
        FORSTERITE.atomic_composition,
        candidate_phases(),
        diagram=diagram,
        composition_end=FAYALITE.atomic_composition,
        settings=settings,
        **domains[diagram],
    )


def plot(result, path, label_fontsize=9.0, *, diagram="PX"):
    condition = "14 GPa" if diagram == "TX" else "1673 K"
    axes = dict(title=f"Olivine polymorphs at {condition}")
    if diagram == "VX":
        axes["ylabel"] = "Volume (cm³ per mole of bulk formula units)"
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        axes=axes,
        formats=("pdf",),
        pressure_unit="GPa",
        temperature_unit="K",
        volume_unit="cm3",
        composition_label=r"$X_{\mathrm{Fe}} = \mathrm{Fe}/(\mathrm{Mg}+\mathrm{Fe})$",
        phase_aliases={"ol": "ol", "wa": "wa", "ri": "ri"},
        show_nodes=True,
        key_path=path.with_name(path.stem + "_labels.md"),
    )


if __name__ == "__main__":
    for diagram in DIAGRAMS:
        stem = f"olivine_{diagram.lower()}"
        result = calculate(verbose=True, diagram=diagram)
        output = OUTPUT_ROOT / f"{stem}_output"
        write_phase_diagram(result, output / stem)
        plot(result, output / f"{stem}.png", diagram=diagram)
