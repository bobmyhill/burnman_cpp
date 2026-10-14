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

"""Olivine-polymorph P-X diagram at 1673 K, from Mg2SiO4 to Fe2SiO4.

The candidate set comprises olivine, wadsleyite and ringwoodite from SLB11.
X is the molar fayalite fraction Fe/(Mg+Fe). All stability searches and phase-
boundary continuation run in C++; plotting uses the standard phase diagram API.

Run: python examples/example_olivine_px.py
Use --quick for fewer seeds, or --no-plots to save only the resumable JSON.
"""

from functools import partial
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import SLB11
from burnman_cpp.tools.phase_diagram_helpers import (
    example_parser,
    plot_phase_diagram_example,
    run_phase_diagram_example,
)

FORSTERITE = bm.Composition({"Mg2SiO4": 1.0}, "molar")
FAYALITE = bm.Composition({"Fe2SiO4": 1.0}, "molar")


def candidate_phases():
    phases = [
        SLB11.mg_fe_olivine(),
        SLB11.mg_fe_wadsleyite(),
        SLB11.mg_fe_ringwoodite(),
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
    settings.pressure_seeds = settings.temperature_seeds = 5 if quick else 9
    settings.volume_seeds = settings.pressure_seeds
    settings.composition_seeds = 5 if quick else 17
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


def main(diagram="PX"):
    stem = f"olivine_{diagram.lower()}"
    condition = "14 GPa" if diagram == "TX" else "1673 K"
    parser = example_parser(
        f"Olivine-polymorph {diagram[0]}-X diagram at {condition}.",
        Path(__file__).resolve().parent / f"{stem}_output",
        label_fontsize=9.0,
    )
    return run_phase_diagram_example(
        parser,
        lambda args: calculate(args.quick, args.verbose, diagram=diagram),
        stem=stem,
        plot=partial(plot, diagram=diagram),
    )


if __name__ == "__main__":
    main()
