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

"""Iron-hematite T-X diagram at 1 bar, from Fe to Fe2O3, 300 to 1500 K.

Uses HGP18 iron, wuestite, magnetite and hematite with the Holland
and Powell EOS. X mixes one mole Fe with one mole Fe2O3: Fe=1+X, O=3X.
The system is closed to oxygen. All equilibrium and tracing run in C++.
These are stoichiometric solids; iron allotropes, nonstoichiometric FeO and
liquids are outside this example's candidate set.

Run: python examples/example_iron_hematite_tx.py
Use --quick for fewer seeds, or --no-plots to save only the resumable JSON.
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import HGP18
from burnman_cpp.tools.phase_diagram_helpers import (
    example_parser,
    plot_phase_diagram_example,
    run_phase_diagram_example,
)

IRON = bm.Composition({"Fe": 1.0}, "molar")
HEMATITE = bm.Composition({"Fe2O3": 1.0}, "molar")


def candidate_phases():
    phases = [
        HGP18.iron(),
        HGP18.wu(),
        HGP18.mt(),
        HGP18.hem(),
    ]
    for phase, name in zip(phases, ["Fe", "FeO", "Fe3O4", "Fe2O3"]):
        phase.set_name(name)
    return phases


def calculate(quick=False, verbose=False):
    domain = dict(temperature_range=(300.0, 1500.0), pressure=1.0e5)
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = 5 if quick else 9
    settings.composition_seeds = 5 if quick else 17
    settings.max_phase_instances = 1
    settings.verbose = verbose
    return bm.phase_diagram(
        IRON.atomic_composition,
        candidate_phases(),
        diagram="TX",
        composition_end=HEMATITE.atomic_composition,
        settings=settings,
        **domain,
    )


def plot(result, path, label_fontsize=9.0):
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        formats=("pdf",),
        temperature_unit="K",
        composition_label="Molar hematite fraction in Fe–Fe$_2$O$_3$ bulk",
        axes=dict(title="Iron-Hematite at 1 bar"),
        show_nodes=False,
        key_path=path.with_name(path.stem + "_labels.md"),
    )


def main():
    parser = example_parser(
        "Iron-Hematite T-X diagram at 1 bar.",
        Path(__file__).resolve().parent / "Fe-O_tx_output",
        label_fontsize=9.0,
    )
    return run_phase_diagram_example(
        parser,
        lambda args: calculate(args.quick, args.verbose),
        stem="Fe-O_tx",
        plot=plot,
    )


if __name__ == "__main__":
    main()
