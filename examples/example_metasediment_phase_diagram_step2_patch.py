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
Sometimes, you might find that the original seeds for a phase diagram
were spaced too far apart for refinement to find small or narrow fields.
One solution to this problem would be to recalculate the entire diagram
with more seeds, but that can be expensive and time-consuming.

This example shows an alternative approach to this problem:
recalculating phase boundaries within a small window of the original diagram,
then merging the results into the original diagram.

The example uses the metasediment phase diagram from
example_metasediment_phase_diagram.py. After running that script,
this example recalculates phase boundaries for the region
873.15-973.15 K and 8.0e8-1.2e9 Pa using a 9x9 seed grid,
then merges the patch into the original diagram.
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import plot_phase_diagram_example

EXAMPLES = Path(__file__).resolve().parent
SOURCE = EXAMPLES / "metasediment_phase_diagram_output/metasediment_phase_diagram.json"
OUTPUT_DIR = EXAMPLES / "patched_phase_diagram_output"
MODEL_SET = model_sets.metapelite()
REGION = ((873.15, 973.15), (8.0e8, 1.2e9))  # x: temperature (K), y: pressure (Pa).
SEEDS = 9


if __name__ == "__main__":
    source = bm.load_phase_diagram(SOURCE)
    if source.get("model_set") != MODEL_SET.to_dict():
        raise ValueError("Use the same metapelite model set as the saved diagram.")
    phases = MODEL_SET.phases
    settings = bm.PhaseDiagramSettings.from_dict(source["settings"])
    settings.pressure_seeds = settings.temperature_seeds = SEEDS
    settings.verbose = True

    # There are a number of ways to select a region for patching.
    # The simplest, demonstrated below, is to specify a rectangular window
    # in temperature and pressure.
    patch = bm.calculate_phase_diagram_patch(
        source, phases, REGION, settings, temperature_unit="K", pressure_unit="Pa"
    )
    # One can also specify a region using other methods.
    # For example, you can select a field by its one-based number
    # on the plot/assemblage key, for example:
    # windows = bm.phase_diagram_field_regions(
    #     source, (1,), padding=0.05, temperature_unit="K", pressure_unit="Pa"
    # )
    # One can also find problem regions in the original diagram
    # based on their saved diagnostics, for example:
    # windows = bm.phase_diagram_problem_regions(
    #     source, padding=0.05, temperature_unit="K", pressure_unit="Pa"
    # )

    # The patch can be saved as usual..
    bm.save_phase_diagram(patch, OUTPUT_DIR / "patch.json", metadata=source)

    # Patches can be merged, one at a time, into the original diagram.
    merged, report = bm.merge_phase_diagrams(source, patch, phases=phases)

    # The merged diagram can also be saved and plotted as usual.
    bm.save_phase_diagram(merged, OUTPUT_DIR / "merged.json")
    plot_phase_diagram_example(
        merged,
        OUTPUT_DIR / "merged.png",
        phase_aliases={"g": "gt"},
        temperature_unit="K",
        pressure_unit="Pa",
    )
    (OUTPUT_DIR / "merge_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(*report, sep="\n")
