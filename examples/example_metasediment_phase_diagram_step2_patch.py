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

"""Step 2: calculate local patches and merge them into a saved metasediment diagram.

Run example_metasediment_phase_diagram.py first. Edit the settings below and
run without arguments. FIELDS selects one-based field numbers; None selects
problematic fields and boundaries from the saved diagnostics. PATCH_FILES
can instead select existing patch JSON files, applied in the given order.
Set MERGE=False to save calculated patches for inspection before merging.
"""

from functools import partial
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.tools.phase_diagram_helpers import (
    load_example,
    plot_phase_diagram_example,
    write_phase_diagram,
)

EXAMPLES = Path(__file__).resolve().parent
SOURCE = EXAMPLES / "metasediment_phase_diagram_output/metasediment_phase_diagram.json"
MODELS = EXAMPLES / "example_metasediment_phase_diagram.py"
OUTPUT_DIR = EXAMPLES / "patched_phase_diagram_output"
FIELDS = None  # E.g. (1, 3) selects Field 1 and Field 3.
SEEDS = (9, 9)  # x/y: temperature/pressure for a P-T diagram.
PADDING = 0.05  # Expand each patch window by 5% per side.
PATCH_FILES = ()  # Existing patches, or () to calculate new ones.
MERGE = True
PLOT = True
VERBOSE = True


def candidate_phases(module_name):
    """Accept an import name or a Python file, including sibling imports."""
    return load_example(module_name).candidate_phases()


def save(result, directory, plots, metadata=None):
    data = result if isinstance(result, dict) else result.to_dict()
    data = dict(metadata or {}) | data
    return write_phase_diagram(
        data,
        directory / "phase_diagram",
        plot=(
            partial(
                plot_phase_diagram_example,
                key_path=directory / "assemblages.md",
                bbox_inches=None,
            )
            if plots
            else None
        ),
    )


def main():
    source = bm.load_phase_diagram(SOURCE)
    phases = candidate_phases(MODELS) if MODELS is not None else None
    if PATCH_FILES:
        patches = [(str(path), bm.load_phase_diagram(path)) for path in PATCH_FILES]
    else:
        windows = (
            {
                f"Field {number}": region
                for number, region in bm.phase_diagram_field_regions(
                    source, FIELDS, padding=PADDING
                ).items()
            }
            if FIELDS is not None
            else bm.phase_diagram_problem_regions(source, padding=PADDING)
        )
        if not windows:
            print(
                "No problematic fields or boundaries found; no patches were calculated."
            )
            return
        settings = bm.PhaseDiagramSettings.from_dict(source.get("settings", {}))
        names = dict(
            P="pressure", T="temperature", S="entropy", V="volume", X="composition"
        )
        for axis, count in zip(source.get("diagram_type", "PT")[::-1], SEEDS):
            setattr(settings, names[axis] + "_seeds", count)
        settings.verbose = VERBOSE
        patches = []
        for label, region in windows.items():
            print(f"Calculating {label}: x={region[0]}, y={region[1]}", flush=True)
            patch = bm.calculate_phase_diagram_patch(source, phases, region, settings)
            patch = save(
                patch,
                OUTPUT_DIR / (label.lower().replace(" ", "_") + "_patch"),
                PLOT,
                metadata=source,
            )
            patches.append((label, patch))
    if MERGE:
        merged, reports = source, []
        for label, patch in patches:
            merged, report = bm.merge_phase_diagrams(merged, patch, phases=phases)
            reports.extend(f"{label}: {message}" for message in report)
        save(merged, OUTPUT_DIR / "final", PLOT)
        (OUTPUT_DIR / "merge_report.txt").write_text(
            "\n".join(reports) + "\n", encoding="utf-8"
        )
        print(*reports, sep="\n")


if __name__ == "__main__":
    main()
