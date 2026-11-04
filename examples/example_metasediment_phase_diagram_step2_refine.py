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
This script is a follow-up to example_metasediment_phase_diagram.py.

It refines the metasediment phase diagram created by that script,
refining each phase boundary so that points are sampled where they
intersect a grid of 101 pressures and 201 temperatures.

Resolved diagrams retain their original connections and labels.
Any calculation failures are reported.
"""

from functools import partial
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.tools.phase_diagram_helpers import (
    load_example,
    plot_phase_diagram_example,
    refine_saved,
    write_phase_diagram,
)

EXAMPLES = Path(__file__).resolve().parent
SOURCE = EXAMPLES / "metasediment_phase_diagram_output/metasediment_phase_diagram.json"
MODELS = EXAMPLES / "example_metasediment_phase_diagram.py"
OUTPUT_DIR = EXAMPLES / "refined_phase_diagram_output"
RESOLUTION = (101, 201)
DISPLAY = dict(pressure_unit="kbar", temperature_unit="C")


if __name__ == "__main__":
    saved = bm.load_phase_diagram(SOURCE)
    result = refine_saved(
        saved,
        load_example(MODELS).candidate_phases(),
        resolution=RESOLUTION,
        verbose=True,
        **DISPLAY,
    )
    write_phase_diagram(
        result,
        OUTPUT_DIR / "phase_diagram",
        metadata=saved,
        plot=partial(plot_phase_diagram_example, **DISPLAY),
    )
