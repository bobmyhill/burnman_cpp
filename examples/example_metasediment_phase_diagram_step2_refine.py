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

"""Step 2: refine a saved metasediment diagram to 101 by 201 axis points.

Run example_metasediment_phase_diagram.py first, then run this script without
arguments. Edit SOURCE, MODELS and RESOLUTION below to refine another diagram.
Resolved diagrams retain their connections and labels while adding solved
points. Unresolved diagrams also run recovery; spacing failures are reported.
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


def main():
    saved = bm.load_phase_diagram(SOURCE)
    result = refine_saved(
        saved,
        load_example(MODELS).candidate_phases(),
        resolution=RESOLUTION,
        verbose=True,
        **DISPLAY,
    )
    return write_phase_diagram(
        result,
        OUTPUT_DIR / "phase_diagram",
        metadata=saved,
        plot=partial(plot_phase_diagram_example, **DISPLAY),
    )


if __name__ == "__main__":
    main()
