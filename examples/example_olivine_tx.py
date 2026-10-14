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

"""Olivine-polymorph T-X diagram at 14 GPa, from 1000 to 3000 K.

X is the molar fayalite fraction Fe/(Mg+Fe). The SLB11 candidate set is
restricted to olivine, wadsleyite and ringwoodite; melting is not included.
The P-X example supplies the shared composition, phases, settings and outputs.
All thermodynamics and phase-boundary continuation run in C++.

Run: python examples/example_olivine_tx.py
Use --quick for fewer seeds, or --no-plots to save only the resumable JSON.
"""

from functools import partial
from pathlib import Path

from burnman_cpp.tools.phase_diagram_helpers import load_example

olivine = load_example(Path(__file__).with_name("example_olivine_px.py"))
calculate = partial(olivine.calculate, diagram="TX")
main = partial(olivine.main, diagram="TX")


if __name__ == "__main__":
    main()
