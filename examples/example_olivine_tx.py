#!/usr/bin/env python3
"""Olivine-polymorph T-X diagram at 14 GPa, from 1000 to 3000 K.

X is the molar fayalite fraction Fe/(Mg+Fe). The SLB11 candidate set is
restricted to olivine, wadsleyite and ringwoodite; melting is not included.
The P-X example supplies the shared composition, phases, settings and outputs.
All thermodynamics and phase-boundary continuation run in C++.

Run: python examples/example_olivine_tx.py
Use --quick for fewer seeds, or --no-plots to save only the resumable JSON.
"""

from functools import partial

from example_olivine_px import calculate as calculate_section, main as section_main

calculate = partial(calculate_section, diagram="TX")
main = partial(section_main, diagram="TX")


if __name__ == "__main__":
    main()
